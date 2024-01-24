import itertools
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import tqdm

import data
import modules
import utils

import einops
from torch_scatter import scatter_mean
from torch_scatter.composite import scatter_std

from pytorch3d.ops.marching_cubes import marching_cubes
import pytorch3d
import skimage

class FineRecon(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        img_feature_dim = config.img_feature_dim
        extra_feats = 7
        self.total_feats = img_feature_dim + extra_feats
        self.cnn2d = modules.Cnn2d(out_dim=img_feature_dim)
        self.fusion = modules.FeatureFusion(
            in_c=self.total_feats,
            include_variance=self.config.feature_volume.append_var,
        )
        self.voxel_feat_dim = self.fusion.out_c

        # just a shorthand
        self.dg = self.config.depth_guidance

        self.perform_point_backprojection = self.config.point_backprojection.enabled

        if self.dg.tsdf_fusion_channel:
            self.voxel_feat_dim += 1
        
        if config.planar_feature_encoder.enabled:
            self.planar_feature_in_channels = img_feature_dim + 4 # CNN2D out + 3 normal channels + 1 depth
            if config.improved_depth.enabled:
                self.planar_feature_in_channels = img_feature_dim * (2 if config.planar_feature_encoder.append_var else 1)
                
            if config.planar_feature_encoder.append_var:
                self.planar_feature_in_channels *=2
            self.triplane_cnns = []
            self.cnn_xy = modules.PlanarFeatureCNN2D(self.planar_feature_in_channels,config.planar_feature_encoder.channels)
            self.cnn_xz = modules.PlanarFeatureCNN2D(self.planar_feature_in_channels,config.planar_feature_encoder.channels)
            self.cnn_yz = modules.PlanarFeatureCNN2D(self.planar_feature_in_channels,config.planar_feature_encoder.channels)
            self.triplane_cnns.append(self.cnn_xy)
            self.triplane_cnns.append(self.cnn_xz)
            self.triplane_cnns.append(self.cnn_yz)
            #self.triplane_cnn = modules.PlanarFeatureCNN2D(self.planar_feature_in_channels,config.planar_feature_encoder.channels)
        if config.feature_volume.enabled:
            if config.feature_volume.use_2dcnn:
                self.fv_mlp = modules.FeatureVolumeMLP(in_c=self.voxel_feat_dim)
                self.fv_cnn2d = modules.FeatureVolumeCNN2D(in_c = self.config.feature_volume.n_voxels[-1]) #use height (z) axis as channel
            else:
                self.cnn3d = modules.Cnn3d(in_c=self.voxel_feat_dim)
        surface_pred_input_dim = occ_pred_input_dim = 0
        if config.feature_volume.enabled:
            surface_pred_input_dim += 1 if config.feature_volume.use_2dcnn else self.cnn3d.out_c
            occ_pred_input_dim += 1 if config.feature_volume.use_2dcnn else self.cnn3d.out_c
        
        if self.perform_point_backprojection:
            self.cnn2d_pb_out_dim = img_feature_dim
            self.cnn2d_pb = modules.Cnn2d(out_dim=self.cnn2d_pb_out_dim)
            self.point_fusion = modules.FeatureFusion(
                in_c=self.total_feats,
                include_variance=self.config.point_backprojection.append_var,
            )
            self.point_feat_mlp = torch.nn.Sequential(
                modules.ResBlock1d(self.point_fusion.out_c),
                modules.ResBlock1d(self.point_fusion.out_c),
            )
            surface_pred_input_dim += self.point_fusion.out_c
            if self.dg.tsdf_fusion_channel:
                surface_pred_input_dim += 1
        if config.planar_feature_encoder.enabled:
            agg = config.planar_feature_encoder.aggregation
            multiplier = 3 if (agg == "concat") else 2 if agg == "mean_var" else 1
            surface_pred_input_dim += 3 if agg=="dot_prod" else config.planar_feature_encoder.channels * multiplier
            occ_pred_input_dim += 3 if agg=="dot_prod" else config.planar_feature_encoder.channels * multiplier

        
        self.surface_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(surface_pred_input_dim, 32, 1),
            modules.ResBlock1d(32),
            modules.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )
        self.occ_predictor = torch.nn.Sequential(
            torch.nn.Conv1d(occ_pred_input_dim, 32, 1),
            modules.ResBlock1d(32),
            modules.ResBlock1d(32),
            torch.nn.Conv1d(32, 1, 1),
        )

        if self.config.do_prediction_timing:
            self.init_time = 0
            self.per_view_time = 0
            self.final_step_time = 0
            self.n_final_steps = 0
            self.n_views = 0
            self.n_inits = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda epoch: 1
            if self.global_step < (self.config.steps - self.config.finetune_steps)
            else 0.1,
            verbose=True,
        )
        return [opt], [sched]

    def tsdf_fusion(self, pred_depth_imgs, poses, K_pred_depth, input_coords):
        depth, valid, z = utils.sample_posed_images(
            pred_depth_imgs[:, :, None],
            poses,
            K_pred_depth,
            input_coords,
            mode="nearest",
            return_z=True,
        )
        depth = depth.squeeze(2)
        valid.masked_fill_(depth == 0, False)
        margin = 3 * self.config.voxel_size
        tsdf = torch.clamp(z - depth, -margin, margin) / margin
        valid &= tsdf < 0.999
        tsdf.masked_fill_(~valid, 0)
        tsdf = torch.sum(tsdf, dim=1)
        weight = torch.sum(valid, dim=1)
        tsdf /= weight
        return tsdf, weight

    def sample_point_features_by_linear_interp(
        self, coords, voxel_feats, voxel_valid, grid_origin
    ):
        """
        coords: BN3
        voxel_feats: BFXYZ
        voxel_valid: BXYZ
        grid_origin: B3
        """
        crop_size_m = (
            torch.tensor(voxel_feats.shape[2:], device=self.device)
            * self.config.voxel_size
        )
        grid = (
            coords - grid_origin[:, None] + self.config.voxel_size / 2
        ) / crop_size_m * 2 - 1
        point_valid = (
            torch.nn.functional.grid_sample(
                voxel_valid[:, None].float(),
                grid[:, None, None, :, [2, 1, 0]],
                align_corners=False,
                mode="nearest",
                padding_mode="zeros",
            )[:, 0, 0, 0]
            > 0.5
        )

        point_feats = torch.nn.functional.grid_sample(
            voxel_feats,
            grid[:, None, None, :, [2, 1, 0]],
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        )[:, :, 0, 0]
        return point_feats, point_valid

    def augment_depth_inplace(self, batch):
        n_views = batch["pred_depth_imgs"].shape[1]
        n_augment = n_views // 2

        for i in range(len(batch["pred_depth_imgs"])):
            j = np.random.choice(
                batch["pred_depth_imgs"].shape[1], size=n_augment, replace=False
            )
            scale = torch.rand(len(j), device=self.device) * 0.2 + 0.9
            batch["pred_depth_imgs"][i, j] *= scale[:, None, None]

    def compute_loss(
        self,
        tsdf_logits,
        occ_logits,
        gt_tsdf,
        gt_occ,
        occupancy_mask,
        surface_mask,
    ):
        occ_loss_mask = (~gt_occ.isnan()) & occupancy_mask
        tsdf_loss_mask = (gt_occ > 0.5) & (~gt_tsdf.isnan()) & surface_mask

        occ_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            occ_logits[occ_loss_mask], gt_occ[occ_loss_mask]
        )

        loss = occ_loss
        if tsdf_loss_mask.sum() > 0:
            tsdf_loss = torch.nn.functional.l1_loss(
                utils.log_transform(torch.tanh(tsdf_logits[tsdf_loss_mask])),
                utils.log_transform(gt_tsdf[tsdf_loss_mask]),
            )
            loss += tsdf_loss
        else:
            tsdf_loss = torch.tensor(torch.nan)

        return loss, tsdf_loss, occ_loss

    def step(self, batch):
        if self.training and self.dg.depth_scale_augmentation:
            self.augment_depth_inplace(batch)
        rgb_imgs = batch["rgb_imgs"]
        batch_size, n_imgs, _, imheight, imwidth = rgb_imgs.shape
        rgb_imsize = (imheight, imwidth)
        img_feats = self.cnn2d(rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth))
        img_feats = img_feats.view(batch_size, n_imgs, *img_feats.shape[1:])
        if self.config.feature_volume.enabled:
            voxel_feats, voxel_valid = utils.sample_voxel_feats_(
                poses=batch["poses"],
                xyz=batch["input_coords"],
                rgb_imsize=rgb_imsize,
                img_feats=img_feats,
                K_rgb=batch["K_color"][:, None],
                depth_imgs=batch["pred_depth_imgs"],
                K_depth=batch["K_pred_depth"][:, None],
                normal_imgs=batch["pred_normal_imgs"],
                K_normal=batch["K_pred_normal"][:, None],
            )
            voxel_feats = self.fusion(voxel_feats, voxel_valid)
            voxel_valid = voxel_valid.sum(dim=1) > 1

        if self.dg.tsdf_fusion_channel:
            tsdf, weight = self.tsdf_fusion(
                batch["pred_depth_imgs"],
                batch["poses"],
                batch["K_pred_depth"][:, None],
                batch["input_coords"],
            )
            tsdf.masked_fill_(weight == 0, 1)
            if self.config.feature_volume.enabled:
                voxel_feats = torch.cat((voxel_feats, tsdf[:, None]), dim=1)
        if self.config.feature_volume.enabled:
            if self.config.feature_volume.use_2dcnn:
                batch_size, channels, width, depth, height = voxel_feats.shape
                voxel_f = voxel_feats.view(-1, channels)
                voxel_f = self.fv_mlp(voxel_f)
                voxel_f = voxel_f.view(batch_size,width, depth, height,-1).squeeze(-1)
                voxel_f = einops.rearrange(voxel_f, "B W D H -> B H W D") # we want the height as channel for CNN, CNN expects B C W H, so move height dimension to channel
                voxel_f = self.fv_cnn2d(voxel_f)
                voxel_f = einops.rearrange(voxel_f, "B H W D -> B W D H") # Get it back
                voxel_feats = voxel_f.unsqueeze(1) # Make it a one-dimensional channel
            else :
                voxel_feats = self.cnn3d(voxel_feats, voxel_valid)


        if self.config.planar_feature_encoder.enabled:
            min_bound = batch["output_coords"].min(dim=1)[0]
            max_bound = batch["output_coords"].max(dim=1)[0]
            scale = (max_bound - min_bound) / 2 # /2 because the center is 0.5,0.5,0.5 (in middle not corner)
            if self.config.improved_depth.enabled:
                params={}
                # build mesh from tsdf
                verts = []
                faces = []
                ## potential bottlneck
                for i in range(batch_size):
                    tsdf_b = tsdf[i]; weight_b = weight[i] ; origin = batch["crop_center"][i] - batch["crop_size_m"][i]/2
                    mask = weight_b > 0
                    mask = mask.cpu().numpy()
                    verts_, faces_, _, _ = skimage.measure.marching_cubes(tsdf_b.clone().cpu().numpy(), level=0.5 ,mask=mask)
                    faces_ = faces_[~np.any(np.isnan(verts_[faces_]), axis=(1, 2))]
                    verts_ = verts_ * self.config.voxel_size + origin.cpu().numpy()
                    verts_ = torch.tensor(verts_.copy(),device=self.device,dtype=torch.float)
                    verts.append(verts_)
                    faces_ = torch.tensor(faces_.copy(),device=self.device,dtype=torch.long)
                    faces.append(faces_)
                meshes = pytorch3d.structures.Meshes(verts, faces)
                # sample mesh
                xyz_mesh_samples = pytorch3d.ops.sample_points_from_meshes(meshes,num_samples=self.config.improved_depth.n_samples)
                # build planar_features
                plane_reso = self.config.planar_feature_encoder.plane_resolution
                append_v = self.config.planar_feature_encoder.append_var

                params ={}
                params["poses"] = batch["poses"]
                params["img_feats"] = img_feats
                params["K_rgb"] = batch["K_color"][:, None]
                params["xyz"] = xyz_mesh_samples
                params["crop_center"] = batch["crop_center"]
                params["crop_rotation"] = batch["crop_rotation"]
                
                params["crop_size_m"] = scale
                params["rgb_imsize"] = rgb_imsize
                params["poses"] = batch["poses"]

                planar_features = utils.build_planar_feature_encoder_from_pc(params,self.triplane_cnns,self.device,append_v,plane_reso)#,save_points=True,filename=f"debug/xyz_norm_{batch['scan_name'][0]}_build_pc.csv")

            else:
                params ={}
                params["poses"] = batch["poses"]
                params["depth_imgs"] = batch["pred_depth_imgs"]
                params["img_feats"] = img_feats
                params["K_rgb"] = batch["K_color"][:, None]
                params["K_depth"] = batch["K_pred_depth"][:, None]
                params["normal_imgs"] = batch["pred_normal_imgs"]
                params["K_normal"] = batch["K_pred_normal"][:, None]
                params["crop_center"] = batch["crop_center"]
                params["crop_rotation"] = batch["crop_rotation"]
                params["crop_size_m"] = scale
                plane_reso = self.config.planar_feature_encoder.plane_resolution
                append_v = self.config.planar_feature_encoder.append_var
                planar_features = utils.build_triplane_feature_encoder(params,self.triplane_cnns,self.device,append_v,plane_reso)#,save_points=True,filename=f"debug/xyz_norm_{batch['scan_name'][0]}_build.csv")

        

        """
        interpolate the features to the points where we have GT tsdf
        """
        surface_prediction_feats = []
        occupancy_prediction_feats = []
        
        t = batch["crop_center"]
        R = batch["crop_rotation"]
        coords = batch["output_coords"]
        occupancy_mask = torch.ones_like(batch["gt_occ"]).bool()
        surface_prediction_mask = torch.ones_like(batch["gt_occ"]).bool()

        with torch.autocast(enabled=False, device_type=self.device.type):
            coords_local = (coords - t[:, None]) @ R
        coords_local += batch["crop_size_m"][:, None] / 2
        origin = torch.zeros_like(batch["gt_origin"])
        if self.config.feature_volume.enabled:
            (
                coarse_point_feats,
                coarse_point_valid,
            ) = self.sample_point_features_by_linear_interp(
                coords_local, voxel_feats, voxel_valid, origin
            )
            occupancy_prediction_feats.append(coarse_point_feats)
            surface_prediction_feats.append(coarse_point_feats)
            occupancy_mask = coarse_point_valid 
            surface_prediction_mask = coarse_point_valid

        if self.perform_point_backprojection:
            fine_img_feats = self.cnn2d_pb(
                rgb_imgs.view(batch_size * n_imgs, 3, imheight, imwidth)
            )
            fine_img_feats = fine_img_feats.view(
                batch_size, n_imgs, *fine_img_feats.shape[1:]
            )
            (
                fine_point_feats,
                fine_point_valid,
            ) = utils.sample_voxel_feats_(
                poses=batch["poses"],
                xyz=batch["output_coords"],
                rgb_imsize=rgb_imsize,
                img_feats=fine_img_feats,
                K_rgb=batch["K_color"][:, None],
                depth_imgs=batch["pred_depth_imgs"],
                K_depth=batch["K_pred_depth"][:, None],
                normal_imgs=batch["pred_normal_imgs"],
                K_normal=batch["K_pred_normal"][:, None],
            )
            fine_point_feats = self.point_fusion(
                fine_point_feats[..., None, None], fine_point_valid[..., None, None]
            )[..., 0, 0]
            fine_point_valid = (fine_point_valid.any(dim=1))
            fine_point_feats = self.point_feat_mlp(fine_point_feats)
            surface_prediction_feats.append(fine_point_feats)
            surface_prediction_mask = fine_point_valid

            if self.dg.tsdf_fusion_channel:
                tsdf, weight = self.tsdf_fusion(
                    batch["pred_depth_imgs"],
                    batch["poses"],
                    batch["K_pred_depth"][:, None],
                    batch["output_coords"],
                )
                tsdf.masked_fill_(weight == 0, 1)
                surface_prediction_feats.append(tsdf[:, None])
        if self.config.planar_feature_encoder.enabled:
            fine_planar_features = utils.sample_planar_features(planar_features,batch["output_coords"],params=params,aggregation=self.config.planar_feature_encoder.aggregation)#,save_points=True,filename=f"debug/xyz_norm_{batch['scan_name'][0]}_sample.csv")
            surface_prediction_feats.append(fine_planar_features)
            occupancy_prediction_feats.append(fine_planar_features)

            

        surface_prediction_feats = torch.cat(surface_prediction_feats, dim=1)
        occupancy_prediction_feats = torch.cat(occupancy_prediction_feats,dim=1)
        tsdf_logits = self.surface_predictor(surface_prediction_feats).squeeze(1)
        occ_logits = self.occ_predictor(occupancy_prediction_feats).squeeze(1)

        




        loss, tsdf_loss, occ_loss = self.compute_loss(
            tsdf_logits,
            occ_logits,
            batch["gt_tsdf"],
            batch["gt_occ"],
            occupancy_mask,
            surface_prediction_mask,
        )
        if self.config.save_train_mesh:

            #import ipdb; ipdb.set_trace()
            oc_idx = batch["oc_idx"]
            oc_shape = batch["oc_shape"]
            oc_shape = (int(oc_shape[0]),int(oc_shape[1]),int(oc_shape[2]))
            
            occ_loss_mask = (~batch["gt_occ"].isnan()) & occupancy_mask
            tsdf_loss_mask = (batch["gt_occ"] > 0.5) & (~batch["gt_tsdf"].isnan()) & surface_prediction_mask
            mask = torch.zeros(oc_shape).bool()
            mask.view(-1)[oc_idx] = (occ_loss_mask.cpu() & tsdf_loss_mask.cpu())

            export_tsdf = torch.ones(oc_shape)

            tsdf_ = utils.log_transform(torch.tanh(tsdf_logits.cpu().float()))
            export_tsdf.view(-1)[oc_idx] = tsdf_ # or batch["gt_tsdf"]

            verts_, faces_, _, _ = skimage.measure.marching_cubes(export_tsdf.detach().cpu().numpy(), level=0.5 ,mask=mask.numpy())
            faces_ = faces_[~np.any(np.isnan(verts_[faces_]), axis=(1, 2))]
            verts_ = torch.tensor(verts_.copy(),device=self.device,dtype=torch.float)
            faces_ = torch.tensor(faces_.copy(),device=self.device,dtype=torch.long)
            meshes = pytorch3d.structures.Meshes([verts_], [faces_])
            from pytorch3d.io import IO
            ptio = IO()
            ptio.save_mesh(meshes,f"debug/{batch['scan_name'][0]}_train_{loss:.2f}.ply")


            export_tsdf = torch.ones(oc_shape)

            tsdf_ = utils.log_transform(batch["gt_tsdf"].cpu().float())
            export_tsdf.view(-1)[oc_idx] = tsdf_ # or batch["gt_tsdf"]
            
            verts_, faces_, _, _ = skimage.measure.marching_cubes(export_tsdf.numpy(), level=0.5 ,mask=mask.numpy())
            faces_ = faces_[~np.any(np.isnan(verts_[faces_]), axis=(1, 2))]
            verts_ = torch.tensor(verts_.copy(),device=self.device,dtype=torch.float)
            faces_ = torch.tensor(faces_.copy(),device=self.device,dtype=torch.long)
            meshes = pytorch3d.structures.Meshes([verts_], [faces_])
            from pytorch3d.io import IO
            ptio = IO()
            ptio.save_mesh(meshes,f"debug/{batch['scan_name'][0]}_train_{loss:.2f}_gt.ply")


            #utils.log_transform(gt_tsdf[tsdf_loss_mask])
        outputs = {
            "loss": loss,
            "tsdf_loss": tsdf_loss,
            "occ_loss": occ_loss,
            "tsdf_logits": tsdf_logits,
            "occ_logits": occ_logits,
        }
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.step(batch)

        logs = {}
        for k in ["loss", "tsdf_loss", "occ_loss"]:
            logs[f"loss_train/{k}"] = outputs[k].item()

        logs["lr"] = self.optimizers().param_groups[0]["lr"]

        self.log_dict(logs, rank_zero_only=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.step(batch)

        batch_size = batch["input_coords"].shape[0]
        assert batch_size == 1, "validation step assumes val batch size == 1"

        logs = {}
        for k in ["loss", "tsdf_loss", "occ_loss"]:
            logs[f"loss_val/{k}"] = outputs[k].item()

        self.log_dict(logs, batch_size=batch_size, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return
        if self.current_epoch % 10 != 0:
           return

        # every 10 epochs run inference on the first test scan
        loader = self.predict_dataloader(first_scan_only=True)
        for i, batch in enumerate(tqdm.tqdm(loader, desc="prediction", leave=False)):
            for k in batch:
                if k in self.transfer_keys:
                    batch[k] = batch[k].to(self.device)
            self.predict_step(batch, i)
        self.predict_cleanup()
        torch.cuda.empty_cache()

    def predict_cleanup(self):
        del self.global_coords
        if self.config.feature_volume.enabled:
            del self.M
            del self.running_count

        del self.keyframe_rgb
        del self.keyframe_pose

        del self.keyframe_depth
        del self.keyframe_normal
        if self.dg.tsdf_fusion_channel:
            del self.running_tsdf
            del self.running_tsdf_weight
        if self.config.planar_feature_encoder.enabled:
            del self.sparse_points_coords_norm
            del self.sparse_points_feats

    def predict_init(self, batch):
        # setup before starting inference on a new scan

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()

        vox4 = self.config.voxel_size * 4
        minbound = batch["gt_origin"][0]
        maxbound = batch["gt_maxbound"][0].float()
        maxbound = (torch.ceil((maxbound - minbound) / vox4) - 0.001) * vox4 + minbound

        x = torch.arange(
            minbound[0], maxbound[0], self.config.voxel_size, dtype=torch.float32
        )
        y = torch.arange(
            minbound[1], maxbound[1], self.config.voxel_size, dtype=torch.float32
        )
        if self.config.feature_volume.enabled and self.config.feature_volume.use_2dcnn: # when using 2D CNN for feature volume, we need to take the same voxel size as train, since the filters of the 2DCNN are weighted
            z = torch.arange(
                minbound[1],(minbound[2]+ (self.config.voxel_size*self.config.feature_volume.n_voxels[2])),self.config.voxel_size,dtype=torch.float32
            )
            z = z[:self.config.feature_volume.n_voxels[2]]
        else:
            z = torch.arange(
                minbound[2], maxbound[2], self.config.voxel_size, dtype=torch.float32
            )
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        self.global_coords = torch.stack((xx, yy, zz), dim=-1).to(self.device)
        nvox = xx.shape
        if self.config.feature_volume.enabled:
            self.running_count = torch.zeros(nvox, dtype=torch.float32, device=self.device)
            self.M = torch.zeros(
                (self.fusion.out_c, *nvox),
                dtype=torch.float32,
                device=self.device,
            )

        self.keyframe_rgb = []
        self.keyframe_pose = []

        self.keyframe_depth = []
        self.keyframe_normal = []
        if self.dg.tsdf_fusion_channel:
            self.running_tsdf = torch.zeros(
                nvox, dtype=torch.float32, device=self.device
            )
            self.running_tsdf_weight = torch.zeros(
                nvox, dtype=torch.int32, device=self.device
            )
        if self.config.planar_feature_encoder.enabled:
            self.sparse_points_coords_norm = []
            self.sparse_points_feats = []

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.init_time += t1 - t0
            self.n_inits += 1

    def predict_per_view(self, batch):
        # for each iamge, update the feature volume or add the projected points with features to the list

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()


        batch_size, n_imgs, _, imheight, imwidth = batch["rgb_imgs"].shape
        imsize = imheight, imwidth
        assert batch_size == 1 and n_imgs == 1
        img_feats = self.cnn2d(
            batch["rgb_imgs"].view(batch_size * n_imgs, 3, imheight, imwidth)
        )
        img_feats = img_feats.view(batch_size, n_imgs, *img_feats.shape[1:])

        uv, z, valid = utils.project(
            self.global_coords[None],
            batch["poses"][None],
            batch["K_color"][None],
            imsize,
        )
        valid = valid[0, 0]
        coords = self.global_coords[valid][None, None, None]
        if self.dg.tsdf_fusion_channel:
            tsdf, tsdf_weight = self.tsdf_fusion(
                batch["pred_depth_imgs"],
                batch["poses"][None],
                batch["K_pred_depth"][:, None],
                coords,
            )
            tsdf = tsdf[0, 0, 0]
            tsdf_weight = tsdf_weight[0, 0, 0]
            tsdf.masked_fill_(tsdf_weight == 0, 0)

            old_count = self.running_tsdf_weight[valid]
            self.running_tsdf_weight[valid] += tsdf_weight
            new_count = self.running_tsdf_weight[valid]
            denom = new_count + (new_count == 0)
            self.running_tsdf[valid] = (
                tsdf / denom + (old_count / denom) * self.running_tsdf[valid]
            )
        if self.config.feature_volume.enabled:
            (
                img_voxel_feats,
                img_voxel_valid,
            ) = utils.sample_voxel_feats_(
                poses=batch["poses"][None],
                xyz=coords,
                rgb_imsize=imsize,
                img_feats=img_feats,
                K_rgb=batch["K_color"][None],
                depth_imgs=batch["pred_depth_imgs"],
                K_depth=batch["K_pred_depth"][None],
                normal_imgs=batch["pred_normal_imgs"],
                K_normal=batch["K_pred_normal"][None],
            )
            


            img_voxel_feats.masked_fill_(~img_voxel_valid[:, :, None], 0)

            old_count = self.running_count[valid].clone()
            self.running_count[valid] += img_voxel_valid[0, 0, 0, 0]
            new_count = self.running_count[valid]
            x = img_voxel_feats[0, 0, :, 0, 0]

            if self.config.feature_volume.append_var:
                c = (
                    self.total_feats
                )  # Assuming the first half is mean, and the second is variance
                valid_mean = self.M[:c, valid]
                valid_var = self.M[c:, valid]

                # Online mean calculation
                new_mean = x / new_count[None] + (old_count / new_count)[None] * valid_mean
                self.M[:c, valid] = new_mean

                # Masking the invalid counts for mean
                self.M[:c].masked_fill_(self.running_count[None] == 0, 0)

                # Online variance calculation based on the updated mean
                delta = x - new_mean
                new_var = valid_var + delta * (x - new_mean - delta)
                self.M[c:, valid] = new_var
                if batch["final_frame"][0]:
                    # to get the true variance, we need to divide the var component of M by the running count in the last frame
                    # update variance part
                    self.M[c:, valid] = self.M[c:, valid] / self.running_count[valid]

                # Masking the invalid counts for variance
                self.M[c:].masked_fill_(self.running_count[None] <= 1, 0)

            else:
                old_m = self.M[:, valid]
                new_m = x / new_count[None] + (old_count / new_count)[None] * old_m
                self.M[:, valid] = new_m
                self.M.masked_fill_(self.running_count[None] == 0, 0)

       
        
        if self.config.planar_feature_encoder.enabled and not self.config.improved_depth.enabled:
            poses = batch["poses"]
            depth_imgs = batch["pred_depth_imgs"]
            img_feats = img_feats
            K_depth = batch["K_pred_depth"][:, None]
            normal_imgs = batch["pred_normal_imgs"]
            crop_center= batch["gt_origin"]
            crop_size_m= (batch["gt_maxbound"] - batch["gt_origin"])
            
            B,Nv,h_depth_imgs,w_depth_imgs = depth_imgs.shape

            # create grids of the 20 depth map image
            # each grid should be (0, 0) to (w - 1, h - 1)
            xs = torch.linspace(0, w_depth_imgs-1, steps=w_depth_imgs)
            ys = torch.linspace(0, h_depth_imgs-1, steps=h_depth_imgs)

            # NOTE: be careful about the indexing
            x,y = torch.meshgrid(xs, ys, indexing='xy')

            x = x[None,None].repeat(B,Nv,1,1).to(self.device) # create b dim and number-of-view dim
            y = y[None,None].repeat(B,Nv,1,1).to(self.device) # create b dim and number-of-view dim

            uv = torch.stack([x,y],dim=-1)
            uv = uv / torch.tensor([w_depth_imgs-1,h_depth_imgs-1]).to(self.device)
            uv = ((uv*2)-1).float() # make uv between -1,1
            points_img_feats = [depth_imgs[:,:,None].float(),normal_imgs.permute(0,1,4,2,3).float(),img_feats.float()]
            fine_sparse_point_feats = utils.sample_image_feats(uv, points_img_feats)



            z = depth_imgs
            x = x * z
            y = y * z # then we get pixel coordinate 
            xyz_pixel = torch.stack([x,y,z], dim=-1) # 1, 20, 192, 256, 3
            # sample point features for these sparse points

            _,_,H,W,_ = xyz_pixel.shape

            
            xyz_cams = torch.inverse(K_depth) @ xyz_pixel.reshape(B,Nv,H*W,-1).permute(0,1,3,2)
            xyz_cams = torch.cat([xyz_cams, torch.ones(B,Nv,1,H*W).to(self.device)], dim=2)


            xyz_world = poses.float() @ xyz_cams


            sparse_point_coords = xyz_world[:,:,:3,:]
            sparse_point_coords = sparse_point_coords.permute(0,1,3,2)
            sparse_point_coords = sparse_point_coords.reshape(B,-1,3)
            c_dim = fine_sparse_point_feats.shape[2]

            # take mean and variance, here I just take mean, after maskin
            #fine_sparse_point_feats = fine_sparse_point_feats.reshape(B,-1,).permute()
            fine_sparse_point_feats = fine_sparse_point_feats.permute(0,1,3,4,2).reshape(B,-1,c_dim)
            depth_valid = z>0
            depth_valid = depth_valid.unsqueeze(-1).reshape(B,-1,1)
            sparse_point_coords_norm = sparse_point_coords.clone()
            sparse_point_coords_norm = sparse_point_coords - crop_center[:,None]
            sparse_point_coords_norm = sparse_point_coords_norm / crop_size_m[:,None]

            valid_in_bound = (sparse_point_coords_norm<=1) & (sparse_point_coords_norm>=-1)
            valid_in_bound = valid_in_bound.all(dim=-1,keepdim=True)

            valid = valid_in_bound & depth_valid
            
            
            #sparse_point_coords_norm = sparse_point_coords_norm[valid.repeat(1,1,3)].reshape(B,-1,3)
            #fine_sparse_point_feats = fine_sparse_point_feats[valid.repeat(1,1,c_dim)].reshape(B,-1,c_dim)

            fine_sparse_point_feats = fine_sparse_point_feats * valid

            self.sparse_points_coords_norm.append(sparse_point_coords_norm)
            self.sparse_points_feats.append(fine_sparse_point_feats)



        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.per_view_time += t1 - t0
            self.n_views += 1

    def predict_final(self, batch):
        coarse_voxel_size = self.config.voxel_size
        fine_voxel_size = self.config.voxel_size / self.config.output_sample_rate
        # final reconstruction: run point back-projection & 3d cnn

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        global_valid = torch.ones(self.global_coords.shape[:-1], dtype=torch.float32, device=self.device).bool()
        if self.config.feature_volume.enabled:
            global_feats = self.M

            global_feats = self.fusion.bn(global_feats[None]).squeeze(0)

            if self.dg.tsdf_fusion_channel:
                self.running_tsdf.masked_fill_(self.running_tsdf_weight == 0, 1)
                extra = self.running_tsdf[None]
                global_feats = torch.cat((global_feats, extra), dim=0)

            if self.config.feature_volume.use_2dcnn:
                batch_size, channels, width, height, depth = global_feats[None].shape
                voxel_f = global_feats[None].view(-1, channels)
                voxel_f = self.fv_mlp(voxel_f)
                voxel_f = voxel_f.view(batch_size,width, height, depth,-1).squeeze(-1)
                voxel_f = einops.rearrange(voxel_f, "B W D H -> B H W D") # we want the height as channel for CNN, CNN expects B C W H, so move height dimension to channel
                voxel_f = self.fv_cnn2d(voxel_f)
                voxel_f = einops.rearrange(voxel_f, "B H W D -> B W D H") # Get it back
                global_feats = voxel_f.unsqueeze(1)

            else:
                global_feats = self.cnn3d(global_feats[None], self.running_count[None] > 0)
                global_valid = global_valid & (self.running_count > 0)

        coarse_spatial_dims = np.array(self.global_coords.shape[:-1])
        fine_spatial_dims = coarse_spatial_dims * self.config.output_sample_rate
        


        if self.config.planar_feature_encoder.enabled:
            if self.config.improved_depth.enabled:
                tsdf_b = self.running_tsdf
                weight_b = self.running_tsdf_weight
                origin = batch["gt_origin"]
                params={}
                # build mesh from tsdf
                verts = []
                faces = []
                ## potential bottlneck
                mask = weight_b > 0
                mask = mask.cpu().numpy()
                verts_, faces_, _, _ = skimage.measure.marching_cubes(tsdf_b.clone().cpu().numpy(), level=0.5 ,mask=mask)
                faces_ = faces_[~np.any(np.isnan(verts_[faces_]), axis=(1, 2))]
                verts_ = verts_ * self.config.voxel_size + origin.cpu().numpy()
                verts_ = torch.tensor(verts_.copy(),device=self.device,dtype=torch.float)
                verts.append(verts_)
                faces_ = torch.tensor(faces_.copy(),device=self.device,dtype=torch.long)
                faces.append(faces_)
                meshes = pytorch3d.structures.Meshes(verts, faces)
                # sample mesh
                xyz_mesh_samples = pytorch3d.ops.sample_points_from_meshes(meshes,num_samples=self.config.improved_depth.n_samples)
                # build planar_features
                plane_reso = self.config.planar_feature_encoder.plane_resolution
                append_v = self.config.planar_feature_encoder.append_var
                rgb_imgs = torch.stack(self.keyframe_rgb)
                poses = torch.stack(self.keyframe_pose)
                n_imgs, _ , imheight, imwidth = rgb_imgs.shape
                img_feats = self.cnn2d(rgb_imgs)
                params ={}
                params["poses"] = poses[None]
                params["img_feats"] = img_feats[None]
                params["K_rgb"] = batch["K_color"][:, None]
                params["xyz"] = xyz_mesh_samples
                params["crop_center"] = origin
                params["crop_size_m"] = (batch["gt_maxbound"] - batch["gt_origin"])
                params["rgb_imsize"] = (imheight, imwidth)
                feature_planes = utils.build_planar_feature_encoder_from_pc(params,self.triplane_cnns,self.device,append_v,plane_reso,range="0:1")#,save_points=True,filename=f"debug/xyz_norm_{batch['scan_name'][0]}_build_pc_pred.csv")
            else:
                plane_resolution = self.config.planar_feature_encoder.plane_resolution
                c_dim = self.planar_feature_in_channels//2
                sparse_point_coords_norm = torch.cat(self.sparse_points_coords_norm,dim=1)
                #utils.points_to_csv(sparse_point_coords_norm,f"debug/xyz_norm_{batch['scan_name'][0]}_build_pred.csv")
                sparse_point_feats = torch.cat(self.sparse_points_feats,dim=1)
                planes = ["xy","xz","yz"]
                feature_planes = []
                for i,plane in enumerate(planes):
                    # convert the sparse points to a unit cube
                    xy = utils.normalize_coordinate(sparse_point_coords_norm.clone(), plane=plane,range="0:1")
                    index = utils.coordinate2index(xy, plane_resolution)
                    fea_plane = sparse_point_feats.new_zeros(1, c_dim, plane_resolution**2)
                    #c = c.permute(0, 2, 1) # B x 512 x T
                    fea_plane = scatter_mean(sparse_point_feats.permute(0,2,1), index, out=fea_plane) # B x 512 x reso^2
                    fea_plane = fea_plane.reshape(1, c_dim, plane_resolution, plane_resolution) # sparce matrix (B x 512 x reso x reso)
                    if self.config.planar_feature_encoder.append_var:
                        var_fea_plane = sparse_point_feats.new_zeros(1, c_dim, plane_resolution**2)
                        var_fea_plane = scatter_std(sparse_point_feats.permute(0,2,1), index, out=var_fea_plane)
                        var_fea_plane = fea_plane.reshape(1, c_dim, plane_resolution, plane_resolution)
                        fea_plane = torch.cat([fea_plane,var_fea_plane],dim=1)
                    # pass through 2D CNN
                    fea_plane = self.triplane_cnns[i](fea_plane)
                    feature_planes.append(fea_plane)
        occ_feats = []
        if self.config.feature_volume.enabled:
            if self.config.feature_volume.use_2dcnn:
                occ_feats.append(einops.rearrange(global_feats,"B C W D H -> B C (W D H)"))
            else:
                occ_feats.append(global_feats.view(1, global_feats.shape[1], -1))
        
        if self.config.planar_feature_encoder.enabled:
            coords = self.global_coords.view(-1, 3).unsqueeze(0)
            coords = coords - batch["gt_origin"][0]
            coords = coords / ((batch["gt_maxbound"][0]) - (batch["gt_origin"][0]))
            # coords is now in the range 0-1
            coarse_planar_features = utils.sample_planar_features(feature_planes, coords,aggregation=self.config.planar_feature_encoder.aggregation)
            occ_feats.append(coarse_planar_features)

        occ_feats = torch.cat(occ_feats, dim=1)
        coarse_occ_logits = self.occ_predictor(
                occ_feats
            ).view(list(coarse_spatial_dims))

        coarse_occ_mask = (coarse_occ_logits > 0) # | True
        coarse_occ_idx = torch.argwhere(coarse_occ_mask)
        n_coarse_vox_occ = len(coarse_occ_idx)

        fine_surface = torch.full(
            tuple(fine_spatial_dims), torch.nan, device="cpu", dtype=torch.float32
        )

       

        x = torch.arange(self.config.output_sample_rate)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
        fine_idx_offset = torch.stack((xx, yy, zz), dim=-1).view(-1, 3).to(self.device)
        fine_offset = (
            fine_idx_offset * fine_voxel_size
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )
        

        coarse_voxel_chunk_size = (2**20) // (self.config.output_sample_rate**3) #coarse voxel chunk size is the number of voxels of the feature volume 
        if self.perform_point_backprojection:
            imheight, imwidth = self.keyframe_rgb[0].shape[1:]
            imsize = imheight, imwidth
            featheight = imheight // 4
            featwidth = imwidth // 4

            keyframe_chunk_size = 32
            highres_img_feats = torch.full(
                (
                    len(self.keyframe_rgb),
                    self.cnn2d_pb_out_dim,
                    featheight,
                    featwidth,
                ),
                torch.nan,
                dtype=torch.float32,
                device="cpu",
            )

            for keyframe_chunk_start in tqdm.trange(
                0,
                len(self.keyframe_rgb),
                keyframe_chunk_size,
                desc="highres img feats",
                leave=False,
            ):
                keyframe_chunk_end = min(
                    keyframe_chunk_start + keyframe_chunk_size,
                    len(self.keyframe_rgb),
                )

                rgb_imgs = torch.stack(
                    self.keyframe_rgb[keyframe_chunk_start:keyframe_chunk_end],
                    dim=0,
                )

                highres_img_feats[
                    keyframe_chunk_start:keyframe_chunk_end
                ] = self.cnn2d_pb(rgb_imgs)

        for coarse_voxel_chunk_start in tqdm.trange(
            0, n_coarse_vox_occ, coarse_voxel_chunk_size, leave=False, desc="chunks"
        ):
            coarse_voxel_chunk_end = min(
                coarse_voxel_chunk_start + coarse_voxel_chunk_size, n_coarse_vox_occ
            )

            chunk_coarse_idx = coarse_occ_idx[
                coarse_voxel_chunk_start:coarse_voxel_chunk_end
            ]
            chunk_coarse_coords = (
                chunk_coarse_idx * coarse_voxel_size + batch["gt_origin"]
            )

            chunk_fine_coords = chunk_coarse_coords[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_coords += fine_offset[None]
            chunk_fine_coords = chunk_fine_coords.view(-1, 3)


            chunk_surface_prediction_feats = []
            if self.config.feature_volume.enabled:
                (
                    chunk_fine_feats,
                    chunk_fine_valid,
                ) = self.sample_point_features_by_linear_interp(
                    chunk_fine_coords,
                    global_feats,
                    global_valid[None],
                    batch["gt_origin"],
                )
                chunk_surface_prediction_feats.append(chunk_fine_feats)

            if self.perform_point_backprojection:
                if self.config.point_backprojection.append_var:
                    fine_bp_feats = torch.zeros(
                        (self.total_feats * 2, len(chunk_fine_coords)),
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    fine_bp_feats = torch.zeros(
                        (self.total_feats, len(chunk_fine_coords)),
                        device=self.device,
                        dtype=torch.float32,
                    )

                counts = torch.zeros(
                    len(chunk_fine_coords), device=self.device, dtype=torch.float32
                )

                if self.dg.tsdf_fusion_channel:
                    fine_tsdf = torch.zeros(len(chunk_fine_coords), device=self.device)
                    fine_tsdf_weights = torch.zeros(
                        len(chunk_fine_coords),
                        device=self.device,
                        dtype=torch.float32,
                    )

                for keyframe_chunk_start in range(
                    0, len(self.keyframe_rgb), keyframe_chunk_size
                ):
                    keyframe_chunk_end = min(
                        keyframe_chunk_start + keyframe_chunk_size,
                        len(self.keyframe_rgb),
                    )

                    chunk_highres_img_feats = highres_img_feats[
                        keyframe_chunk_start:keyframe_chunk_end
                    ].to(self.device)
                    rgb_img_placeholder = torch.empty(
                        1, len(chunk_highres_img_feats), 3, imheight, imwidth
                    )

                    poses = torch.stack(
                        self.keyframe_pose[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )
                    pred_depth_imgs = torch.stack(
                        self.keyframe_depth[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )
                    pred_normal_imgs = torch.stack(
                        self.keyframe_normal[keyframe_chunk_start:keyframe_chunk_end],
                        dim=0,
                    )

                    (
                        _fine_bp_feats,
                        valid,
                    ) = utils.sample_voxel_feats_(
                        poses=poses[None],
                        xyz=chunk_fine_coords[None],
                        rgb_imsize=imsize,
                        img_feats=chunk_highres_img_feats[None],
                        K_rgb=batch["K_color"][:,None],
                        depth_imgs=pred_depth_imgs[None],
                        K_depth=batch["K_pred_depth"][:, None],
                        normal_imgs=pred_normal_imgs[None],
                        K_normal=batch["K_pred_normal"][:,None],
                    )

                    old_counts = counts.clone()
                    current_counts = valid.squeeze(0).sum(dim=0)
                    counts += current_counts

                    

                    if self.config.point_backprojection.append_var:
                        c = self.total_feats
                        m = current_counts
                        mean_m = _fine_bp_feats.squeeze(0).sum(dim=0) / m
                        mean_n = fine_bp_feats[:c]
                        n = old_counts

                        new_mean = (mean_m * m / (m + n)) + (mean_n * n / (m + n))
                        new_mean.masked_fill_(current_counts == 0, 0)
                        fine_bp_feats[:c] = new_mean

                        var_m = ((_fine_bp_feats.squeeze(0) - mean_m) ** 2).sum(
                            dim=0
                        ) / m
                        var_n = fine_bp_feats[c:]

                        new_var = (
                            (m / (m + n) * (var_m + mean_m**2))
                            + (n / (m + n) * (var_n + mean_n**2))
                            - new_mean**2
                        )
                        new_var.masked_fill_(current_counts <= 1, 0)
                        fine_bp_feats[c:] = new_var

                    else:
                        # masked fill for invalid points is not needed here, since it is done in sampling voxel feats
                        denom = torch.clamp_min(counts, 1)
                        _fine_bp_feats = _fine_bp_feats.squeeze(0)
                        _fine_bp_feats /= denom
                        _fine_bp_feats = _fine_bp_feats.sum(dim=0)
                        _fine_bp_feats *= current_counts
                        fine_bp_feats *= old_counts / denom
                        fine_bp_feats += _fine_bp_feats

                    if self.dg.tsdf_fusion_channel:
                        tsdf, weight = self.tsdf_fusion(
                            pred_depth_imgs[None],
                            poses[None],
                            batch["K_pred_depth"][:, None],
                            chunk_fine_coords[None],
                        )
                        tsdf.masked_fill_(weight == 0, 0)

                        old_count = fine_tsdf_weights.clone()
                        fine_tsdf_weights += weight.squeeze(0)
                        new_count = fine_tsdf_weights
                        denom = torch.clamp_min(new_count, 1)
                        fine_tsdf = (
                            tsdf.squeeze(0) / denom + (old_count / denom) * fine_tsdf
                        )
                fine_bp_feats = self.point_fusion.bn(
                    fine_bp_feats[None, ..., None, None]
                )[..., 0, 0]
                fine_bp_feats = self.point_feat_mlp(fine_bp_feats)
                chunk_surface_prediction_feats.append(fine_bp_feats)
                
                if self.dg.tsdf_fusion_channel:
                    fine_tsdf.masked_fill_(fine_tsdf_weights == 0, 1)
                    extra = fine_tsdf[None]
                    chunk_surface_prediction_feats.append(extra[None])
            if self.config.planar_feature_encoder.enabled:
                coords = chunk_fine_coords[None].clone()
                coords = coords - batch["gt_origin"][0]
                coords = coords / ((batch["gt_maxbound"][0]) - (batch["gt_origin"][0]))
                fine_planar_features = utils.sample_planar_features(feature_planes,coords,aggregation=self.config.planar_feature_encoder.aggregation)
                #utils.points_to_csv(chunk_fine_coords[None],f"debug/{batch['scan_name'][0]}-chunk_{coarse_voxel_chunk_start}_fine_coords_.csv")
                chunk_surface_prediction_feats.append(fine_planar_features)
            

            chunk_surface_prediction_feats = torch.cat(chunk_surface_prediction_feats, dim=1)
            chunk_fine_surface_logits = (
                self.surface_predictor(chunk_surface_prediction_feats)[0, 0].cpu().float()
            )
            chunk_fine_idx = chunk_coarse_idx[:, None].repeat(
                1, self.config.output_sample_rate**3, 1
            )
            chunk_fine_idx *= self.config.output_sample_rate
            chunk_fine_idx += fine_idx_offset[None]
            chunk_fine_idx = chunk_fine_idx.view(-1, 3).cpu()

            fine_surface[
                chunk_fine_idx[:, 0],
                chunk_fine_idx[:, 1],
                chunk_fine_idx[:, 2],
            ] = chunk_fine_surface_logits

        torch.tanh_(fine_surface)
        fine_surface *= 0.5
        fine_surface += 0.5

        fine_surface[fine_surface.isnan()] = 1

        if self.config.do_prediction_timing:
            torch.cuda.synchronize()
            t1 = time.time()
            self.final_step_time += t1 - t0
            self.n_final_steps += 1

        os.makedirs(self.logger.save_dir, exist_ok=True)
        name = batch["scan_name"][0]
        step = str(self.global_step).zfill(8)
        name = f"{name}_step_{step}" if self.training else name

        origin = (
            batch["gt_origin"].cpu().numpy()[0]
            - coarse_voxel_size / 2
            + fine_voxel_size / 2
        )

        try:
            pred_mesh = utils.tsdf2mesh(
                fine_surface.numpy(),
                voxel_size=fine_voxel_size,
                origin=origin,
                level=0.5,
            )
        except Exception as e:
            print(e)
        else:
            _ = pred_mesh.export(
                os.path.join(self.logger.save_dir, "outputs", f"{name}.ply")
            )
            np.savez(
                os.path.join(self.logger.save_dir, "outputs", f"{name}.npz"),
                tsdf=fine_surface.numpy(),
                origin=origin,
                voxel_size=fine_voxel_size,
            )

    def predict_step(self, batch, batch_idx):
        if batch["initial_frame"][0]:
            self.predict_init(batch)

        self.predict_per_view(batch)

        if self.perform_point_backprojection:
            # store any frames that are marked as keyframes for later point back-projection
            if batch["keyframe"][0]:
                self.keyframe_rgb.append(batch["rgb_imgs"][0, 0])
                self.keyframe_pose.append(batch["poses"][0])
                self.keyframe_depth.append(batch["pred_depth_imgs"][0, 0])
                self.keyframe_normal.append(batch["pred_normal_imgs"][0,0])

        if batch["final_frame"][0]:
            self.predict_final(batch)

    def on_predict_epoch_end(self, _):
        if self.config.do_prediction_timing:
            per_init_time = self.init_time / self.n_inits
            per_view_time = self.per_view_time / self.n_views
            final_step_time = self.final_step_time / self.n_final_steps

            print("========")
            print("========")
            print(f"per_init_time: {per_init_time:.4f}")
            print(f"per_view_time: {per_view_time:.4f}")
            print(f"final_step_time: {final_step_time:.4f}")
            print("========")
            print("========")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        self.transfer_keys = [
            "input_coords",
            "output_coords",
            "crop_center",
            "crop_rotation",
            "crop_size_m",
            "gt_tsdf",
            "gt_occ",
            "K_color",
            "K_pred_depth",
            "K_pred_normal",
            "rgb_imgs",
            "pred_depth_imgs",
            "pred_normal_imgs",
            "poses",
            "gt_origin",
            "gt_maxbound",
        ]
        self.no_transfer_keys = [
            "scan_name",
            "gt_tsdf_npzfile",
            "keyframe",
            "initial_frame",
            "final_frame",
            "oc_idx",
            "oc_shape",
            "oc_all",
            "oc_all_idx",
            "gt_tsdf_all",
            "gt_occ_all"
        ]

        transfer_batch = {}
        no_transfer_batch = {}
        for k in batch:
            if k in self.transfer_keys:
                transfer_batch[k] = batch[k]
            elif k in self.no_transfer_keys:
                no_transfer_batch[k] = batch[k]
            else:
                print(f"Cannot transfer, key '{k}' not found")
                raise NotImplementedError

        transfer_batch = super().transfer_batch_to_device(
            transfer_batch, device, dataloader_idx
        )
        transfer_batch.update(no_transfer_batch)
        return transfer_batch

    def get_scans(self):
        train_scans, val_scans, test_scans = data.get_scans(
            self.config.dataset_dir,
            self.config.tsdf_dir,
            self.dg.pred_depth_dir,
            self.config.normals_dir
        )
        return train_scans, val_scans, test_scans

    def train_dataloader(self):
        train_scans, _ , test_scans = self.get_scans()
        if self.config.debug_scan:
            train_scans = test_scans
        train_dataset = data.Dataset(
            train_scans,
            self.config.voxel_size,
            self.config.feature_volume.n_voxels,
            self.config.n_views_train,
            random_translation=False,
            random_rotation=False,
            random_view_selection=True,
            image_augmentation=True,
            load_depth=True,
            load_normals=True,
            keyframes_file=self.config.test_keyframes_file,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_device,
            num_workers=self.config.workers_train,
            persistent_workers=self.config.workers_train > 0,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        _, val_scans, test_scans = self.get_scans()
        if self.config.debug_scan:
            val_scans = test_scans
        val_dataset = data.Dataset(
            val_scans,
            self.config.voxel_size,
            self.config.feature_volume.n_voxels,
            self.config.n_views_val,
            random_translation=False,
            random_rotation=False,
            load_depth=True,
            load_normals=True,
            keyframes_file=self.config.test_keyframes_file,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.val_batch_size_per_device,
            num_workers=self.config.workers_val,
        )
        return val_loader

    def predict_dataloader(self, first_scan_only=False):
        _, _, test_scans = self.get_scans()

        if first_scan_only:
            test_scans = test_scans[:1] 
        if self.config.scene_id:
            test_scans = [data.get_scan(
            self.config.scene_id,
            self.config.dataset_dir,
            self.config.tsdf_dir,
            self.dg.pred_depth_dir,
            self.config.normals_dir)]



        predict_dataset = data.InferenceDataset(
            test_scans,
            load_depth=True,
            load_normals=True,
            keyframes_file=self.config.test_keyframes_file,
        )
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=1,
            num_workers=self.config.workers_predict,
        )
        return predict_loader
