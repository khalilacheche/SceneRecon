import glob
import json
import os

import cv2
import numpy as np
import PIL.Image
import scipy.spatial
import torch
import torchvision
import trimesh

import utils

TARGET_RGB_IMG_SIZE = (480, 640)


def get_scans(dataset_dir, tsdf_dir, pred_depth_dir,pred_normal_dir):
    with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
        train_scan_names = sorted(set(f.read().strip().split()))
    with open(os.path.join(dataset_dir, "val.txt"), "r") as f:
        val_scan_names = sorted(set(f.read().strip().split()))
    with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
        test_scan_names = sorted(set(f.read().strip().split()))

    scan_dirs = sorted(
        [f for f in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(f)]
    )
    train_scans = []
    val_scans = []
    test_scans = []
    for scan_dir in scan_dirs:
        scan_name = os.path.basename(scan_dir)
        tsdf_npzfile = os.path.join(tsdf_dir, f"{scan_name}.npz")
        zero_crossings_npzfile = os.path.join(tsdf_dir, f"{scan_name}_zc.npz")
        pose_npyfile = os.path.join(dataset_dir, scan_name, "pose.npy")
        image_dir = os.path.join(dataset_dir, scan_name, "color")
        if os.path.exists(pose_npyfile):
            scan = {
                "scan_dir": scan_dir,
                "scan_name": scan_name,
                "tsdf_npzfile": tsdf_npzfile,
                "zero_crossings_npzfile": zero_crossings_npzfile,
                "pose_npyfile": pose_npyfile,
                "image_dir": image_dir,
                "pred_depth_dir": os.path.join(pred_depth_dir, scan_name),
                "pred_normals_dir": os.path.join(pred_normal_dir,scan_name)
            }

            if scan_name in train_scan_names:
                train_scans.append(scan)
            if scan_name in val_scan_names:
                val_scans.append(scan)
            if scan_name in test_scan_names:
                test_scans.append(scan)
            else:
                pass
    return train_scans, val_scans, test_scans

def get_scan(scan_name,dataset_dir, tsdf_dir, pred_depth_dir,pred_normal_dir):
    tsdf_npzfile = os.path.join(tsdf_dir, f"{scan_name}.npz")
    zero_crossings_npzfile = os.path.join(tsdf_dir, f"{scan_name}_zc.npz")
    pose_npyfile = os.path.join(dataset_dir, scan_name, "pose.npy")
    image_dir = os.path.join(dataset_dir, scan_name, "color")
    scan = {
        "scan_dir": os.path.join(dataset_dir, scan_name),
        "scan_name": scan_name,
        "tsdf_npzfile": tsdf_npzfile,
        "zero_crossings_npzfile": zero_crossings_npzfile,
        "pose_npyfile": pose_npyfile,
        "image_dir": image_dir,
        "pred_depth_dir": os.path.join(pred_depth_dir, scan_name),
        "pred_normals_dir": os.path.join(pred_normal_dir,scan_name)
    }

    return scan

def load_scan(scan, keyframes_file=None):
    scan_dir = scan["scan_dir"]

    rgb_imgfiles = glob.glob(os.path.join(scan["image_dir"], "*.jpg"))
    gt_depth_imgfiles = glob.glob(os.path.join(scan_dir, "depth/*.png"))

    intr_file = os.path.join(scan_dir, "intrinsic_depth.txt")
    K_gt_depth = torch.from_numpy(np.loadtxt(intr_file)[:3, :3]).float()

    depth_imheight, depth_imwidth = load_depth_img(gt_depth_imgfiles[0]).shape
    gt_depth_img_size = (depth_imheight, depth_imwidth)

    actual_rgb_img_height, actual_rgb_img_width, _ = cv2.imread(rgb_imgfiles[0]).shape

    intr_file = os.path.join(scan_dir, "intrinsic_color.txt")
    K_color = torch.from_numpy(np.loadtxt(intr_file)[:3, :3]).float()
    K_color[0] *= TARGET_RGB_IMG_SIZE[1] / actual_rgb_img_width
    K_color[1] *= TARGET_RGB_IMG_SIZE[0] / actual_rgb_img_height

    poses = np.load(scan["pose_npyfile"])
    good_pose = ~np.any(np.isinf(poses), axis=(1, 2))

    frames = {i: {"pose": poses[i]} for i in range(len(poses))}

    for f in rgb_imgfiles:
        i = int(f.split("/")[-1][:-4])
        frames[i]["rgb_imgfile"] = f

    for f in gt_depth_imgfiles:
        i = int(f.split("/")[-1][:-4])
        frames[i]["gt_depth_imgfile"] = f

    frames = {i: frame for i, frame in frames.items() if good_pose[i]}

    if keyframes_file is not None:
        with open(keyframes_file, "r") as f:
            kf_idxs = np.array(json.load(f)[scan["scan_name"]])
            
        frames = {i: frames[i] for i in kf_idxs}

    kf_idx = np.ones(len(frames))

    frame_idxs = sorted(frames.keys())
    poses = torch.from_numpy(np.stack([frames[i]["pose"] for i in frame_idxs]))
    rgb_imgfiles = np.array([frames[i]["rgb_imgfile"] for i in frame_idxs])
    gt_depth_imgfiles = np.array([frames[i]["gt_depth_imgfile"] for i in frame_idxs])

    return (
        rgb_imgfiles,
        gt_depth_imgfiles,
        gt_depth_img_size,
        poses,
        K_color,
        K_gt_depth,
        kf_idx,
    )
def select_views(bbox, poses, K, imsize):
    imheight, imwidth = imsize

    coords = bbox.grid_coords(grid_count=(7, 7, 7)).view(-1, 3)

    _, _, valid = utils.project(
        coords[None], poses[None], K[None, None], (imheight, imwidth)
    )
    valid = valid.squeeze(0)

    dist = torch.norm(poses[:, None, :3, 3] - coords[None], dim=-1)
    return torch.where((valid & (dist < 4)).any(dim=-1))[0]

def load_rgb_imgs(rgb_imgfiles, target_size=None):
    imgs = np.empty((len(rgb_imgfiles), *target_size, 3), dtype=np.uint8)
    for i in range(len(rgb_imgfiles)):
        img = PIL.Image.open(rgb_imgfiles[i])
        if not (target_size[0] == img.height and target_size[1] == img.width):
            imheight, imwidth = target_size
            img = img.resize((imwidth, imheight))
        imgs[i] = img
    return torch.from_numpy(imgs.transpose(0, 3, 1, 2)) / 255


def augment_images_inplace(rgb_imgs):
    a = 0.25
    brightness, contrast, saturation, hue = np.random.uniform(-a, a, size=4)

    for i in range(len(rgb_imgs)):
        rgb_imgs[i] = torchvision.transforms.functional.adjust_brightness(
            rgb_imgs[i], 1 + brightness
        )
        rgb_imgs[i] = torchvision.transforms.functional.adjust_contrast(
            rgb_imgs[i], 1 + contrast
        )
        rgb_imgs[i] = torchvision.transforms.functional.adjust_saturation(
            rgb_imgs[i], 1 + saturation
        )
        rgb_imgs[i] = torchvision.transforms.functional.adjust_hue(rgb_imgs[i], hue)


def load_depth_img(f):
    return cv2.imread(f, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000

def load_normal_img(f):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    normal = img / 255.0
    normal = normal * 2 - 1
    return -normal


def load_depth_imgs(depth_imgfiles, imsize):
    result = np.zeros((len(depth_imgfiles), *imsize), dtype=np.float32)
    for i in range(len(depth_imgfiles)):
        result[i] = load_depth_img(depth_imgfiles[i])
    return torch.from_numpy(result)

def load_normal_imgs(normal_imgfiles,imsize):
    result = np.zeros((len(normal_imgfiles), *imsize), dtype=np.float32)
    for i in range(len(normal_imgfiles)):
        result[i] = load_normal_img(normal_imgfiles[i])
    return torch.from_numpy(result)


def dilate_gt_occ(gt_occ):
    nanmask = gt_occ.isnan()
    gt_occ.masked_fill_(nanmask, 0)
    gt_occ = torch.nn.functional.max_pool3d(
        gt_occ[None, None].float(), 3, stride=1, padding=1
    )[0, 0]

    nanmask[gt_occ > 0.5] = False
    gt_occ.masked_fill_(nanmask, torch.nan)
    return gt_occ


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, scans, load_depth=False, load_normals= False,keyframes_file=None):
        self.scans = scans
        self.load_depth = load_depth
        self.load_normals = load_normals

        self.frames = []
        for scan in self.scans:
            (
                rgb_imgfiles,
                gt_depth_imgfiles,
                gt_depth_img_size,
                poses,
                K_color,
                K_gt_depth,
                kf_idx,
            ) = load_scan(scan, keyframes_file=keyframes_file)

            for i in range(len(rgb_imgfiles)):
                initial_frame = i == 0
                final_frame = i == len(rgb_imgfiles) - 1
                keyframe = kf_idx[i]

                if load_depth:
                    pred_depth_imgfile = os.path.join(
                        scan["pred_depth_dir"],
                        "depth",
                        os.path.basename(rgb_imgfiles[i])[:-4] + ".png",
                    )
                    K_pred_depth = torch.from_numpy(
                        np.loadtxt(
                            os.path.join(scan["pred_depth_dir"], "intrinsic_depth.txt")
                        )[:3, :3]
                    ).float()
                else:
                    pred_depth_imgfile = None
                    K_pred_depth = None

                if load_normals:
                    pred_normal_imgfile = os.path.join(
                        scan["pred_normals_dir"],
                        "normal",
                        os.path.basename(rgb_imgfiles[i])[:-4] + ".png",
                    )
                    K_pred_normal = torch.from_numpy(
                        np.loadtxt(
                            os.path.join(scan["pred_normals_dir"], "intrinsic_depth.txt")
                        )[:3, :3]
                    ).float()
                else:
                    pred_normal_imgfile = None
                    K_pred_normal = None

                self.frames.append(
                    [
                        rgb_imgfiles[i],
                        gt_depth_imgfiles[i],
                        pred_depth_imgfile,
                        pred_normal_imgfile,
                        poses[i],
                        K_color,
                        K_pred_depth,
                        K_pred_normal,
                        scan["tsdf_npzfile"],
                        scan["scan_name"],
                        keyframe,
                        initial_frame,
                        final_frame,
                    ]
                )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        (
            rgb_imgfile,
            gt_depth_imgfile,
            pred_depth_imgfile,
            pred_normal_imgfile,
            pose,
            K_color,
            K_pred_depth,
            K_pred_normal,
            tsdf_npzfile,
            scan_name,
            keyframe,
            initial_frame,
            final_frame,
        ) = self.frames[idx]

        rgb_img = load_rgb_imgs([rgb_imgfile], TARGET_RGB_IMG_SIZE)

        tsdf_npzfile = tsdf_npzfile
        npz = np.load(tsdf_npzfile)

        gt_origin = torch.from_numpy(npz["origin"])
        gt_maxbound = torch.from_numpy(npz["maxbound"])

        result = {
            "rgb_imgs": rgb_img,
            "poses": pose,
            "K_color": K_color,
            "scan_name": scan_name,
            "gt_origin": gt_origin,
            "gt_maxbound": gt_maxbound,
            "keyframe": keyframe,
            "initial_frame": initial_frame,
            "final_frame": final_frame,
        }
        if self.load_depth:
            pred_depth_img = load_depth_img(pred_depth_imgfile)[None]

            result["pred_depth_imgs"] = pred_depth_img
            result["K_pred_depth"] = K_pred_depth
        if self.load_normals:
            pred_normal_img = load_normal_img(pred_normal_imgfile)[None]

            result["pred_normal_imgs"] = pred_normal_img
            result["K_pred_normal"] = K_pred_normal

        return result


class RotatedBoundingBox:
    def __init__(self, edge_lengths, R, t):
        self.edge_lengths = edge_lengths
        self.R = R
        self.t = t  # center of the box

    def grid_coords(self, grid_res=None, grid_count=None):
        if (grid_res is None) + (grid_count is None) != 1:
            raise Exception("one of res or count must be set")

        if grid_res is not None:
            grid_count = (self.edge_lengths / grid_res).round().int()
            x = torch.arange(grid_count[0]) * grid_res - self.edge_lengths[0] / 2
            y = torch.arange(grid_count[1]) * grid_res - self.edge_lengths[1] / 2
            z = torch.arange(grid_count[2]) * grid_res - self.edge_lengths[2] / 2
        elif grid_count is not None:
            x = torch.linspace(
                -self.edge_lengths[0] / 2, self.edge_lengths[0] / 2, grid_count[0]
            )
            y = torch.linspace(
                -self.edge_lengths[1] / 2, self.edge_lengths[1] / 2, grid_count[1]
            )
            z = torch.linspace(
                -self.edge_lengths[2] / 2, self.edge_lengths[2] / 2, grid_count[2]
            )

        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        xyz = torch.stack((xx, yy, zz), dim=-1)

        xyz = xyz @ self.R.T + self.t
        return xyz

    def bounding_box(self):
        corners = 0.5 * torch.tensor(
            [
                [-self.edge_lengths[0], -self.edge_lengths[1], -self.edge_lengths[2]],
                [self.edge_lengths[0], -self.edge_lengths[1], -self.edge_lengths[2]],
                [-self.edge_lengths[0], self.edge_lengths[1], -self.edge_lengths[2]],
                [-self.edge_lengths[0], -self.edge_lengths[1], self.edge_lengths[2]],
                [self.edge_lengths[0], self.edge_lengths[1], -self.edge_lengths[2]],
                [-self.edge_lengths[0], self.edge_lengths[1], self.edge_lengths[2]],
                [self.edge_lengths[0], -self.edge_lengths[1], self.edge_lengths[2]],
                [self.edge_lengths[0], self.edge_lengths[1], self.edge_lengths[2]],
            ]
        )
        corners = corners @ self.R.T + self.t
        minbound = torch.min(corners, dim=0)[0]
        maxbound = torch.max(corners, dim=0)[0]
        return minbound, maxbound

    def contains(self, pts):
        pts = (pts - self.t) @ self.R

        return (
            (pts[:, 0] > -self.edge_lengths[0] / 2)
            & (pts[:, 1] > -self.edge_lengths[1] / 2)
            & (pts[:, 2] > -self.edge_lengths[2] / 2)
            & (pts[:, 0] < self.edge_lengths[0] / 2)
            & (pts[:, 1] < self.edge_lengths[1] / 2)
            & (pts[:, 2] < self.edge_lengths[2] / 2)
        )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scans,
        crop_voxel_size,
        crop_size_nvox,
        n_views,
        random_translation=False,
        random_rotation=False,
        random_view_selection=False,
        image_augmentation=False,
        load_depth=False,
        load_normals=False,
        keyframes_file=None

    ):
        self.scans = scans
        self.n_views = n_views
        self.crop_voxel_size = crop_voxel_size
        self.crop_size_nvox = torch.tensor(crop_size_nvox)
        self.random_translation = random_translation
        self.random_rotation = random_rotation
        self.random_view_selection = random_view_selection
        self.image_augmentation = image_augmentation
        self.load_depth = load_depth
        self.load_normals = load_normals
        self.keyframes_file = keyframes_file

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, scan_idx):
        scan = self.scans[scan_idx]

        (
            rgb_imgfiles,
            gt_depth_imgfiles,
            gt_depth_img_size,
            poses,
            K_color,
            K_gt_depth,
            kf_idx,
        ) = load_scan(scan,self.keyframes_file)

        tsdf_npzfile = scan["tsdf_npzfile"]
        npz = np.load(tsdf_npzfile)

        gt_tsdf = torch.from_numpy(npz["tsdf"])
        gt_origin = torch.from_numpy(npz["origin"])
        gt_voxel_size = np.float32(npz["voxel_size"])
        gt_maxbound = torch.from_numpy(npz["maxbound"])


        crop_size_m = self.crop_voxel_size * self.crop_size_nvox
        if self.random_translation:
            minbound = gt_origin + crop_size_m / 2
            maxbound = gt_maxbound - crop_size_m / 2
            crop_center = torch.tensor(
                [
                    np.random.uniform(minbound[0], maxbound[0]),
                    np.random.uniform(minbound[1], maxbound[1]),
                    np.random.uniform(minbound[2], maxbound[2]),
                ]
            ).float()
        else:
            crop_center = 0.5 * (gt_origin + gt_maxbound).float()

        if self.random_rotation:
            z_angle = np.random.uniform(2 * np.pi)
            r_z = scipy.spatial.transform.Rotation.from_rotvec(
                [0, 0, z_angle]
            ).as_matrix()

            x_angle = np.random.uniform(-3, 3) / 180 * np.pi
            r_x = scipy.spatial.transform.Rotation.from_rotvec(
                [x_angle, 0, 0]
            ).as_matrix()

            crop_rotation = torch.from_numpy(r_z @ r_x).float()
        else:
            crop_rotation = torch.eye(3).float()

        crop_bounds = RotatedBoundingBox(crop_size_m, crop_rotation, crop_center)

        view_inds = select_views(crop_bounds, poses, K_color, TARGET_RGB_IMG_SIZE)
        if len(view_inds) < self.n_views:

            # not enough views made it past view selection
            # choose extras from among the views that were not selected
            all_view_inds = set(range(len(poses)))
            unused_view_inds = list(all_view_inds - set(view_inds.tolist()))
            n_needed = self.n_views - len(view_inds)
            n_sample = min(n_needed, len(unused_view_inds))
            extra_view_inds = np.random.choice(
                unused_view_inds, size=n_sample, replace=False
            )
            view_inds = torch.cat((view_inds, torch.from_numpy(extra_view_inds))).long()

            if len(view_inds) < self.n_views:
                # still not enough. add duplicates
                n_needed = self.n_views - len(view_inds)
                extra_view_inds = np.random.choice(
                    list(all_view_inds), size=n_needed, replace=True
                )
                view_inds = torch.cat((view_inds, torch.from_numpy(extra_view_inds))).long()

        if self.random_view_selection:
            view_inds = np.random.choice(view_inds, size=self.n_views, replace=False)
        else:
            idx = np.round(
                np.linspace(0, len(view_inds), self.n_views, endpoint=False)
            ).astype(int)
            view_inds = view_inds[idx]

        result = {
            "crop_center": crop_bounds.t,
            "crop_rotation": crop_bounds.R,
            "crop_size_m": crop_bounds.edge_lengths,
            "gt_origin": gt_origin,
            "gt_maxbound": gt_maxbound,
            "scan_name": scan["scan_name"],
        }


        rgb_imgs = load_rgb_imgs(rgb_imgfiles[view_inds], TARGET_RGB_IMG_SIZE)
        if self.image_augmentation:
            augment_images_inplace(rgb_imgs)
        result["rgb_imgs"] = rgb_imgs
        result["K_color"] = K_color

        if self.load_depth:
            pred_depth_imgfiles = []
            for rgb_imgfile in rgb_imgfiles[view_inds]:
                pred_depth_imgfiles.append(
                    os.path.join(
                        scan["pred_depth_dir"],
                        "depth",
                        os.path.basename(rgb_imgfile)[:-4] + ".png",
                    )
                )

            pred_depth_img_shape = load_depth_img(pred_depth_imgfiles[0]).shape
            pred_depth_imgs = load_depth_imgs(pred_depth_imgfiles, pred_depth_img_shape)

            K_pred_depth = torch.from_numpy(
                np.loadtxt(os.path.join(scan["pred_depth_dir"], "intrinsic_depth.txt"))[
                    :3, :3
                ]
            ).float()

            result["K_pred_depth"] = K_pred_depth
            result["pred_depth_imgs"] = pred_depth_imgs
        if self.load_normals:
            pred_normals_imgfiles = []
            for rgb_imgfile in rgb_imgfiles[view_inds]:
                pred_normals_imgfiles.append(
                    os.path.join(
                        scan["pred_normals_dir"],
                        "normal",
                        os.path.basename(rgb_imgfile)[:-4] + ".png",
                    )
                )

            pred_normal_img_shape = load_normal_img(pred_normals_imgfiles[0]).shape
            pred_normal_imgs = load_normal_imgs(pred_normals_imgfiles, pred_normal_img_shape)

            K_pred_normal = torch.from_numpy(
                np.loadtxt(os.path.join(scan["pred_normals_dir"], "intrinsic_depth.txt"))[
                    :3, :3
                ]
            ).float()

            result["K_pred_normal"] = K_pred_normal
            result["pred_normal_imgs"] = pred_normal_imgs
        #view_inds = view_inds.long()
        result["poses"] = poses[view_inds]
        result["gt_tsdf_npzfile"] = tsdf_npzfile

        input_coords = crop_bounds.grid_coords(grid_res=self.crop_voxel_size)

        gt_occ = (gt_tsdf.abs() < 0.999).float()
        gt_occ[gt_tsdf.isnan()] = torch.nan
        gt_occ = dilate_gt_occ(gt_occ)

        # start with the coords of the bounding box of the (maybe rotated) crop
        minbound, maxbound = crop_bounds.bounding_box()
        min_idx = ((minbound - gt_origin) / gt_voxel_size).floor()
        max_idx = ((maxbound - gt_origin) / gt_voxel_size).ceil()
        min_idx = torch.clamp(min_idx, torch.zeros(3), torch.tensor(gt_tsdf.shape))
        max_idx = torch.clamp(max_idx, torch.zeros(3), torch.tensor(gt_tsdf.shape))

        x_idx = torch.arange(min_idx[0], max_idx[0], dtype=torch.long)
        y_idx = torch.arange(min_idx[1], max_idx[1], dtype=torch.long)
        z_idx = torch.arange(min_idx[2], max_idx[2], dtype=torch.long)
        xx, yy, zz = torch.meshgrid(x_idx, y_idx, z_idx, indexing="ij")
        idxs = torch.stack((xx, yy, zz), dim=-1).view(-1, 3)
        original_shape = torch.stack((xx, yy, zz), dim=-1).shape

        bad_idxs = torch.any(
            (idxs < 0) | (idxs >= torch.tensor(gt_tsdf.shape)), dim=-1
        )
        idxs = idxs[~bad_idxs]
        gt_tsdf = gt_tsdf[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        gt_occ = gt_occ[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        output_coords = idxs * gt_voxel_size + gt_origin

        # narrow down to just the coords inside the crop
        idx = crop_bounds.contains(output_coords)
        output_coords = output_coords[idx]
        gt_tsdf = gt_tsdf[idx]
        gt_occ = gt_occ[idx]

        nsample = len(gt_tsdf) #torch.div(torch.prod(self.crop_size_nvox), 4,rounding_mode="trunc")
        if len(gt_tsdf) < nsample:
            print(
                f"resampling ({len(gt_tsdf)} / {nsample})",
                scan["scan_name"],
            )
            return self[np.random.choice(len(self))]

        result["oc_all"] = output_coords
        result["oc_all_idx"] = idx
        result["gt_tsdf_all"] = gt_tsdf
        result["gt_occ_all"] = gt_occ

        idx = np.random.choice(len(gt_tsdf), size=int(nsample), replace=False)
        output_coords = output_coords[idx]
        gt_tsdf = gt_tsdf[idx]
        gt_occ = gt_occ[idx]
        

        result["output_coords"] = output_coords
        result["gt_tsdf"] = gt_tsdf
        result["gt_occ"] = gt_occ
        result["input_coords"] = input_coords
        result["oc_idx"] = idx
        result["oc_shape"] = original_shape
        
        return result
