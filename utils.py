import scipy.spatial
import skimage.measure
import trimesh
import torch
import numpy as np
import einops
import pandas as pd

from torch_scatter import scatter_mean
from torch_scatter.composite import scatter_std


def log_transform(tsdf):
    tsdf = tsdf + 1e-6 
    result = torch.log(tsdf.abs() + 1)
    result *= torch.sign(tsdf)
    return result


def points_to_csv(xyz,filename,valid=None):
    """
    xyz: B x N x 3
    filename: str, path to save the points in csv format
    valid: B x N, whether the point is valid
    """
    B,N,_ = xyz.shape
    xyz = xyz.to("cpu").numpy()
    x = xyz[:,:,0].reshape(B,-1)
    y = xyz[:,:,1].reshape(B,-1)
    z = xyz[:,:,2].reshape(B,-1)
    if valid is not None:
        valid = valid.to("cpu").numpy()
        valid = valid.reshape(B,-1)
    
    df = pd.DataFrame({"x":x.flatten(),"y":y.flatten(),"z":z.flatten()})
    df["idx"] = np.repeat(np.arange(B),N)
    if valid is not None:
        df["valid"] = valid.flatten()
    df.to_csv(filename,index=False)


def tsdf2mesh(tsdf, voxel_size, origin, level=0):
    verts, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=level)
    faces = faces[~np.any(np.isnan(verts[faces]), axis=(1, 2))]
    verts = verts * voxel_size + origin
    return trimesh.Trimesh(verts, faces)


def project(xyz, poses, K, imsize):
    """
    xyz: b x (*spatial_dims) x 3
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    imsize: (imheight, imwidth)
    """

    device = xyz.device
    batch_size = xyz.shape[0]
    spatial_dims = xyz.shape[1:-1]
    n_views = poses.shape[1]

    xyz = xyz.view(batch_size, 1, -1, 3).transpose(3, 2)
    xyz = torch.cat((xyz, torch.ones_like(xyz[:, :, :1])), dim=2)

    with torch.autocast(enabled=False, device_type=device.type):
        xyz_cam = (torch.inverse(poses).float() @ xyz.float())[:, :, :3]
        uv = K @ xyz_cam

    z = uv[:, :, 2]
    uv = uv[:, :, :2] / uv[:, :, 2:]
    imheight, imwidth = imsize
    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel
    then we allow values between (-.5, w-.5) because they are inside the border pixel
    """
    valid = (
        (uv[:, :, 0] >= -0.5)
        & (uv[:, :, 1] >= -0.5)
        & (uv[:, :, 0] <= imwidth - 0.5)
        & (uv[:, :, 1] <= imheight - 0.5)
        & (z > 0)
    )
    uv = uv.transpose(2, 3)

    uv = uv.view(batch_size, n_views, *spatial_dims, 2)
    z = z.view(batch_size, n_views, *spatial_dims)
    valid = valid.view(batch_size, n_views, *spatial_dims)
    return uv, z, valid


def sample_posed_images(
    imgs, poses, K, xyz, mode="bilinear", padding_mode="zeros", return_z=False
):
    """
    imgs: b x nviews x C x H x W
    poses: b x nviews x 4 x 4
    K: (b x nviews x 3 x 3)
    xyz: b x (*spatial_dims) x 3
    """

    device = imgs.device
    batch_size, n_views, _, imheight, imwidth = imgs.shape
    spatial_dims = xyz.shape[1:-1]

    """
    assuming that these uv coordinates have
        (0, 0) = center of top left pixel
        (w - 1, h - 1) = center of bottom right pixel

    adjust because grid_sample(align_corners=False) assumes
        (0, 0) = top left corner of top left pixel
        (w, h) = bottom right corner of bottom right pixel
    """
    uv, z, valid = project(xyz, poses, K, (imheight, imwidth))
    imsize = torch.tensor([imwidth, imheight], device=device)
    # grid = (uv + 0.5) / imsize * 2 - 1
    grid = uv / (0.5 * imsize) + (1 / imsize - 1)
    grid = grid.float()
    imgs = imgs.float()
    vals = torch.nn.functional.grid_sample(
        imgs.view(batch_size * n_views, *imgs.shape[2:]),
        grid.view(batch_size * n_views, 1, -1, 2),
        align_corners=False,
        mode=mode,
        padding_mode=padding_mode,
    )
    vals = vals.view(batch_size, n_views, -1, *spatial_dims)
    if return_z:
        return vals, valid, z
    else:
        return vals, valid


def sample_voxel_feats(img_feats, poses, K, xyz, imsize, invalid_fill_value=0):
    base_imheight, base_imwidth = imsize
    featheight = img_feats.shape[3]
    featwidth = img_feats.shape[4]
    _K = K.clone()
    _K[:, :, 0] *= featwidth / base_imwidth
    _K[:, :, 1] *= featheight / base_imheight

    voxel_feats, valid = sample_posed_images(
        img_feats,
        poses,
        _K,
        xyz,
        mode="bilinear",
        padding_mode="border",
    )
    voxel_feats.masked_fill_(~valid[:, :, None], invalid_fill_value)

    return voxel_feats, valid

def str_unfold_dim(original_pattern,pattern_to_unfold,dims_to_unfold):
    stringified_dim = " ".join([f"dim{i}" for i in len(dims_to_unfold)])
    final_pattern = original_pattern.replace(pattern_to_unfold,stringified_dim)
    return final_pattern


def sample_voxel_feats_(
    poses,
    xyz,
    rgb_imsize,
    img_feats,
    K_rgb,
    depth_imgs,
    K_depth,
    normal_imgs,
    K_normal,
    invalid_fill_value=0,
):
    #### Image features ######
    rgb_imheight, rgb_imwidth = rgb_imsize
    featheight = img_feats.shape[3]
    featwidth = img_feats.shape[4]
    K_img_feats = K_rgb.clone()
    K_img_feats[:, :, 0] *= featwidth / rgb_imwidth
    K_img_feats[:, :, 1] *= featheight / rgb_imheight
    img_feats, img_feats_valid = sample_posed_images(
        img_feats,
        poses,
        K_img_feats,
        xyz,
        mode="bilinear",
        padding_mode="border",
    )
    img_feats.masked_fill_(~img_feats_valid[:, :, None], invalid_fill_value)
    ##########

    #### Normal features ####
    normal_imgs = einops.rearrange(normal_imgs, "B N H W C -> B N C H W")
    normals, normals_valid, z_normal = sample_posed_images(
        normal_imgs, poses, K_normal, xyz, mode="nearest", return_z=True
    )
    normals.masked_fill_(~normals_valid[:, :, None], invalid_fill_value)

    #########

    #### Camera Angle ####
    spatial_dims = xyz.shape[1:-1]
    cam_world_coords = poses[
        :, :, :3, 3
    ]  # Coordinates of the camera, expressed in world coordinates
    
    if len(spatial_dims)==3:

        cam_world_coords = einops.repeat(
            cam_world_coords,
            "B N XYZ -> B N dim1 dim2 dim3 XYZ",
            dim1=spatial_dims[0],
            dim2=spatial_dims[1],
            dim3=spatial_dims[2],
        )
    else:
        cam_world_coords = einops.repeat(
        cam_world_coords,
        "B N XYZ -> B N dim XYZ",dim = spatial_dims[0]
    )
    _, n_imgs, _, _, _ = normal_imgs.shape
    # Direction between voxel and cam coords
    if len(spatial_dims) == 3:
        vect1 = cam_world_coords - einops.repeat(
            xyz, "B dim1 dim2 dim3 XYZ -> B N dim1 dim2 dim3 XYZ", N=n_imgs
        )
        vect1 = torch.nn.functional.normalize(vect1, dim=-1)
        vect2 = einops.repeat(
            poses[:, :, :3, 2],
            "B N XYZ -> B N dim1 dim2 dim3 XYZ",
            dim1=spatial_dims[0],
            dim2=spatial_dims[1],
            dim3=spatial_dims[2],
        )  # cam direction, already unit norm
    else :
        vect1 = cam_world_coords - einops.repeat(
            xyz, "B dim XYZ -> B N dim XYZ", N=n_imgs
        )
        vect1 = torch.nn.functional.normalize(vect1, dim=-1)
        vect2 = einops.repeat(
            poses[:, :, :3, 2],
            "B N XYZ -> B N dim XYZ",
            dim=spatial_dims[0],
        )

        
    relative_dir = vect1 - vect2
    if len(spatial_dims) == 3:
        relative_dir = einops.rearrange(
            relative_dir, "B N dim1 dim2 dim3 XYZ -> B N XYZ dim1 dim2 dim3"
        )
    else:
        relative_dir = einops.rearrange(
            relative_dir, "B N dim XYZ -> B N XYZ dim"
        )

    ########

    #### Signed Distance ####
    depth_imgs = depth_imgs[:, :, None]
    depths, depths_valid, z_depth = sample_posed_images(
        depth_imgs, poses, K_depth, xyz, mode="nearest", return_z=True
    )
    sdf = depths - z_depth[:, :, None]
    sdf.masked_fill_(
        ~depths_valid[:, :, None], invalid_fill_value
    )  # set invalid sdf to 0

    ########

    voxel_feats = torch.cat(
        [img_feats.float(), normals.float(), relative_dir.float(), sdf.float()],
        dim=2,
    ) # B N C H W
    voxel_valid = torch.max(
        depths_valid, normals_valid
    )  # equivalent to logical or, while keeping differentiability
    voxel_valid = torch.max(voxel_valid, img_feats_valid)

    return voxel_feats, voxel_valid


def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model
    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index

def normalize_coordinate(p, plane='xz',range="-1:1"):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]
    if range == "-1:1":
        xy_new = (xy / 2) + 0.5 # -1 -> 1 translated to 0 ->1
    else:
        xy_new = xy
    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new

def sample_image_feats(uv,imgs):
    # batched image sampling, given the uv coordinates
    # return shape is B Nv C Hin Win

    #uv represents the coordinates between -1 and 1 to sample the image features
    # img is the list of image to take from to cat, expected shape is B Nv C Hin Win 
    B, Nv, H, W , _ = uv.shape
    point_feats = []
    for b in range(B): #We need to use 4-D input for the grid_sample to eprform bilinear interpolation, otherewise it will be trilinear (i.e. take from other images)
        #Grid sample expects N C Hin Win for input and N Hout Wout 2 for grid
        grid = uv[b,...]
        batch_point_feats = []
        for img in imgs: ## Cannot use stack because the imgs might not have the same shape, need to loop
            input = img[b,...]
            img_point_feats = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            batch_point_feats.append(img_point_feats)
        batch_point_feats = torch.cat(batch_point_feats,dim = 1) # append on channels dim
        point_feats.append(batch_point_feats)
    point_feats = torch.stack(point_feats,dim=0) # concat on batch dimensiom
    return point_feats
    



def build_triplane_feature_encoder(params,cnns,device,append_var=False,plane_resolution=512,filename=None):
    """
    this function takes the images (depth, rgb, normal) and the poses and returns the features for each plane (xy, xz, yz)
    This is done by projecting the point cloud (each pixel in depth image, along with their corresponding image features) to the 3 planes and passing them through the CNNs

    params: dict with the following keys
    cnns: list of 3 CNNs, one for each plane
    device: torch.device
    append_var: bool, whether to append the variance to the mean
    plane_resolution: int, resolution of the plane
    filename: str, path to save the points in csv format

    """
    save_points = filename is not None
    poses = params["poses"]
    depth_imgs = params["depth_imgs"]
    img_feats = params["img_feats"]
    K_depth = params["K_depth"]
    normal_imgs = params["normal_imgs"]
    crop_center= params["crop_center"]
    crop_rotation= params["crop_rotation"]
    crop_size_m= params["crop_size_m"]



    B,Nv,h_depth_imgs,w_depth_imgs = depth_imgs.shape
    
    # create grids of the 20 depth map image
    # each grid should be (0, 0) to (w - 1, h - 1)
    xs = torch.linspace(0, w_depth_imgs-1, steps=w_depth_imgs)
    ys = torch.linspace(0, h_depth_imgs-1, steps=h_depth_imgs)

    # NOTE: be careful about the indexing
    x,y = torch.meshgrid(xs, ys, indexing='xy')

    x = x[None,None].repeat(B,Nv,1,1).to(device) # create b dim and number-of-view dim
    y = y[None,None].repeat(B,Nv,1,1).to(device) # create b dim and number-of-view dim

    uv = torch.stack([x,y],dim=-1) 
    uv = uv / torch.tensor([w_depth_imgs-1,h_depth_imgs-1]).to(device) # make uv between 0 and 1
    uv = ((uv*2)-1) # make uv between -1,1, for grid_sample
    points_img_feats = [depth_imgs[:,:,None],normal_imgs.permute(0,1,4,2,3),img_feats]
    fine_sparse_point_feats = sample_image_feats(uv, points_img_feats)



    z = depth_imgs
    x = x * z
    y = y * z # then we get pixel coordinate 
    xyz_pixel = torch.stack([x,y,z], dim=-1) 
    # sample point features for these sparse points

    _,_,H,W,_ = xyz_pixel.shape
    
    xyz_cams = torch.inverse(K_depth) @ xyz_pixel.reshape(B,Nv,H*W,-1).permute(0,1,3,2)
    xyz_cams = torch.cat([xyz_cams, torch.ones(B,Nv,1,H*W).to(device)], dim=2)
    xyz_world = poses.float() @ xyz_cams


    sparse_point_coords = xyz_world[:,:,:3,:]
    sparse_point_coords = sparse_point_coords.permute(0,1,3,2)
    sparse_point_coords = sparse_point_coords.reshape(B,-1,3)
    c_dim = fine_sparse_point_feats.shape[2]

    fine_sparse_point_feats = fine_sparse_point_feats.permute(0,1,3,4,2).reshape(B,-1,c_dim)
    depth_valid = z>0
    depth_valid = depth_valid.unsqueeze(-1).reshape(B,-1,1)
    sparse_point_coords_norm = sparse_point_coords - crop_center[:,None]
    sparse_point_coords_norm = torch.bmm(sparse_point_coords_norm, crop_rotation.transpose(1,2))
    sparse_point_coords_norm = sparse_point_coords_norm / crop_size_m[:,None]

    valid_in_bound = (sparse_point_coords_norm<=1) & (sparse_point_coords_norm>=-1)
    valid_in_bound = valid_in_bound.all(dim=-1,keepdim=True)

    valid = valid_in_bound & depth_valid


    fine_sparse_point_feats = fine_sparse_point_feats * valid

    if save_points:
        xyz = sparse_point_coords_norm.clone()
        B,N,_ = xyz.shape
        xyz = xyz.to("cpu").numpy()
        x = xyz[:,:,0].reshape(B,-1)
        y = xyz[:,:,1].reshape(B,-1)
        z = xyz[:,:,2].reshape(B,-1)
        if valid is not None:
            p_valid = valid.clone().to("cpu").numpy()
            p_valid = p_valid.reshape(B,-1)
        
        df = pd.DataFrame({"x":x.flatten(),"y":y.flatten(),"z":z.flatten()})
        if valid is not None:
            df["valid"] = p_valid.flatten()
        
        xys = []
        indexes = []
    
    planes = ["xy","xz","yz"]
    feature_planes = []
    for i,plane in enumerate(planes):
        # convert the sparse points to a unit cube
        xy = normalize_coordinate(sparse_point_coords_norm.clone(), plane=plane) # normalize to the range of (0, 1)
        index = coordinate2index(xy, plane_resolution)
        if save_points:
            xys.append(xy)
            indexes.append(index)    
        # scatter plane features from points
        fea_plane = fine_sparse_point_feats.new_zeros(B, c_dim, plane_resolution**2)
        fea_plane = scatter_mean(fine_sparse_point_feats.permute(0,2,1), index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(B, c_dim, plane_resolution, plane_resolution) # sparce matrix (B x 512 x reso x reso)
        if append_var:
            var_fea_plane = fine_sparse_point_feats.new_zeros(B, c_dim, plane_resolution**2)
            var_fea_plane = scatter_std(fine_sparse_point_feats.permute(0,2,1), index, out=var_fea_plane)
            var_fea_plane = fea_plane.reshape(B, c_dim, plane_resolution, plane_resolution)
            fea_plane = torch.cat([fea_plane,var_fea_plane],dim=1)
        # pass through 2D CNN
        fea_plane = cnns[i](fea_plane)
        feature_planes.append(fea_plane)
    if save_points:
        for i,plane in enumerate(planes):
            df["index_"+plane] = indexes[i].clone().to("cpu").numpy().reshape(B,-1).flatten()
            df["u_"+plane] = xys[i][:,:,0].clone().to("cpu").numpy().reshape(B,-1).flatten()
            df["v_"+plane] = xys[i][:,:,1].clone().to("cpu").numpy().reshape(B,-1).flatten()
        df.to_csv(filename,index=True)

    return feature_planes

def build_planar_feature_encoder_from_pc(params,cnns,device,append_var=False,plane_resolution=512,range="-1:1",save_points=False,filename="points.csv"):
        
    img_feats = params["img_feats"]
    K_rgb = params["K_rgb"]
    crop_center= params["crop_center"]
    #crop_rotation = params["crop_rotation"]
    crop_size_m= params["crop_size_m"]
    sparse_point_coords = params["xyz"]
    poses = params["poses"]
    B = img_feats.shape[0]

    rgb_imheight, rgb_imwidth = params["rgb_imsize"]
    featheight = img_feats.shape[3]
    featwidth = img_feats.shape[4]
    K_img_feats = K_rgb.clone()
    K_img_feats[:, :, 0] *= featwidth / rgb_imwidth
    K_img_feats[:, :, 1] *= featheight / rgb_imheight

    img_feats, img_feats_valid = sample_posed_images(
        img_feats,
        poses,
        K_img_feats,
        sparse_point_coords,
        mode="bilinear",
        padding_mode="border",
    )
    img_feats.masked_fill_(~img_feats_valid[:, :, None], 0)
    valid_in_views = (img_feats_valid.sum(dim=1)>0)[:,:,None]
    fine_sparse_point_feats = [img_feats.mean(dim = 1)]
    if append_var:
        fine_sparse_point_feats.append(img_feats.var(dim = 1))
    fine_sparse_point_feats = torch.cat(fine_sparse_point_feats,dim=1)
    fine_sparse_point_feats = fine_sparse_point_feats.permute(0,2,1) # make fine_sparse_point_feats B N_points C
    c_dim = fine_sparse_point_feats.shape[2]

    sparse_point_coords_norm = sparse_point_coords - crop_center[:,None]
    #sparse_point_coords_norm = torch.bmm(sparse_point_coords_norm, crop_rotation.transpose(1,2))
    sparse_point_coords_norm = sparse_point_coords_norm / crop_size_m[:,None]

    valid_in_bound = (sparse_point_coords_norm<=1) & (sparse_point_coords_norm>=-1)
    valid_in_bound = valid_in_bound.all(dim=-1,keepdim=True)

    valid = valid_in_bound & valid_in_views


    fine_sparse_point_feats = fine_sparse_point_feats * valid

    if save_points:
        xyz = sparse_point_coords_norm.clone()
        B,N,_ = xyz.shape
        xyz = xyz.to("cpu").numpy()
        x = xyz[:,:,0].reshape(B,-1)
        y = xyz[:,:,1].reshape(B,-1)
        z = xyz[:,:,2].reshape(B,-1)
        if valid is not None:
            p_valid = valid.clone().to("cpu").numpy()
            p_valid = p_valid.reshape(B,-1)
        
        df = pd.DataFrame({"x":x.flatten(),"y":y.flatten(),"z":z.flatten()})
        if valid is not None:
            df["valid"] = p_valid.flatten()
        
        xys = []
        indexes = []
    
    planes = ["xy","xz","yz"]
    feature_planes = []
    for i,plane in enumerate(planes):
        # convert the sparse points to a unit cube
        xy = normalize_coordinate(sparse_point_coords_norm.clone(), plane=plane,range=range) # normalize to the range of (0, 1)
        index = coordinate2index(xy, plane_resolution)
        if save_points:
            xys.append(xy)
            indexes.append(index)    

        # FIXME: the center is not 0,0
        # scatter plane features from points
        fea_plane = fine_sparse_point_feats.new_zeros(B, c_dim, plane_resolution**2)
        #c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(fine_sparse_point_feats.permute(0,2,1), index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(B, c_dim, plane_resolution, plane_resolution) # sparce matrix (B x 512 x reso x reso)
        if append_var:
            var_fea_plane = fine_sparse_point_feats.new_zeros(B, c_dim, plane_resolution**2)
            var_fea_plane = scatter_std(fine_sparse_point_feats.permute(0,2,1), index, out=var_fea_plane)
            var_fea_plane = fea_plane.reshape(B, c_dim, plane_resolution, plane_resolution)
            fea_plane = torch.cat([fea_plane,var_fea_plane],dim=1)
        # pass through 2D CNN
        fea_plane = cnns[i](fea_plane)
        feature_planes.append(fea_plane)
    if save_points:
        for i,plane in enumerate(planes):
            df["index_"+plane] = indexes[i].clone().to("cpu").numpy().reshape(B,-1).flatten()
            df["u_"+plane] = xys[i][:,:,0].clone().to("cpu").numpy().reshape(B,-1).flatten()
            df["v_"+plane] = xys[i][:,:,1].clone().to("cpu").numpy().reshape(B,-1).flatten()
        df.to_csv(filename,index=True)
    return feature_planes

def sample_planar_features(planar_features, xyz,aggregation="concat",params=None,save_points=False,filename="points.csv"):
    if params: 
        # make between -1 and 1
        crop_center = params["crop_center"]
        #crop_rotation= params["crop_rotation"]
        crop_size_m = params["crop_size_m"]
        xyz_norm = xyz - crop_center[:,None]
        xyz_norm = xyz_norm / crop_size_m[:,None]
    else: 
        # assume xyz is between 0 and 1
        xyz_norm = ((xyz*2) - 1) # make uv between -1,1, for grid_sample
    if save_points:
        points_to_csv(xyz_norm,filename)
    planes = ["xy","xz","yz"]
    xyz_feats = []
    for plane,plane_features in zip(planes,planar_features):
        if plane == 'xz':
            xy = xyz_norm[:, :, [0, 2]]
        elif plane =='xy':
            xy = xyz_norm[:, :, [0, 1]]
        else:
            xy = xyz_norm[:, :, [1, 2]]
        xyz_feats_plane = torch.nn.functional.grid_sample(plane_features, xy[:,None], mode='bilinear', padding_mode='zeros', align_corners=False)
        xyz_feats_plane = xyz_feats_plane.squeeze(2)
        xyz_feats.append(xyz_feats_plane)
    if aggregation == "concat":
        xyz_feats = torch.cat(xyz_feats,dim=1)
    elif aggregation == "mean_var":
        mean_xyz_feats = torch.mean(torch.stack(xyz_feats,dim=1), dim=1)
        var_xyz_feats = torch.var(torch.stack(xyz_feats,dim=1),dim=1)
        xyz_feats = torch.cat([mean_xyz_feats,var_xyz_feats],dim=1)
    elif aggregation == "dot_prod":
        B,c,n = xyz_feats[0].shape
        dot_product = []
        partial_dot_product = torch.bmm(xyz_feats[0].view(B*n,1,c),xyz_feats[1].view(B*n,c,1))
        dot_product.append(partial_dot_product.view(B,1,-1))
        partial_dot_product = torch.bmm(xyz_feats[1].view(B*n,1,c),xyz_feats[2].view(B*n,c,1))
        dot_product.append(partial_dot_product.view(B,1,-1))
        partial_dot_product = torch.bmm(xyz_feats[2].view(B*n,1,c),xyz_feats[0].view(B*n,c,1))
        dot_product.append(partial_dot_product.view(B,1,-1))
        xyz_feats = torch.cat(dot_product,dim=1)
    elif aggregation == "hadamard_prod":
        partial_had_product = xyz_feats[0].mul(xyz_feats[1])
        xyz_feats = partial_had_product.mul(xyz_feats[2])
    else:
        xyz_feats = torch.cat(xyz_feats,dim=1) # concat as default
        
    return xyz_feats

