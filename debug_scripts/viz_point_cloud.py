import numpy as np
import torch
import skimage
import trimesh
import open3d as o3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import pandas as pd

def points_to_pc(points, color=None):
    """
    Convert an array of points to an Open3D PointCloud object.
    Args:
    - points: A numpy array of point coordinates.
    - color: Optional color for the points.

    Returns: An Open3D PointCloud object.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    if color is not None:
        if color.shape[0] == 3:
            color = np.tile(color, (points.shape[0], 1))
        pc.colors = o3d.utility.Vector3dVector(color)

    return pc

def data_to_points(data):
    """
    Convert a DataFrame with columns 'x', 'y', 'z' to a numpy array of points.
    Args:
    - data: A pandas DataFrame containing 'x', 'y', 'z' columns.

    Returns: A numpy array of shape (n_points, 3).
    """
    if "valid" in data.columns:
        data = data[data["valid"] == 1]
    return np.vstack((data["x"], data["y"], data["z"])).T

def load_and_sample_csv_data(filenames, sample_frac=0.1):
    """
    Load and sample data from CSV files.
    Args:
    - filenames: A list of filenames to load data from.
    - sample_frac: Fraction of data to sample.

    Returns: A list of sampled data DataFrames.
    """
    return [pd.read_csv(filename).sample(frac=sample_frac).reset_index(drop=True) for filename in filenames]

def load_npz_data(filename):
    """
    Load data from a NPZ file.
    Args:
    - filename: The filename of the NPZ file to load data from.

    Returns: The loaded data.
    """
    return np.load(filename)

def sample_mesh_from_tsdf(tsdf, origin, voxel_size=0.02, n_samples=100000):
    """
    Create a mesh from a TSDF volume and sample points from it.
    Args:
    - tsdf: TSDF volume.
    - origin: Origin of the TSDF volume.
    - voxel_size: The size of each voxel.
    - n_samples: Number of points to sample from the mesh.

    Returns: Sampled points from the created mesh.
    """
    verts, faces, _, _ = skimage.measure.marching_cubes(tsdf, level=0.5)
    faces = faces[~np.any(np.isnan(verts[faces]), axis=(1, 2))]
    verts = (verts * voxel_size) + origin
    verts = torch.tensor(verts, dtype=torch.float)
    faces = torch.tensor(faces, dtype=torch.long)
    mesh = Meshes([verts], [faces])
    return sample_points_from_meshes(mesh, num_samples=n_samples).detach().cpu().numpy()[0]

# Main script
csv_filenames = ['data/xyz_norm_scene0707_00_build_mesh_sampling.csv', 
                 'data/xyz_norm_scene0707_00_build_projection.csv',
                 'data/xyz_norm_scene0707_00_gt_depth.csv']

# Load and sample data from CSV files
csv_datas = load_and_sample_csv_data(csv_filenames)

# Define colors for different point clouds
colors = [np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 0, 0])]

# Convert data to point clouds
points = [data_to_points(data) for data in csv_datas]
pcs = [points_to_pc(p, color=c) for p, c in zip(points, colors)]

# Load TSDF data and create mesh
npz_data = load_npz_data('data/scene707_tsdf.npz')
tsdf_params = (npz_data["tsdf"][0], npz_data["crop_center"] - npz_data["crop_size_m"] / 2)
mesh_samples = sample_mesh_from_tsdf(*tsdf_params)

# Create point cloud from mesh samples
mesh_pc = points_to_pc(mesh_samples, color=np.array([1, 0, 0]))

# Visualize the point clouds
geometries = pcs + [mesh_pc, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])]
o3d.visualization.draw_geometries(geometries)
