import os
import torch
import open3d as o3d
import numpy as np
from classification_ModelNet40.data import ModelNet40, UniformSampling, GradientSampling, SplitSampling, label_to_category_name
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transformations (if needed)
transform = transforms.Compose([
    # UniformSampling(num_points=1024, dropout_rate=1.0),
    # GradientSampling(num_points=1024),
    # SplitSampling(num_points=1024),
    # transforms.ToTensor(),
    # Add other transformations if required
])

# Load the ModelNet40 dataset
train_dataset = ModelNet40(partition='train', transform=transform)
test_dataset = ModelNet40(partition='test', transform=transform)

def visualize_point_cloud(points):
    # Convert tensor to numpy
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize
    o3d.visualization.draw_geometries([pcd])

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    mask = sqrdists <= radius ** 2
    inside_radius = mask.sum(dim=-1).float()
      
    
    max_inside_radius = inside_radius.max(dim=-1)[0]
    min_inside_radius = inside_radius.min(dim=-1)[0]
    mean_inside_radius = inside_radius.mean(dim=-1)
    std_inside_radius = inside_radius.std(dim=-1)
    print("max_inside_radius", max_inside_radius)
    print("min_inside_radius", min_inside_radius)
    print("mean_inside_radius", mean_inside_radius) 
    print("std_inside_radius", std_inside_radius)


if __name__ == "__main__":
    # Print dataset information
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # Draw a random index from the train dataset
    random_index = torch.randint(0, len(train_dataset), (1,)).item()
    data, label = train_dataset[random_index]
    radius = 0.15
    nsample = 32
    for i in range(30):
        print(f"Sample {i+1}:")
        xyz = train_dataset[random_index][0]
        # compute maximum dimention
        max_dim = np.max(xyz, axis=0) - np.min(xyz, axis=0)
        print(f"Max dimension: {max_dim}")
        xyz = torch.from_numpy(xyz).unsqueeze(0)
        query_ball_point(radius, nsample, xyz, xyz)
    
    # print(f"Data shape: {data.shape}, Label: {label_to_category_name(label[0])}")
    
    # # Visualize the sample
    # visualize_point_cloud(data)
