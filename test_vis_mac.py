import open3d as o3d
import numpy as np
import plotly.graph_objects as go

# Set visualization backend to Metal for Mac
o3d.visualization.webrtc_server.enable_webrtc()

kitti_bin_path = "/home/workspace/KITTI/dataset/sequences/00/velodyne/000000.bin"

bin_file = open(kitti_bin_path, "rb")

points = np.fromfile(bin_file, dtype=np.float32)

points = points.reshape(-1, 4)

# Create a regular CPU PointCloud
pcd = o3d.geometry.PointCloud()
# Convert points to CPU numpy array before creating Vector3dVector
pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))

# Convert to numpy array for plotly
points_np = np.asarray(pcd.points)

# Create 3D scatter plot using plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode="markers",
            marker=dict(size=1, color=points_np[:, 2], colorscale="Viridis", opacity=0.8),  # Color by z-value
        )
    ]
)

# Update layout
fig.update_layout(title="Point Cloud Visualization", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

# Show the plot
fig.show()

print(points.shape)
