import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import wget

print(o3d.__version__)
pipe = rs.pipeline()

# LIST OF OPERATIONS

# io.read_point_cloud
# io.write_point_cloud
# visualization.draw_geometries
# voxel_down_sample
# crop_point_cloud
# paint_uniform_color
# get_axis_aligned_bounding_box
# get_oriented_bounding_box

## TUTORIAL ##########################################################################################################################

url = "https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/test/bunny.pcd"
# filename = wget.download(url)



# Import pointcloud
pcd = o3d.io.read_point_cloud("bunny.pcd", format='pcd')
print(pcd)

# Write/save pointcloud
o3d.io.write_point_cloud("copy_of_bunny.pcd", pcd)

# Import mesh
# mesh = o3d.io.read_triangle_mesh("knot.ply")



#Download the point cloud using below command
import wget
url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
# filename = wget.download(url)

# Import ply
bunny_path = "./bunny/reconstruction/bun_zipper.ply"
mesh = o3d.io.read_triangle_mesh(bunny_path)

# Compute normals for shading
mesh.compute_vertex_normals()

# Show mesh
o3d.visualization.draw_geometries([mesh])

# Sample mesh to get ptcloud
pcd = mesh.sample_points_uniformly(number_of_points=500)
# Show ptcloud
o3d.visualization.draw_geometries([pcd])





# Voxel downsampling
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],width=1920, height=1080, left=50, top=50)





# Another downsampling example
#read_point_cloud reads a point cloud from a file. It tries to decode the file based on the extension name.
pcd = o3d.io.read_point_cloud("./bunny/reconstruction/bun_zipper.ply")
print(pcd)
print(np.asarray(pcd.points))

#draw_geometries visualizes the point cloud. 
o3d.visualization.draw_geometries([pcd],width=1920, height=1080, left=50, top=50)

#downsample
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],width=1920, height=1080, left=50, top=50)






