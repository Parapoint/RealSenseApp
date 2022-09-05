import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# LIST OF open3d OPERATIONS

# io.read_point_cloud
# io.write_point_cloud
# visualization.draw_geometries
# voxel_down_sample
# crop_point_cloud
# paint_uniform_color
# get_axis_aligned_bounding_box
# get_oriented_bounding_box


# Import pointcloud
# pcd = o3d.io.read_point_cloud("bunny.pcd", format='pcd')
# pcd = o3d.io.read_point_cloud("./bunny/reconstruction/bun_zipper.ply")

# Show ptCloud
# o3d.visualization.draw_geometries([pcd])


def point_cloud_bounding_box():
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box.color = (1, 0, 0)
    oriented_bounding_box = pcd.get_oriented_bounding_box()
    oriented_bounding_box.color = (0, 1, 0)
    print(
        "Displaying axis_aligned_bounding_box in red and oriented bounding box in green ..."
    )
    o3d.visualization.draw(
        [pcd, axis_aligned_bounding_box, oriented_bounding_box])

def point_cloud_convex_hull():
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    pcl = mesh.sample_points_poisson_disk(number_of_points=10000)
    hull, _ = pcl.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    o3d.visualization.draw([pcl, hull_ls])

def point_cloud_crop():
    print("Load a ply point cloud, crop it, and render it")
    sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.point_cloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(
        sample_ply_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)
    # Flip the pointclouds, otherwise they will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    chair.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    print("Displaying original pointcloud ...")
    o3d.visualization.draw([pcd])
    print("Diplaying cropped pointcloud")
    o3d.visualization.draw([chair])

def point_cloud_dbscan_clustering():
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw([pcd])

def point_cloud_distance():
    sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.point_cloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(
        sample_ply_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)

    chair.paint_uniform_color([0, 0, 1])
    pcd.paint_uniform_color([1, 0, 0])
    print("Displaying the two point clouds used for calculating distance ...")
    o3d.visualization.draw([pcd, chair])

    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    print("Printing average distance between the two point clouds ...")
    print(dists)


def point_cloud_hidden_point_removal():

    # Convert mesh to a point cloud and estimate dimensions.
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print("Displaying input point cloud ...")
    o3d.visualization.draw([pcd], point_size=5)

    # Define parameters used for hidden_point_removal.
    camera = [0, 0, diameter]
    radius = diameter * 100

    # Get all points that are visible from given view point.
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Displaying point cloud after hidden point removal ...")
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw([pcd], point_size=5)

def point_cloud_iss_keypoint_detector():
    # Compute ISS Keypoints on armadillo pointcloud.
    armadillo_data = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    tic = time.time()
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.visualization.draw([keypoints, mesh], point_size=5)


def point_cloud_normal_estimation():
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(5000)
    # Invalidate existing normals.
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    print("Displaying input pointcloud ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    pcd.estimate_normals()
    print("Displaying pointcloud with normals ...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    print("Printing the normal vectors ...")
    print(np.asarray(pcd.normals))


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw([inlier_cloud, outlier_cloud])


def point_cloud_outlier_removal_radius():
    ptcloud_data = o3d.data.PLYPointCloud()

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(ptcloud_data.path)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    o3d.visualization.draw([pcd])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw([voxel_down_pcd])

    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)

def point_cloud_outlier_removal_statistical():
    ptcloud_data = o3d.data.PLYPointCloud()

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(ptcloud_data.path)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))
    o3d.visualization.draw([pcd])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw([voxel_down_pcd])

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)


def point_cloud_paint():
    print("Load a ply point cloud, print it, and render it")
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    o3d.visualization.draw([pcd])
    print("Paint pointcloud")
    pcd.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw([pcd])

def point_cloud_plane_segmentation():

    sample_pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print("Displaying pointcloud with planar points in red ...")
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw([inlier_cloud, outlier_cloud])

def point_cloud_to_depth():
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(tum_data.depth_path)
    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                 [0, 0, 1]])

    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth,
                                                            intrinsic,
                                                            depth_scale=5000.0,
                                                            depth_max=10.0)
    o3d.visualization.draw([pcd])
    depth_reproj = pcd.project_to_depth_image(640,
                                              480,
                                              intrinsic,
                                              depth_scale=5000.0,
                                              depth_max=10.0)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(depth.to_legacy()))
    axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
    plt.show()

def point_cloud_to_rgbd():
    device = o3d.core.Device('CPU:0')
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(tum_data.depth_path).to(device)
    color = o3d.t.io.read_image(tum_data.color_path).to(device)

    intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                 [0, 0, 1]])
    rgbd = o3d.t.geometry.RGBDImage(color, depth)

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                                           intrinsic,
                                                           depth_scale=5000.0,
                                                           depth_max=10.0)
    o3d.visualization.draw([pcd])
    rgbd_reproj = pcd.project_to_rgbd_image(640,
                                            480,
                                            intrinsic,
                                            depth_scale=5000.0,
                                            depth_max=10.0)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(rgbd_reproj.color.to_legacy()))
    axs[1].imshow(np.asarray(rgbd_reproj.depth.to_legacy()))
    plt.show()

def translate():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_tx = copy.deepcopy(pcd).translate((150, 0, 0))
    pcd_ty = copy.deepcopy(pcd).translate((0, 200, 0))
    print('Displaying original and translated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Translated (in X) Geometry",
        "geometry": pcd_tx
    }, {
        "name": "Translated (in Y) Geometry",
        "geometry": pcd_ty
    }],
                        show_ui=True)

def rotate():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_r = copy.deepcopy(pcd).translate((200, 0, 0))
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    pcd_r.rotate(R)
    print('Displaying original and rotated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Rotated Geometry",
        "geometry": pcd_r
    }],
                           show_ui=True)

def scale():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_s = copy.deepcopy(pcd).translate((200, 0, 0))
    pcd_s.scale(0.5, center=pcd_s.get_center())
    print('Displaying original and scaled geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Scaled Geometry",
        "geometry": pcd_s
    }],
                           show_ui=True)

def transform():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[0, 3] = 150
    T[1, 3] = 200
    print(T)
    pcd_t = copy.deepcopy(pcd).transform(T)
    print('Displaying original and transformed geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Transformed Geometry",
        "geometry": pcd_t
    }],
                           show_ui=True)
                           
def point_cloud_transformation():

    translate()
    rotate()
    scale()
    transform()

def point_cloud_voxel_downsampling():
    print("Load a ply point cloud, print it, and render it")
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    o3d.visualization.draw([pcd])
    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw([downpcd])


def point_cloud_with_numpy():
    # Generate some n x 3 matrix using a variant of sync function.
    x = np.linspace(-3, 3, 201)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print("Printing numpy array used to make Open3D pointcloud ...")
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying Open3D pointcloud made using numpy array ...")
    o3d.visualization.draw([pcd])

    # Convert Open3D.o3d.geometry.PointCloud to numpy array.
    xyz_converted = np.asarray(pcd.points)
    print("Printing numpy array made using Open3D pointcloud ...")
    print(xyz_converted)

def camera_trajectory():

    print("Testing camera in open3d ...")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print(intrinsic.intrinsic_matrix)
    print(o3d.camera.PinholeCameraIntrinsic())
    ## HERE ##########################################################
    x = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    ##################################################################
    print(x)
    print(x.intrinsic_matrix)
    o3d.io.write_pinhole_camera_intrinsic("test.json", x)
    y = o3d.io.read_pinhole_camera_intrinsic("test.json")
    print(y)
    print(np.asarray(y.intrinsic_matrix))

    print("Read a trajectory and combine all the RGB-D images.")
    pcds = []
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        redwood_rgbd.trajectory_log_path)
    o3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.parameters[0].extrinsic)
    print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(5):
        im1 = o3d.io.read_image(redwood_rgbd.depth_paths[i])
        im2 = o3d.io.read_image(redwood_rgbd.color_paths[i])
        im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

def main():
    # point_cloud_bounding_box()
    # point_cloud_convex_hull()
    # point_cloud_crop()
    # point_cloud_dbscan_clustering()
    # point_cloud_distance()
    # point_cloud_hidden_point_removal()
    # point_cloud_iss_keypoint_detector()
    # point_cloud_normal_estimation()
    # point_cloud_outlier_removal_radius()
    # point_cloud_outlier_removal_statistical()
    # point_cloud_paint()
    # point_cloud_plane_segmentation()
    # point_cloud_to_depth()
    # point_cloud_to_rgbd()
    # point_cloud_transformation()
    # point_cloud_voxel_downsampling()
    # point_cloud_with_numpy()
    print('End of program')

# Entry point
if __name__ == "__main__":
    print("\nStarting Realsense.py...")
    main()

else:
    print("Warning: Executing as imported file")
    pass