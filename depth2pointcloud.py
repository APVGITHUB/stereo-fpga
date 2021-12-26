# import open3d as o3d
# from matplotlib import pyplot as plt

# color_raw = o3d.io.read_image("../build/000002_10.png")
# depth_raw = o3d.io.read_image("../build/result_fadnet.jpg")
# rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
#     color_raw, depth_raw)
# # print(rgbd_image)


# # plt.subplot(1, 2, 1)
# # plt.title('grayscale image')
# # plt.imshow(rgbd_image.color)
# # plt.subplot(1, 2, 2)
# # plt.title('depth image')
# # plt.imshow(rgbd_image.depth)
# # plt.show()
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#     rgbd_image,
#     o3d.camera.PinholeCameraIntrinsic(
#         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# # Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pcd])

# import numpy as np

# def convert_from_uvd(self, u, v, d):
#     d *= self.pxToMetre
#     x_over_z = (self.cx - u) / self.focalx
#     y_over_z = (self.cy - v) / self.focaly
#     z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
#     x = x_over_z * z
#     y = y_over_z * z
#     return x, y, z

