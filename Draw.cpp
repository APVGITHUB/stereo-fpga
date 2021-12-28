#include <string>
#include <stdlib.h>
#include <iostream>
#include "open3d/Open3D.h"


typedef open3d::geometry::Image o3Image;
typedef std::shared_ptr<o3Image> o3ImagePtr;
typedef open3d::geometry::PointCloud o3PointCloud;
typedef std::shared_ptr<o3PointCloud> o3PointCloudPtr;

int main(int argc, char *argv[]) {

    auto image_ptr = std::make_shared<o3Image>();
    open3d::io::ReadImageFromJPG("/home/alex/open3d-cmake-find-package/image.jpg", *image_ptr);
    auto depth_ptr = std::make_shared<o3Image>();
    open3d::io::ReadImageFromJPG("/home/alex/open3d-cmake-find-package/result_fadnet.jpg", *depth_ptr);

    std::shared_ptr<open3d::geometry::RGBDImage> rgbd_ptr =
            open3d::geometry::RGBDImage::CreateFromColorAndDepth(
                *image_ptr, *depth_ptr); 

    open3d::camera::PinholeCameraIntrinsic intrinsic(
                open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    o3PointCloudPtr ptcd_ptr = o3PointCloud::CreateFromRGBDImage(*rgbd_ptr, intrinsic);  

    Eigen::Matrix<double, 4, 4> transform {
        {1,0,0,0},
        {0,1,0,0},
        {0,0,-1,0},
        {0,0,0,1},
    };
    // [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    ptcd_ptr->Transform(transform);
    open3d::io::WritePointCloud("./pointcloud.pcd",*ptcd_ptr);
    open3d::visualization::DrawGeometries({ptcd_ptr}, "point cloud from rgbd");
//
    return 0;
}
