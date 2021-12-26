#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <vitis/ai/proto/dpu_model_param.pb.h>
#include <vitis/ai/library/tensor.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <math.h>
#include <utility>

#include <vitis/ai/profiling.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>

#include "open3d/Open3D.h"

typedef open3d::geometry::Image o3Image;
typedef std::shared_ptr<o3Image> o3ImagePtr;
typedef open3d::geometry::PointCloud o3PointCloud;
typedef std::shared_ptr<o3PointCloud> o3PointCloudPtr;


using namespace std;
using namespace cv;

void cost_volume(vitis::ai::library::OutputTensor input_l, vitis::ai::library::OutputTensor input_r, 
                        vitis::ai::library::InputTensor output) {

  vector<float> temp(output.size);
      
  for (size_t a = 0; a < input_l.batch; ++a) {
    int8_t* left_input = (int8_t*)input_l.get_data(a);
    float left_scale = vitis::ai::library::tensor_scale(input_l);
    int8_t* right_input = (int8_t*)input_r.get_data(a);
    float right_scale = vitis::ai::library::tensor_scale(input_r);
  
    int8_t* volume = (int8_t*)output.get_data(a);
    float v_scale = vitis::ai::library::tensor_scale(output);
    size_t out_size = output.height * output.width * output.channel;
  
    // compute the first channel of the cost volume
    for (size_t b = 0; b < input_l.height; ++b)
      for (size_t c = 0; c < input_l.width; ++c) {
        float sum = 0.0;
        for (size_t d = 0; d < input_l.channel; ++d) {
          size_t pos = (b*input_l.width + c)*input_l.channel + d;
          // only write the first channel
          sum += (float)left_input[pos] * (float)right_input[pos];
        }
        size_t out_pos = (b*output.width + c)*output.channel;
        temp[out_pos] = sum;
      }
    
    // compute the rest channels of the cost volume
    for(size_t b = 0; b < input_l.height; ++b)
      for(size_t i = 1; i < output.channel; ++i)
        for(size_t c = i, e = 0; c < input_l.width; ++c, ++e) {
          float sum = 0.0;
          for(size_t d = 0; d < input_l.channel; ++d) {
            size_t pos = b*input_l.width*input_l.channel;
            sum += (float)left_input[pos + c*input_l.channel + d]
                   * (float)right_input[pos + e*input_l.channel + d];
          }
          size_t out_pos = a*out_size + (b*output.width + c)*output.channel + i;
  	      temp[out_pos] = sum;
      	}

    // compute the average figure of volume
    float real_scale = right_scale * left_scale * v_scale;
    float tmp = 0.0;
    for (size_t i = 0; i < out_size; ++i) {
      tmp = temp[i];
      if (tmp >= 0)
        volume[i] = (int8_t)round(tmp/(float)input_l.channel * real_scale);
      else {
        // leaky relu
        tmp = tmp/(float)input_l.channel * real_scale * 0.1015625;
        if (tmp - floor(tmp) == 0.5)
          volume[i] = (int8_t)ceil(tmp);
        else
          volume[i] = (int8_t)round(tmp);
      }
    }

  }
}

int8_t dpu_round(float num) {
  if(num - floor(num) == 0.5) return ceil(num);
  else return round(num);
}

 void copy_into_tensor (const vector<int8_t>& input,
	       	vitis::ai::library::InputTensor tensor, int o_fixpos) {

  int i_fixpos = tensor.fixpos;
  auto size = tensor.height * tensor.width * tensor.channel;

  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = (int8_t*)tensor.get_data(b);
    if(i_fixpos == o_fixpos) {
      memcpy(data, input.data() + b*size, size);
    } else {
  
      float o_scale = exp2f(-1.0 * o_fixpos);
      float i_scale = vitis::ai::library::tensor_scale(tensor);
      float scale = o_scale * i_scale;

    }
  }
}

vector<int8_t> copy_from_tensor (const vitis::ai::library::InputTensor tensor) {
  auto size = tensor.height * tensor.width * tensor.channel;
  vector<int8_t> output(tensor.size);
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = tensor.get_data(b);
    memcpy(output.data()+b*size, data, size);
  }

  return output;
}

vector<int8_t> copy_from_tensor (const vitis::ai::library::OutputTensor tensor) {
  auto size = tensor.height * tensor.width * tensor.channel;
  vector<int8_t> output(tensor.size);
  for (size_t b = 0; b < tensor.batch; ++b) {
    auto data = tensor.get_data(b);
    memcpy(output.data()+b*size, data, size);
  }

  return output;
}

// reorder the tensors with name
vector<vitis::ai::library::InputTensor> sort_tensors (
            const vector<vitis::ai::library::InputTensor>& tensors, vector<string>& layer_names) {
  vector<vitis::ai::library::InputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

vector<vitis::ai::library::OutputTensor> sort_tensors (
            const vector<vitis::ai::library::OutputTensor>& tensors, vector<string>& layer_names) {
  vector<vitis::ai::library::OutputTensor> ordered_tensors;
  for (auto i = 0u; i < layer_names.size(); ++i)
    for (auto j = 0u; j < tensors.size(); ++j)
      if (tensors[j].name.find(layer_names[i]) != std::string::npos) {
        ordered_tensors.push_back(tensors[j]);
        break;
      }
  return ordered_tensors;
}

// run the fadnet
vector<Mat> FADnet_run(vector<unique_ptr<vitis::ai::DpuTask>>& task,
                       const vector<pair<cv::Mat, cv::Mat>>& input_images) {
  vector<cv::Mat> left_mats;
  vector<cv::Mat> right_mats;
  auto input_tensor_left = task[0]->getInputTensor(0u)[0];
  auto sWidth = input_tensor_left.width;
  auto sHeight = input_tensor_left.height;

  for(size_t i = 0; i < input_tensor_left.batch; ++i) {
    cv::Mat left_mat;
    resize(input_images[i].first, left_mat, cv::Size(sWidth, sHeight));
    left_mats.push_back(left_mat);
    cv::Mat right_mat;
    resize(input_images[i].second, right_mat, cv::Size(sWidth, sHeight));
    right_mats.push_back(right_mat);
  }

  // ### kernel 0 part ###
  task[0]->setImageRGB(left_mats);
  // store the input 
  vector<int8_t> data_left = copy_from_tensor(input_tensor_left);

  task[0]->run(0u);

  // store the outputs of kernel_0
  auto outputs_l_unsort = task[0]->getOutputTensor(0u);
  vector<string> output_names_k0 = {"conv1", "conv2", "conv3"};
  auto outputs_l = sort_tensors(outputs_l_unsort, output_names_k0);

  vector<int8_t> data_conv1_l = copy_from_tensor(outputs_l[0]);
  vector<int8_t> data_conv2_l = copy_from_tensor(outputs_l[1]);
  vector<int8_t> data_conv3a_l = copy_from_tensor(outputs_l[2]);

  // ### kernel 1 part ###
  auto input_tensor_right = task[1]->getInputTensor(0u)[0];
  task[1]->setImageRGB(right_mats);
  vector<int8_t> data_right = copy_from_tensor(input_tensor_right);

  task[1]->run(0u);

  //  cost volume
  auto output_tensor_l = outputs_l[2];
  auto output_tensor_r = task[1]->getOutputTensor(0u)[0];

  auto input_kernel_2_unsort = task[2]->getInputTensor(0u);
  vector<string> input_names_k2 = {"3585", "input_34", "3581",
                                   "3582", "3583", "4236_inserted_fix_30",
                                   "4236_inserted_fix_16", "4237"};
  auto input_kernel_2 = sort_tensors(input_kernel_2_unsort, input_names_k2); 

  cost_volume(output_tensor_l, output_tensor_r, input_kernel_2[0]);

  // run the rest kernel
  copy_into_tensor(data_conv3a_l, input_kernel_2[1], outputs_l[2].fixpos);
  copy_into_tensor(data_conv1_l,  input_kernel_2[2], outputs_l[0].fixpos);
  copy_into_tensor(data_conv2_l,  input_kernel_2[3], outputs_l[1].fixpos);
  copy_into_tensor(data_left,     input_kernel_2[4], input_tensor_left.fixpos);
  copy_into_tensor(data_left,     input_kernel_2[5], input_tensor_left.fixpos);
  copy_into_tensor(data_left,     input_kernel_2[6], input_tensor_left.fixpos);
  copy_into_tensor(data_right,    input_kernel_2[7], input_tensor_right.fixpos);

  //exit(0);
  task[2]->run(0u);


  vector<Mat> rets;
  int ret_height = input_images[0].first.rows;
  int ret_width = input_images[0].first.cols;
  auto final_tensor = task[2]->getOutputTensor(0u)[0];

  float f_scale = vitis::ai::library::tensor_scale(final_tensor);
  for (size_t b = 0; b < final_tensor.batch; ++b) {
    Mat final_img(final_tensor.height, final_tensor.width, CV_8UC1);
    Mat ret;
    auto final_data = (int8_t*)final_tensor.get_data(b);
    if (f_scale == 1.f) {
      final_img = Mat(Size(final_tensor.width, final_tensor.height), CV_8UC1, (void*)final_data);
    } else {
      for(size_t i = 0; i < final_tensor.width * final_tensor.height; ++i)
        final_img.data[i] = (uint8_t)(final_data[i] * f_scale);
    }
    resize(final_img, ret, cv::Size(ret_width, ret_height));
    rets.push_back(ret);
  }
  return rets;
}



int main(int argc, char* argv[]) {

  auto kernel_name_0 = argv[1];
  auto kernel_name_1 = argv[2];
  auto kernel_name_2 = argv[3];

  Mat img_l = cv::imread(argv[4]);
  Mat img_r = cv::imread(argv[5]);

  // Create a dpu task object.
  vector<unique_ptr<vitis::ai::DpuTask>> task;
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_0));
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_1));
  task.emplace_back( vitis::ai::DpuTask::create(kernel_name_2));

  // Set the mean values and scale values.
  task[0]->setMeanScaleBGR({103.53, 116.28, 123.675},
                        {0.017429, 0.017507, 0.01712475});
  task[1]->setMeanScaleBGR({103.53, 116.28, 123.675},
                        {0.017429, 0.017507, 0.01712475});
  vector<pair<Mat, Mat>> imgs;
  imgs.push_back(make_pair(img_l, img_r));
  imgs.push_back(make_pair(img_l, img_r));
  imgs.push_back(make_pair(img_l, img_r));

  // Execute the FADnet post-processing.
  auto result = FADnet_run(task, imgs);

  imshow("Depth Map",result[0]);
  imshow("Right Image",img_r);
  imshow("Left Image",img_l);
  waitKey(0);
  imwrite("result_fadnet.jpg", result[0]);
  imwrite("image.jpg", img_l);

  auto image_ptr = std::make_shared<o3Image>();
  open3d::io::ReadImageFromJPG("image.jpg", *image_ptr);
  auto depth_ptr = std::make_shared<o3Image>();
  open3d::io::ReadImageFromJPG("result_fadnet.jpg", *depth_ptr);
  
  std::shared_ptr<open3d::geometry::RGBDImage> rgbd_ptr =
          open3d::geometry::RGBDImage::CreateFromColorAndDepth(
              *image_ptr, *depth_ptr); 

  open3d::camera::PinholeCameraIntrinsic intrinsic(
              open3d::camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
  o3PointCloudPtr ptcd_ptr = o3PointCloud::CreateFromRGBDImage(*rgbd_ptr, intrinsic);  
 
  open3d::io::WritePointCloud("pointcloud.pcd",*ptcd_ptr);

  open3d::visualization::DrawGeometries({ptcd_ptr}, "point cloud from rgbd");

  return 0;
}
