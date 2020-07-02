/*
 * ICPOdometry.h
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */
// 防止头文件重复定义
// #pragma once 也可以
#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "Cuda/internal.h"
// 相机位姿李代数头文件
#include <sophus/se3.h>
// std::vector 
#include <vector>
// Eigen的core和geometry，包含矩阵的定义和操作函数
#include <Eigen/Core>
#include <Eigen/Geometry>
// 声明ICPOdometry类
class ICPOdometry {
public:
  // 让Eigen在内存对齐
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // 与类同名的构造函数，需要的变量是图片的宽和高(resolution),相机内参cx，cy，fx，fy，以及ICP拒绝outliers的最大距离和最大角度
  ICPOdometry(int width, int height, float cx, float cy, float fx, float fy,
              float distThresh = 0.10f,
              float angleThresh = sinf(20.f * 3.14159254f / 180.f));
  // 析构函数，用来自动释放内存
  virtual ~ICPOdometry();
  // ICP初始化depth 0-255 , depthcutoff 超过这个阈值的深度直接设置为0
  void initICP(unsigned short *depth, const float depthCutoff = 20.0f);
  // 初始化，不知道这里为什么有两个一样的初始化函数？
  void initICPModel(unsigned short *depth, const float depthCutoff = 20.0f);
  // 计算增量式的delta T，是从前一帧到当前帧的转移矩阵，threads是线程数，blocks是cuda的block个数
  void getIncrementalTransformation(Sophus::SE3 &T_prev_curr, int threads,
                                    int blocks);
  // 公有成员变量 
  float lastError;
  float lastInliers;

private:
  std::vector<DeviceArray2D<unsigned short>> depth_tmp;

  std::vector<DeviceArray2D<float>> vmaps_prev;
  std::vector<DeviceArray2D<float>> nmaps_prev;

  std::vector<DeviceArray2D<float>> vmaps_curr;
  std::vector<DeviceArray2D<float>> nmaps_curr;

  Intr intr;

  DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> sumData;
  DeviceArray<Eigen::Matrix<float, 29, 1, Eigen::DontAlign>> outData;
  // 金字塔采样层数，静态的写死
  static const int NUM_PYRS = 3;
  // 对三层金字塔的不同ICP迭代次数
  std::vector<int> iterations;
  // 距离阈值和角度阈值
  float dist_thresh;
  float angle_thresh;
  // 图像分辨率以及相机内参
  const int width;
  const int height;
  const float cx, cy, fx, fy;
};

#endif /* ICPODOMETRY_H_ */
