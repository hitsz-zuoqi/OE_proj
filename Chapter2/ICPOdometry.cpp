/*
 * ICPOdometry.cpp
 *
 *  Created on: 17 Sep 2012
 *      Author: thomas
 */

#include "ICPOdometry.h"

ICPOdometry::ICPOdometry(int width, int height, float cx, float cy, float fx,
                         float fy, float distThresh, float angleThresh)
    : lastError(0), lastInliers(width * height), dist_thresh(distThresh),
      angle_thresh(angleThresh), width(width), height(height), cx(cx), cy(cy),
      fx(fx), fy(fy) {
  // 这两句没懂
  sumData.create(MAX_THREADS);
  outData.create(1);
  // intr 应该是相机内参的数据结构
  intr.cx = cx;
  intr.cy = cy;
  intr.fx = fx;
  intr.fy = fy;
  // static 变量include头文件之后可以直接用
  // reserve函数之后一个参数，即需要预留的容器的空间；
  // resize函数可以有两个参数，第一个参数是容器新的大小， 第二个参数是要加入容器中的新元素，如果这个参数被省略，那么就调用元素对象的默认构造函数
  // 开辟大小为3个int数据类型的内存但是没有初始化
  iterations.reserve(NUM_PYRS);
  // 开辟内存同时使用默认构造函数初始化resize()
  depth_tmp.resize(NUM_PYRS);

  vmaps_prev.resize(NUM_PYRS);
  nmaps_prev.resize(NUM_PYRS);

  vmaps_curr.resize(NUM_PYRS);
  nmaps_curr.resize(NUM_PYRS);

  for (int i = 0; i < NUM_PYRS; ++i) {
    // 这里应该是表示右移多少位，右移1位就是除以二
    int pyr_rows = height >> i;
    int pyr_cols = width >> i;
    // 三层金字塔的深度图
    depth_tmp[i].create(pyr_rows, pyr_cols);
    // 前一帧三层金字塔的TSDF
    vmaps_prev[i].create(pyr_rows * 3, pyr_cols);
    // 前一帧三层金字塔的法向量
    nmaps_prev[i].create(pyr_rows * 3, pyr_cols);
    // 当前帧的三层金字塔的TSDF
    vmaps_curr[i].create(pyr_rows * 3, pyr_cols);
    // 当前帧的三层金字塔的法向量
    nmaps_curr[i].create(pyr_rows * 3, pyr_cols);
  }
}
// 析构函数
ICPOdometry::~ICPOdometry() {}
// 初始化ICP算法
void ICPOdometry::initICP(unsigned short *depth, const float depthCutoff) {
  // 在开辟的内存上传金字塔底层的深度图
  depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);
  // 逐层计算深度金字塔
  for (int i = 1; i < NUM_PYRS; ++i) {
    pyrDown(depth_tmp[i - 1], depth_tmp[i]);
  }
  // 逐层计算TSDF金字塔和法向量金字塔
  for (int i = 0; i < NUM_PYRS; ++i) {
    createVMap(intr(i), depth_tmp[i], vmaps_curr[i], depthCutoff);
    createNMap(vmaps_curr[i], nmaps_curr[i]);
  }
  // cuda 设备线程同步
  cudaDeviceSynchronize();
}
// 初始化ICP模型，好像跟上面那个函数一样。。。
void ICPOdometry::initICPModel(unsigned short *depth, const float depthCutoff) {
  depth_tmp[0].upload(depth, sizeof(unsigned short) * width, height, width);

  for (int i = 1; i < NUM_PYRS; ++i) {
    pyrDown(depth_tmp[i - 1], depth_tmp[i]);
  }

  for (int i = 0; i < NUM_PYRS; ++i) {
    createVMap(intr(i), depth_tmp[i], vmaps_prev[i], depthCutoff);
    createNMap(vmaps_prev[i], nmaps_prev[i]);
  }

  cudaDeviceSynchronize();
}
// ICP 得到增量式的转移矩阵T
void ICPOdometry::getIncrementalTransformation(Sophus::SE3 &T_prev_curr,
                                               int threads, int blocks) {
  iterations[0] = 10; // 底层迭代次数
  iterations[1] = 5;  // 第二层迭代次数
  iterations[2] = 4;  // 第三层迭代次数
  // 金字塔层数由高到低，计算由粗到细
  for (int i = NUM_PYRS - 1; i >= 0; i--) {
    // 每层进行迭代优化
    for (int j = 0; j < iterations[i]; j++) {
      // 记录残差
      float residual_inliers[2]; 
      // ICP本质是解AX+b=0，用增量式去解可以用GPU加速
      Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
      Eigen::Matrix<float, 6, 1> b_icp;
      // GPU里面并行计算sum（A）和sum（b），这里具体的实现还要看cu函数
      estimateStep(T_prev_curr.rotation_matrix().cast<float>().eval(),
                   T_prev_curr.translation().cast<float>().eval(),
                   vmaps_curr[i], nmaps_curr[i], intr(i), vmaps_prev[i],
                   nmaps_prev[i], dist_thresh, angle_thresh, sumData, outData,
                   A_icp.data(), b_icp.data(), &residual_inliers[0], threads,
                   blocks);
      // 记录error和inliers
      lastError = sqrt(residual_inliers[0]) / residual_inliers[1];
      lastInliers = residual_inliers[1];
      // 调用solve函数解update的增量李代数
      const Eigen::Matrix<double, 6, 1> update =
          A_icp.cast<double>().ldlt().solve(b_icp.cast<double>());
      // 计算更新后的转移矩阵
      T_prev_curr = Sophus::SE3::exp(update) * T_prev_curr;
    }
  }
}
