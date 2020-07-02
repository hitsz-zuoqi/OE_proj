// some functions of ICP
#include "ICPOdometry.h"
// for time counting
#include <chrono>
// for reading and saving files
#include <fstream>
// I/O 流控制工具，控制输出的格式
#include <iomanip>
// pangolin 依赖（一款开源的OPENGL显示库，可以用来视频显示、而且开发容易）
#include <pangolin/image/image_io.h>
// input file stream 
std::ifstream asFile;
// directory 自然就是文件路径啦
std::string directory;
// tokenize 用来分词，对参数文件进行读取
void tokenize(const std::string &str, std::vector<std::string> &tokens,
              std::string delimiters = " ") {
  // vector 清零
  tokens.clear();
  // 找到除掉空格之外的字符串的第一个字符
  std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
  // 找到第一次出现空格的字符,从lastpos开始
  std::string::size_type pos = str.find_first_of(delimiters, lastPos);
  // string::npos就是字符串结束了
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(str.substr(lastPos, pos - lastPos));
    lastPos = str.find_first_not_of(delimiters, pos);
    pos = str.find_first_of(delimiters, lastPos);
  }
}
// 读取深度图片，返回
uint64_t loadDepth(pangolin::Image<unsigned short> &depth) {
  std::string currentLine;
  std::vector<std::string> tokens;
  std::vector<std::string> timeTokens;

  do {
    getline(asFile, currentLine);
    tokenize(currentLine, tokens);
  } while (tokens.size() > 2);

  if (tokens.size() == 0)
    return 0;
  // 深度图的存储位置
  std::string depthLoc = directory;
  depthLoc.append(tokens[1]);
  // pangolin读取深度图
  pangolin::TypedImage depthRaw =
      pangolin::LoadImage(depthLoc, pangolin::ImageFileTypePng);
  // 为16位深度图开辟内存
  pangolin::Image<unsigned short> depthRaw16(
      (unsigned short *)depthRaw.ptr, depthRaw.w, depthRaw.h,
      depthRaw.w * sizeof(unsigned short));
  // 从文件路径中以"."为分割
  tokenize(tokens[0], timeTokens, ".");
  
  std::string timeString = timeTokens[0];
  timeString.append(timeTokens[1]);

  uint64_t time;
  std::istringstream(timeString) >> time;

  for (unsigned int i = 0; i < 480; i++) {
    for (unsigned int j = 0; j < 640; j++) {
      depth.RowPtr(i)[j] = depthRaw16(j, i) / 5;
    }
  }
  // 销毁内存
  depthRaw.Dealloc();
  // 返回时间
  return time;
}
// 输出 freiburg数据集的currentPose到文件
void outputFreiburg(const std::string filename, const uint64_t &timestamp,
                    const Eigen::Matrix4f &currentPose) {
  std::ofstream file;
  file.open(filename.c_str(), std::fstream::app);

  std::stringstream strs;

  strs << std::setprecision(6) << std::fixed << (double)timestamp / 1000000.0
       << " ";
  // 得到旋转和平移向量
  Eigen::Vector3f trans = currentPose.topRightCorner(3, 1);
  Eigen::Matrix3f rot = currentPose.topLeftCorner(3, 3);
  
  file << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";
  // 把旋转矩阵转化成四元数
  Eigen::Quaternionf currentCameraRotation(rot);
  // 四元数输出到文件
  file << currentCameraRotation.x() << " " << currentCameraRotation.y() << " "
       << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";
  // 关闭文件
  file.close();
}
// 得到现在的时间
uint64_t getCurrTime() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}
// 主函数入口
int main(int argc, char *argv[]) {
  assert((argc == 2 || argc == 3) &&
         "Please supply the depth.txt dir as the first argument");

  directory.append(argv[1]);

  if (directory.at(directory.size() - 1) != '/') {
    directory.append("/");
  }
  // TUM数据集有一个depth.txt
  std::string associationFile = directory;
  associationFile.append("depth.txt");
  // 打开depth.txt
  asFile.open(associationFile.c_str());

  pangolin::ManagedImage<unsigned short> firstData(640, 480);
  pangolin::ManagedImage<unsigned short> secondData(640, 480);

  pangolin::Image<unsigned short> firstRaw(firstData.w, firstData.h,
                                           firstData.pitch,
                                           (unsigned short *)firstData.ptr);
  pangolin::Image<unsigned short> secondRaw(secondData.w, secondData.h,
                                            secondData.pitch,
                                            (unsigned short *)secondData.ptr);

  ICPOdometry icpOdom(640, 480, 319.5, 239.5, 528, 528);

  assert(!asFile.eof() && asFile.is_open());

  loadDepth(firstRaw);
  uint64_t timestamp = loadDepth(secondRaw);

  Sophus::SE3 T_wc_prev;
  Sophus::SE3 T_wc_curr;

  std::ofstream file;
  file.open("output.poses", std::fstream::out);
  file.close();

  cudaDeviceProp prop;

  cudaGetDeviceProperties(&prop, 0);

  std::string dev(prop.name);

  std::cout << dev << std::endl;

  float mean = std::numeric_limits<float>::max();
  int count = 0;

  int threads = 224;
  int blocks = 96;

  int bestThreads = threads;
  int bestBlocks = blocks;
  float best = mean;

  if (argc == 3) {
    std::string searchArg(argv[2]);
    // 这里还包含了一个搜索GPU最佳配置的操作
    if (searchArg.compare("-v") == 0) {
      std::cout
          << "Searching for the best thread/block configuration for your GPU..."
          << std::endl;
      std::cout << "Best: " << bestThreads << " threads, " << bestBlocks
                << " blocks (" << best << "ms)";
      std::cout.flush();

      float counter = 0;

      for (threads = 16; threads <= 512; threads += 16) {
        for (blocks = 16; blocks <= 512; blocks += 16) {
          mean = 0.0f;
          count = 0;

          for (int i = 0; i < 5; i++) {
            icpOdom.initICPModel(firstRaw.ptr);
            icpOdom.initICP(secondRaw.ptr);

            uint64_t tick = getCurrTime();

            T_wc_prev = T_wc_curr;

            Sophus::SE3 T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

            icpOdom.getIncrementalTransformation(T_prev_curr, threads, blocks);

            T_wc_curr = T_wc_prev * T_prev_curr;

            uint64_t tock = getCurrTime();

            mean = (float(count) * mean + (tock - tick) / 1000.0f) /
                   float(count + 1);
            count++;
          }

          counter++;

          if (mean < best) {
            best = mean;
            bestThreads = threads;
            bestBlocks = blocks;
          }

          std::cout << "\rBest: " << bestThreads << " threads, " << bestBlocks
                    << " blocks (" << best << "ms), "
                    << int((counter / 1024.f) * 100.f) << "%    ";
          std::cout.flush();
        }
      }

      std::cout << std::endl;
    }
  }

  threads = bestThreads;
  blocks = bestBlocks;

  mean = 0.0f;
  count = 0;
  // 置0
  T_wc_prev = Sophus::SE3();
  T_wc_curr = Sophus::SE3();
  // 读到文件末尾
  while (!asFile.eof()) {
    icpOdom.initICPModel(firstRaw.ptr);
    icpOdom.initICP(secondRaw.ptr);
    
    uint64_t tick = getCurrTime();

    T_wc_prev = T_wc_curr;

    Sophus::SE3 T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

    icpOdom.getIncrementalTransformation(T_prev_curr, threads, blocks);

    T_wc_curr = T_wc_prev * T_prev_curr;

    uint64_t tock = getCurrTime();

    mean = (float(count) * mean + (tock - tick) / 1000.0f) / float(count + 1);
    count++;

    std::cout << std::setprecision(4) << std::fixed << "\rICP: " << mean
              << "ms";
    std::cout.flush();

    std::swap(firstRaw, secondRaw);

    outputFreiburg("output.poses", timestamp, T_wc_curr.matrix().cast<float>());

    timestamp = loadDepth(secondRaw);
  }

  std::cout << std::endl;

  std::cout << "ICP speed: " << int(1000.f / mean) << "Hz" << std::endl;

  return 0;
}
