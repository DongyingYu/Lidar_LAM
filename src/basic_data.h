/**
 * @file basic_data.h
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */

#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <opencv/cv.h>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <ros/ros.h>

using namespace std;

/**
   * @brief 用以存储哈希表键值，自定义网格结构的数据存储方式
   *        因在unordered_map中存世我们自定义类型数据，
   *        故这里需要对==运算符进行重载
   */
class VoxelStrcuture
{
public:
    int64_t x_, y_, z_;

    VoxelStrcuture(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0);
    // 运算符重载
    bool operator==(const VoxelStrcuture &other) const
    {
        return (x_ == other.x_ && y_ == other.y_ && z_ == other.z_);
    }
};

/**
   * @brief 为便于快速的查找，本书设计建立哈希表使用unordered_map，
   *        其特性为：若元素为自定义类型时需要提供自定义的hash函数，
   *        unordered_map内部元素无序，而map是有序的，
   *        该结构体即为自定义的哈希函数
   */
namespace std
{
    // 模板专门化，模板声明中不包含模板参数，所有模板参数均以指定好，<>中没有剩余参数
    template <>
    struct hash<VoxelStrcuture>
    {
        size_t operator()(const VoxelStrcuture &s) const
        {
            using std::hash;
            using std::size_t;
            // ROS_INFO("Run the hash vox function.");
            return ((hash<int64_t>()(s.x_) ^ (hash<int64_t>()(s.y_) << 1)) >> 1) ^ (hash<int64_t>()(s.z_) << 1);
        }
    };
} // namespace std

struct PointCount
{
    float xyz_[3];
    int count_ = 0;
};

/**
   * @brief 当滑动窗口已满，最早scan中的点被存入固定点集Pfix，
   *        该类即用来存储这些点
   */
class SigmaVector
{
public:
    SigmaVector();

    void toZero();

    Eigen::Matrix3d sigma_vTv_;
    Eigen::Vector3d sigma_vi_;
    int sigma_size_;
};
