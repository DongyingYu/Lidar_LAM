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


using namespace std;

// 用以存储哈希表键值
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

// 用以存储哈希表value
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
            return ((hash<int64_t>()(s.x_) ^ (hash<int64_t>()(s.y_) << 1)) >> 1) ^ (hash<int64_t>()(s.z_) << 1);
        }
    };
} // namespace std

struct PointCount
{
    float xyz_[3];
    int count_ = 0;
};

// P_fix in the paper
// Summation of P_fix
class SigmaVector
{
public:
    Eigen::Matrix3d sigma_vTv_;
    Eigen::Vector3d sigma_vi_;
    int sigma_size_;

    SigmaVector();

    void toZero();
};

