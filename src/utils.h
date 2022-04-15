/**
 * @file utils.h
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-27
 *
 * @copyright Copyright (c) 2022
 */

#pragma once

#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include "basic_data.h"

typedef std::vector<Eigen::Vector3d> PointVector;
typedef pcl::PointXYZINormal PointType;
#define MIN_POINT_SIZE 7

/**
   * @brief 发布点云topic数据
   * @param[in] 类型为T的点云数据
   * @param[in] 定义好的数据发布器
   * @param[in] 当前ros时间
   * @return void 
   */
template <typename T>
void pubFunction(T &pl, ros::Publisher &pub, const ros::Time &current_time)
{
  pl.height = 1;
  pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "laser_init";
  output.header.stamp = current_time;
  pub.publish(output);
}

/**
   * @brief 对特征点云做网格化滤波处理
   * @param[in] PointXYZINormal格式点云数据
   * @param[in] 网格滤波的尺寸大小
   * @return void 
   */
void downSamplingVoxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size);

/**
   * @brief 对点向量做网格化滤波处理
   * @param[in/out] 三维点向量容器
   * @param[in] 网格滤波的尺寸大小
   * @return void 
   */
void downSamplingVoxel(PointVector &pl_feat, double voxel_size);

/**
   * @brief 对特征点做位姿转换
   * @param[in] 原始点坐标容器
   * @param[out] 经转换矩阵作用后的坐标容器
   * @param[in] 旋转矩阵
   * @param[in] 平移向量
   * @return void 
   */
void pointvecTransform(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R, Eigen::Vector3d t);

/**
   * @brief 点云数据转换，PointCloud2-->PointType
   * @param[in] ros中接收到的PointCloud2格式数据
   * @param[in] PCL中点云格式
   * @return void 
   */
void rosmsgToPointtype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt);

/**
   * @brief 计算协方差矩阵
   * @param[in] ros中接收到的PointCloud2格式数据
   * @param[in] PCL中点云格式
   * @return void 
   */
Eigen::Matrix3d computeCovMat(const pcl::PointCloud<PointType> &point_cloud, int closest_point_cnt,
                              const vector<int> &point_search_id);
