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
#define MIN_PS 7

template <typename T>
void pub_func(T &pl, ros::Publisher &pub, const ros::Time &current_time)
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
   * @brief 对特征点云做网格化滤波处理，
   * @param[in] PointXYZINormal格式点云数据
   * @param[in] 网格滤波的尺寸大小
   * @return void 
   */
void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size);

void down_sampling_voxel(PointVector &pl_feat, double voxel_size);

void plvec_trans_func(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R, Eigen::Vector3d t);

void rosmsg2ptype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt);
