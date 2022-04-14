/**
 * @file feature_extract.h
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 1.0
 * @date 2022-03-22
 *
 * @copyright Copyright (c) 2022
 */

#pragma once

#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <mutex>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

typedef pcl::PointXYZI PointType;
const double scan_period = 0.1;
// 指定激光雷达线数，后续需对该变量改写为内部参数
const int num_scan = 16;

/**
   * @brief 对激光雷达原始点滤波，移除过近的点
   * @param[in] x3D 点云
   * @param[in] x3D 点云
   * @param[in] 距离阈值
   * @return void 
   */
template <typename PointT>
void filterCloseDistance(const pcl::PointCloud<PointT> &input_cloud,
                         pcl::PointCloud<PointT> &output_cloud, float thres)
{
    if (&input_cloud != &output_cloud)
    {
        output_cloud.header = input_cloud.header;
        output_cloud.points.resize(input_cloud.points.size());
    }

    size_t cloud_cnt = 0;
    for (size_t i = 0; i < input_cloud.points.size(); ++i)
    {
        if (input_cloud.points[i].x * input_cloud.points[i].x + input_cloud.points[i].y * input_cloud.points[i].y +
                input_cloud.points[i].z * input_cloud.points[i].z <
            thres * thres)
        {
            continue;
        }
        output_cloud.points[cloud_cnt] = input_cloud.points[i];
        cloud_cnt++;
    }

    if (cloud_cnt != input_cloud.points.size())
    {
        output_cloud.points.resize(cloud_cnt);
    }
    output_cloud.height = 1;
    output_cloud.width = static_cast<uint32_t>(cloud_cnt);
    output_cloud.is_dense = true;
}

class FeatureExtract
{
public:
    using Ptr = std::shared_ptr<FeatureExtract>;

    FeatureExtract();

    /**
   * @brief 仅供构造函数使用，参数初始化
   */
    void init();

    /**
   * @brief 计算整帧点云的“偏航角”
   * @param[in] x3D 点云
   * @return void 
   */
    void computeCloudYawAngle(const pcl::PointCloud<pcl::PointXYZ> &point_cloud);

    /**
   * @brief 对整幅雷达点云划分为不同的scan
   * @param[in] x3D 点云
   * @return void 
   */
    void cloudDevideToScan(const pcl::PointCloud<pcl::PointXYZ> &point_cloud);

    /**
   * @brief 返回组合点云指针
   * @param[in] 
   * @return 点云指针 
   */
    pcl::PointCloud<PointType>::Ptr getCombineCloud();

    /**
   * @brief 计算激光点云曲率
   * @param[in] void
   * @return void 
   */
    void computeCurvature();

    /**
   * @brief 提取激光点云点、面特征
   */
    void extractFeature();

    /**
   * @brief 激光scan去畸变 TODO
   */
    void cloudDeskew();

    /**
   * @brief 获取角点特征
   */
    pcl::PointCloud<PointType>::Ptr getCornerFeature();

    /**
   * @brief 获取less角点特征
   */
    pcl::PointCloud<PointType>::Ptr getCornerFeatureLess();

    /**
   * @brief 获取平面特征
   */
    pcl::PointCloud<PointType>::Ptr getSurfaceFeature();

    /**
   * @brief 获取less平面特征
   */
    pcl::PointCloud<PointType>::Ptr getSurfaceFeatureLess();

public:
    float start_angle_;
    float end_angle_;

    std::vector<int> scan_start_id_;
    std::vector<int> scan_end_id_;

    int real_cloud_size_;

    std::mutex mutex_cloud_;

    pcl::PointCloud<PointType>::Ptr combine_cloud_;

    // 计算曲率提取信息点
    float *cloud_curvature_;
    int *cloud_sort_id_;
    int *cloud_neighbor_picked_;
    int *cloud_label_;

    std::mutex mutex_features_;
    // 特征存储变量，同可写成Ptr形式，需要在init()中做初始化
    pcl::PointCloud<PointType>::Ptr corner_feature_sharp_;
    pcl::PointCloud<PointType>::Ptr corner_feature_less_sharp_;
    pcl::PointCloud<PointType>::Ptr surface_feature_flat_;
    pcl::PointCloud<PointType>::Ptr surface_feature_less_flat_;
};