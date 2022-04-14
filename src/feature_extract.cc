/**
 * @file feature_extract.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief
 * @version 1.0
 * @date 2022-03-22
 *
 * @copyright Copyright (c) 2022
 */

#include "feature_extract.h"

FeatureExtract::FeatureExtract()
{
    init();
}

void FeatureExtract::init()
{
    corner_feature_sharp_.reset(new pcl::PointCloud<PointType>());
    corner_feature_less_sharp_.reset(new pcl::PointCloud<PointType>());
    surface_feature_flat_.reset(new pcl::PointCloud<PointType>());
    surface_feature_less_flat_.reset(new pcl::PointCloud<PointType>());

    cloud_curvature_ = new float[40000];
    cloud_sort_id_ = new int[40000];
    cloud_neighbor_picked_ = new int[40000];
    cloud_label_ = new int[40000];
}

// atan2计算，将数据划分为四个象限来计算正切值
void FeatureExtract::computeCloudYawAngle(const pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
    ROS_INFO("Run the computeCloudYawAngle function. ");
    int cloud_size = point_cloud.points.size();
    float start_angle = -std::atan2(point_cloud.points[0].y, point_cloud.points[0].x);
    float end_angle = -std::atan2(point_cloud.points[cloud_size - 1].y, point_cloud.points[cloud_size - 1].x) + 2 * M_PI;

    // end_angle start_angle
    if (end_angle - start_angle > 3 * M_PI)
    {
        end_angle -= 2 * M_PI;
    }
    else if (end_angle - start_angle < M_PI)
    {
        end_angle += 2 * M_PI;
    }
    ROS_INFO("The value of start angle: %f", start_angle);
    ROS_INFO("The value of end angle: %f", end_angle);
    start_angle_ = start_angle;
    end_angle_ = end_angle;
}

void FeatureExtract::cloudDevideToScan(const pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
    ROS_INFO("Run the cloudDevideToScan function. ");
    int cloud_size = point_cloud.points.size();
    int cnt = cloud_size;
    PointType point;
    // 对整体点云按照单独的scan做划分
    std::vector<pcl::PointCloud<PointType>> cloud_scan(num_scan);
    // 具体含义有序需测试
    bool half_scan_flag = false;
    for (int i = 0; i < cloud_size; ++i)
    {
        point.x = point_cloud.points[i].x;
        point.y = point_cloud.points[i].y;
        point.z = point_cloud.points[i].z;

        // x y以激光雷达为中心测量周围空间，相对一某一个测量z值有共同的正切值
        float pitch_angle = std::atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scan_id = 0;
        if (num_scan == 16)
        {
            scan_id = int((pitch_angle + 15) / 2 + 0.5);
            if (scan_id > (num_scan - 1) || scan_id < 0)
            {
                cnt--;
                continue;
            }
        }
        else if (num_scan == 32)
        {
            scan_id = int((pitch_angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scan_id > (num_scan - 1) || scan_id < 0)
            {
                cnt--;
                continue;
            }
        }
        else if (num_scan == 64)
        {
            if (pitch_angle >= -8.83)
                scan_id = int((2 - pitch_angle) * 3.0 + 0.5);
            else
            {
                scan_id = num_scan / 2 + int((-8.83 - pitch_angle) * 2.0 + 0.5);
            }
            if (pitch_angle > 2 || pitch_angle < -24.33 || scan_id > 50 || scan_id < 0)
            {
                cnt--;
                continue;
            }
        }
        else
        {
            ROS_WARN("The scan id is wrong. ");
            // 用以中断程序并输出本句所在文件及行数
            ROS_BREAK();
        }

        float cur_angle = -std::atan2(point.y, point.x);
        if (!half_scan_flag)
        {
            if (cur_angle < start_angle_ - M_PI / 2)
            {
                cur_angle += 2 * M_PI;
            }
            else if (cur_angle > start_angle_ + M_PI * 3 / 2)
            {
                cur_angle -= 2 * M_PI;
            }

            if (cur_angle - start_angle_ > M_PI)
            {
                half_scan_flag = true;
            }
        }
        else
        {
            cur_angle += 2 * M_PI;
            if (cur_angle < end_angle_ - M_PI * 3 / 2)
            {
                cur_angle += 2 * M_PI;
            }
            else if (cur_angle > end_angle_ + M_PI / 2)
            {
                cur_angle -= 2 * M_PI;
            }
        }
        float cur_period_time = (cur_angle - start_angle_) / (end_angle_ - start_angle_);
        point.intensity = scan_id + scan_period * cur_period_time;
        cloud_scan[scan_id].push_back(point);
    }
    int scan_size = cloud_scan[0].points.size();
    ROS_INFO("The size of each scan: %d", scan_size);
    real_cloud_size_ = cnt;
    // 合并各个scan，组成点云
    // pcl::PointCloud<PointType>::Ptr combine_cloud(new pcl::PointCloud<PointType>());
    scan_start_id_.clear();
    scan_end_id_.clear();
    scan_start_id_.resize(num_scan);
    scan_end_id_.resize(num_scan);
    combine_cloud_.reset(new pcl::PointCloud<PointType>());
    for (int i = 0; i < num_scan; ++i)
    {
        std::unique_lock<std::mutex> lock(mutex_cloud_);
        scan_start_id_[i] = combine_cloud_->size() + 5;
        *combine_cloud_ += cloud_scan[i];
        scan_end_id_[i] = combine_cloud_->size() - 6;
    }
}

pcl::PointCloud<PointType>::Ptr FeatureExtract::getCombineCloud()
{
    std::unique_lock<std::mutex> lock(mutex_cloud_);
    return combine_cloud_;
}

void FeatureExtract::computeCurvature()
{

    for (int i = 5; i < real_cloud_size_ - 5; ++i)
    {
        std::unique_lock<std::mutex> lock(mutex_cloud_);
        float diff_x = combine_cloud_->points[i - 5].x + combine_cloud_->points[i - 4].x + combine_cloud_->points[i - 3].x +
                       combine_cloud_->points[i - 2].x + combine_cloud_->points[i - 1].x - 10 * combine_cloud_->points[i].x +
                       combine_cloud_->points[i + 1].x + combine_cloud_->points[i + 2].x + combine_cloud_->points[i + 3].x +
                       combine_cloud_->points[i + 4].x + combine_cloud_->points[i + 5].x;
        float diff_y = combine_cloud_->points[i - 5].y + combine_cloud_->points[i - 4].y + combine_cloud_->points[i - 3].y +
                       combine_cloud_->points[i - 2].y + combine_cloud_->points[i - 1].y - 10 * combine_cloud_->points[i].y +
                       combine_cloud_->points[i + 1].y + combine_cloud_->points[i + 2].y + combine_cloud_->points[i + 3].y +
                       combine_cloud_->points[i + 4].y + combine_cloud_->points[i + 5].y;
        float diff_z = combine_cloud_->points[i - 5].z + combine_cloud_->points[i - 4].z + combine_cloud_->points[i - 3].z +
                       combine_cloud_->points[i - 2].z + combine_cloud_->points[i - 1].z - 10 * combine_cloud_->points[i].z +
                       combine_cloud_->points[i + 1].z + combine_cloud_->points[i + 2].z + combine_cloud_->points[i + 3].z +
                       combine_cloud_->points[i + 4].z + combine_cloud_->points[i + 5].z;

        cloud_curvature_[i] = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        cloud_sort_id_[i] = i;
        // 用以标记该点是否被选取
        cloud_neighbor_picked_[i] = 0;
        // 用以标记特征类型
        cloud_label_[i] = 0;
    }
}

void FeatureExtract::extractFeature()
{
    ROS_INFO("Run the extractFeature function. ");
    for (int i = 0; i < num_scan; ++i)
    {
        std::unique_lock<std::mutex> lock(mutex_features_);
        // 若scan中点过少，放弃对该scan提取特征
        if (scan_end_id_[i] - scan_start_id_[i] < 6)
            continue;
        // corner_feature_sharp_.clear();

        // 对每个scan分六段进行特征提取
        for (int j = 0; j < 6; ++j)
        {
            std::unique_lock<std::mutex> lock(mutex_cloud_);
            // 设定每一个局部分段的起始点
            int start_point = scan_start_id_[i] + (scan_end_id_[i] - scan_start_id_[i]) * j / 6;
            int end_point = scan_start_id_[i] + (scan_end_id_[i] - scan_start_id_[i]) * (j + 1) / 6 - 1;

            // 指定起始地址，按照曲率大小排序，升序排列
            std::sort(cloud_sort_id_ + start_point, cloud_sort_id_ + end_point + 1,
                      [&](int x, int y) { return (cloud_curvature_[x] < cloud_curvature_[y]); });

            int largest_picked_num = 0;
            for (int k = end_point; k >= start_point; --k)
            {
                int index = cloud_sort_id_[k];
                if (cloud_neighbor_picked_[index] == 0 &&
                    cloud_curvature_[index] > 0.1)
                {
                    largest_picked_num++;
                    if (largest_picked_num <= 2)
                    {
                        cloud_label_[index] = 2;
                        corner_feature_sharp_->push_back(combine_cloud_->points[index]);
                        corner_feature_less_sharp_->push_back(combine_cloud_->points[index]);
                    }
                    else if (largest_picked_num <= 20)
                    {
                        cloud_label_[index] = 1;
                        corner_feature_less_sharp_->push_back(combine_cloud_->points[index]);
                    }
                    else
                    {
                        break;
                    }

                    cloud_neighbor_picked_[index] = 1;

                    for (int l = 1; l <= 5; ++l)
                    {
                        float diff_x = combine_cloud_->points[index + l].x - combine_cloud_->points[index + l - 1].x;
                        float diff_y = combine_cloud_->points[index + l].y - combine_cloud_->points[index + l - 1].y;
                        float diff_z = combine_cloud_->points[index + l].z - combine_cloud_->points[index + l - 1].z;
                        if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
                        {
                            break;
                        }
                        cloud_neighbor_picked_[index + l] = 1;
                    }

                    for (int l = -1; l >= -5; --l)
                    {
                        float diff_x = combine_cloud_->points[index + l].x - combine_cloud_->points[index + l + 1].x;
                        float diff_y = combine_cloud_->points[index + l].y - combine_cloud_->points[index + l + 1].y;
                        float diff_z = combine_cloud_->points[index + l].z - combine_cloud_->points[index + l + 1].z;
                        if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
                        {
                            break;
                        }
                        cloud_neighbor_picked_[index + l] = 1;
                    }
                }
            }

            int smallest_picked_num = 0;
            // 以曲率最小开始检索
            for (int k = start_point; k < end_point; ++k)
            {
                int index = cloud_sort_id_[k];
                if (cloud_neighbor_picked_[index] == 0 &&
                    cloud_curvature_[index] < 0.1)
                {
                    cloud_label_[index] = -1;
                    surface_feature_flat_->push_back(combine_cloud_->points[index]);

                    smallest_picked_num++;
                    if (smallest_picked_num >= 4)
                    {
                        break;
                    }

                    cloud_neighbor_picked_[index] = 1;

                    // 对该点周围点以曲率为判别依据进行选取；
                    for (int l = 1; l <= 5; ++l)
                    {
                        float diff_x = combine_cloud_->points[index + l].x - combine_cloud_->points[index + l - 1].x;
                        float diff_y = combine_cloud_->points[index + l].y - combine_cloud_->points[index + l - 1].y;
                        float diff_z = combine_cloud_->points[index + l].z - combine_cloud_->points[index + l - 1].z;
                        if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
                        {
                            break;
                        }
                        cloud_neighbor_picked_[index + l] = 1;
                    }
                    for (int l = -1; l >= -5; --l)
                    {
                        float diff_x = combine_cloud_->points[index + l].x - combine_cloud_->points[index + l + 1].x;
                        float diff_y = combine_cloud_->points[index + l].y - combine_cloud_->points[index + l + 1].y;
                        float diff_z = combine_cloud_->points[index + l].z - combine_cloud_->points[index + l + 1].z;
                        if (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z > 0.05)
                        {
                            break;
                        }
                        cloud_neighbor_picked_[index + l] = 1;
                    }
                }
            }

            pcl::PointCloud<PointType>::Ptr surf_feature_less_flat_scan(new pcl::PointCloud<PointType>());
            for (int i = start_point; i <= end_point; ++i)
            {
                if (cloud_label_[i] <= 0)
                {
                    surf_feature_less_flat_scan->push_back(combine_cloud_->points[i]);
                }
            }

            // 对surf_feature_less_flat_scan执行滤波，以获得最终的surface_feature_less_flat_特征
            pcl::PointCloud<PointType> surf_feature_less_flat_scan_downfilter;
            pcl::VoxelGrid<PointType> down_filter;
            down_filter.setInputCloud(surf_feature_less_flat_scan);
            down_filter.setLeafSize(0.2, 0.2, 0.2);
            down_filter.filter(surf_feature_less_flat_scan_downfilter);

            *surface_feature_less_flat_ += surf_feature_less_flat_scan_downfilter;
        }
        int corner_feature_size;
        corner_feature_size = corner_feature_sharp_->points.size();

        ROS_INFO("The corner features size: %d", corner_feature_size);
        ROS_INFO("Feature extracted successfully. ");
    }
}

void FeatureExtract::cloudDeskew() {}

pcl::PointCloud<PointType>::Ptr FeatureExtract::getCornerFeature()
{
    pcl::PointCloud<PointType>::Ptr corner_features;
    std::unique_lock<std::mutex> lock(mutex_features_);
    corner_features = corner_feature_sharp_;
    return corner_features;
}

pcl::PointCloud<PointType>::Ptr FeatureExtract::getCornerFeatureLess()
{
    pcl::PointCloud<PointType>::Ptr corner_features_less;
    std::unique_lock<std::mutex> lock(mutex_features_);
    corner_features_less = corner_feature_less_sharp_;
    return corner_features_less;
}

pcl::PointCloud<PointType>::Ptr FeatureExtract::getSurfaceFeature()
{
    pcl::PointCloud<PointType>::Ptr surface_features;
    std::unique_lock<std::mutex> lock(mutex_features_);
    surface_features = surface_feature_flat_;
    return surface_features;
}

pcl::PointCloud<PointType>::Ptr FeatureExtract::getSurfaceFeatureLess()
{
    pcl::PointCloud<PointType>::Ptr surface_features_less;
    std::unique_lock<std::mutex> lock(mutex_features_);
    surface_features_less = surface_feature_less_flat_;
    return surface_features_less;
}
