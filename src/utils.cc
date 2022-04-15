/**
 * @file utils.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-27
 *
 * @copyright Copyright (c) 2022
 */

#include <iostream>
#include "utils.h"

void downSamplingVoxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size)
{
    ROS_INFO("Run the downSamplingVoxel function 1. ");
    if (voxel_size < 0.01)
    {
        return;
    }

    unordered_map<VoxelStructure, PointCount> feat_map;
    uint plsize = pl_feat.size();

    for (uint i = 0; i < plsize; i++)
    {
        PointType &p_c = pl_feat[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = p_c.data[j] / voxel_size;
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VoxelStructure position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end())
        {
            iter->second.xyz_[0] += p_c.x;
            iter->second.xyz_[1] += p_c.y;
            iter->second.xyz_[2] += p_c.z;
            iter->second.count_++;
        }
        else
        {
            PointCount anp;
            anp.xyz_[0] = p_c.x;
            anp.xyz_[1] = p_c.y;
            anp.xyz_[2] = p_c.z;
            anp.count_ = 1;
            feat_map[position] = anp;
        }
    }

    plsize = feat_map.size();
    pl_feat.clear();
    pl_feat.resize(plsize);

    uint i = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter)
    {
        pl_feat[i].x = iter->second.xyz_[0] / iter->second.count_;
        pl_feat[i].y = iter->second.xyz_[1] / iter->second.count_;
        pl_feat[i].z = iter->second.xyz_[2] / iter->second.count_;
        i++;
    }
}

void downSamplingVoxel(PointVector &pl_feat, double voxel_size)
{
    ROS_INFO("Run the downSamplingVoxel function. ");
    unordered_map<VoxelStructure, PointCount> feat_map;
    uint plsize = pl_feat.size();

    // 借助于哈希网格查找检测以实现对数据的滤波
    for (uint i = 0; i < plsize; i++)
    {
        Eigen::Vector3d &p_c = pl_feat[i];
        double loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = p_c[j] / voxel_size;
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }

        VoxelStructure position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end())
        {
            iter->second.xyz_[0] += p_c[0];
            iter->second.xyz_[1] += p_c[1];
            iter->second.xyz_[2] += p_c[2];
            iter->second.count_++;
        }
        else
        {
            PointCount anp;
            anp.xyz_[0] = p_c[0];
            anp.xyz_[1] = p_c[1];
            anp.xyz_[2] = p_c[2];
            anp.count_ = 1;
            feat_map[position] = anp;
        }
    }

    plsize = feat_map.size();
    pl_feat.resize(plsize);

    uint i = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter)
    {
        pl_feat[i][0] = iter->second.xyz_[0] / iter->second.count_;
        pl_feat[i][1] = iter->second.xyz_[1] / iter->second.count_;
        pl_feat[i][2] = iter->second.xyz_[2] / iter->second.count_;
        i++;
    }
}

void pointvecTransform(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R, Eigen::Vector3d t)
{
    uint orig_size = orig.size();
    tran.resize(orig_size);

    for (uint i = 0; i < orig_size; i++)
    {
        tran[i] = R * orig[i] + t;
    }
}

// Convert PointCloud2 to PointType
void rosmsgToPointtype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt)
{
    pcl::PointCloud<pcl::PointXYZI> pl;
    pcl::fromROSMsg(pl_msg, pl);

    uint asize = pl.size();
    plt.resize(asize);

    for (uint i = 0; i < asize; i++)
    {
        plt[i].x = pl[i].x;
        plt[i].y = pl[i].y;
        plt[i].z = pl[i].z;
        plt[i].intensity = pl[i].intensity;
    }
}

Eigen::Matrix3d computeCovMat(const pcl::PointCloud<PointType> &point_cloud, int closest_point_cnt,
                              const vector<int> &point_search_id)
{
    // 计算平面特征的协方差矩阵
    // 对应于论文中计算平均点坐标及协方差矩阵A
    Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center_coor(0, 0, 0);
    for (int j = 0; j < closest_point_cnt; j++)
    {
        Eigen::Vector3d tvec;
        tvec[0] = point_cloud[point_search_id[j]].x;
        tvec[1] = point_cloud[point_search_id[j]].y;
        tvec[2] = point_cloud[point_search_id[j]].z;
        cov_mat += tvec * tvec.transpose();
        center_coor += tvec;
    }

    center_coor /= closest_point_cnt;
    cov_mat -= closest_point_cnt * center_coor * center_coor.transpose();
    cov_mat /= closest_point_cnt;
}