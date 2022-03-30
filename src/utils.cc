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


void down_sampling_voxel(pcl::PointCloud<PointType> &pl_feat, double voxel_size)
{
    ROS_INFO("Run the down_sampling_voxel function 1. ");
    if (voxel_size < 0.01)
    {
        return;
    }

    unordered_map<VoxelStrcuture, PointCount> feat_map;
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

        VoxelStrcuture position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
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

void down_sampling_voxel(PointVector &pl_feat, double voxel_size)
{
    ROS_INFO("Run the down_sampling_voxel function. ");
    unordered_map<VoxelStrcuture, PointCount> feat_map;
    uint plsize = pl_feat.size();

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

        VoxelStrcuture position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
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

void plvec_trans_func(vector<Eigen::Vector3d> &orig, vector<Eigen::Vector3d> &tran, Eigen::Matrix3d R, Eigen::Vector3d t)
{
    uint orig_size = orig.size();
    tran.resize(orig_size);

    for (uint i = 0; i < orig_size; i++)
    {
        tran[i] = R * orig[i] + t;
    }
}

// Convert PointCloud2 to PointType
void rosmsg2ptype(const sensor_msgs::PointCloud2 &pl_msg, pcl::PointCloud<PointType> &plt)
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
