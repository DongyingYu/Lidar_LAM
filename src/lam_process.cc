/**
 * @file lam_process.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 定位与建图模块主函数，topic发送接受，回调函数
 * @version 1.0
 * @date 2022-04-12
 *
 * @copyright Copyright (c) 2022
 */
#include "lam_process.h"

/**
   * @brief 默认构造函数
   */
LamProcess::LamProcess()
{
    init();
}

void LamProcess::init()
{
    // 此方式输入数据格式为：w,x,y,z
    qua_incre_ = {1.0, 0.0, 0.0, 0.0};
    // 此方式输入数据格式为：x,y,z,w
    // qua_incre_ = Vector4d(0.0, 0.0, 0.0, 1.0);
    trans_incre_ = Vector3d(0.0, 0.0, 0.0);

    ROS_INFO("The parameters of q: %f, %f, %f, %f", qua_incre_.x(), qua_incre_.y(), qua_incre_.z(), qua_incre_.w());
}

void LamProcess::cutVoxel(unordered_map<VoxelStructure, OctoTree *> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat,
                          Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
    uint cloud_size = pl_feat->size();
    for (uint i = 0; i < cloud_size; i++)
    {
        // 将特征点云转换至世界坐标系
        PointType &ori_point = pl_feat->points[i];
        Eigen::Vector3d pvec_orig(ori_point.x, ori_point.y, ori_point.z);
        Eigen::Vector3d pvec_tran = R_p * pvec_orig + t_p;

        // 确定哈希表键值
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            ROS_INFO("cutVoxel test ...");
            loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }
        ROS_INFO("The value of loc_xyz %f, %f, %f ", loc_xyz[0], loc_xyz[1], loc_xyz[2]);
        // 哈希编码得到表示该位置的唯一键值
        VoxelStructure position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

        // 找到对应的网格位置
        // 在当前的哈希表中查找该键值，若查找到，将新点的数据存入容器中
        auto iter = feat_map.find(position);
        if (iter != feat_map.end())
        {
            ROS_INFO("cutVoxel --> Found the point. ");
            iter->second->point_vec_orig_[fnum]->push_back(pvec_orig);
            iter->second->point_vec_tran_[fnum]->push_back(pvec_tran);
            iter->second->is2opt_ = true;
        }
        else
        {
            // 若未找到，则将当前数据存入八叉树中，所有的八叉树均在哈希表中建立索引
            // 新建网格数据，对每一个新特征点执行此操作，以计算出的position作为网格的中心点坐标
            ROS_INFO("cutVoxel --> Not found the point. ");
            OctoTree *ot = new OctoTree(feattype, capacity);
            ot->point_vec_orig_[fnum]->push_back(pvec_orig);
            ot->point_vec_tran_[fnum]->push_back(pvec_tran);

            // 设定网格中心坐标
            ot->voxel_center_[0] = (0.5 + position.x_) * voxel_size[feattype];
            ot->voxel_center_[1] = (0.5 + position.y_) * voxel_size[feattype];
            ot->voxel_center_[2] = (0.5 + position.z_) * voxel_size[feattype];
            // 取边长的四分之一值
            ot->quater_length_ = voxel_size[feattype] / 4.0;
            feat_map[position] = ot;
        }
    }
}

// 注意：这里的PointType是：PointXYZINormal
void LamProcess::cloudDeskew(PointType const *const point_in, PointType *const point_out)
{
    // 插值比率
    double period_ratio;
    if (deskew_flag_)
    {
        period_ratio = (point_in->intensity - int(point_in->intensity)) / scan_period_;
    }
    else
    {
        period_ratio = 1.0;
    }

    Eigen::Quaterniond qua_temp = Eigen::Quaterniond::Identity().slerp(period_ratio, qua_incre_);
    Eigen::Vector3d t_point_last = period_ratio * trans_incre_;

    Eigen::Vector3d orig_point(point_in->x, point_in->y, point_in->z);
    Eigen::Vector3d deskew_point = qua_temp * orig_point + t_point_last;
    point_out->x = deskew_point.x();
    point_out->y = deskew_point.y();
    point_out->z = deskew_point.z();
    point_out->intensity = point_in->intensity;
    point_out->normal_x = point_in->normal_x;
    point_out->normal_y = point_in->normal_y;
    point_out->normal_z = point_in->normal_z;
}

void LamProcess::cloudDeskew(pcl::PointCloud<PointType> &point_in)
{
    // 插值比率
    double period_ratio;

    uint point_size = point_in.size();
    for (uint i = 0; i < point_size; ++i)
    {
        if (deskew_flag_)
        {
            period_ratio = (point_in[i].intensity - int(point_in[i].intensity)) / scan_period_;
        }
        else
        {
            period_ratio = 1.0;
        }

        Eigen::Quaterniond qua_temp = Eigen::Quaterniond::Identity().slerp(period_ratio, qua_incre_);
        Eigen::Vector3d t_point_last = period_ratio * trans_incre_;

        Eigen::Vector3d orig_point(point_in[i].x, point_in[i].y, point_in[i].z);
        Eigen::Vector3d deskew_point = qua_temp * orig_point + t_point_last;
        point_in[i].x = deskew_point.x();
        point_in[i].y = deskew_point.y();
        point_in[i].z = deskew_point.z();
    }
}

void LamProcess::cloudPositionTrans(PointType const *const point_in, PointType *const point_out)
{
    PointType deskew_point_temp;
    cloudDeskew(point_in, &deskew_point_temp);

    Eigen::Vector3d deskew_point(deskew_point_temp.x, deskew_point_temp.y, deskew_point_temp.z);
    Eigen::Vector3d trans_point = qua_incre_.inverse() * (deskew_point - trans_incre_);

    point_out->x = trans_point.x();
    point_out->y = trans_point.y();
    point_out->z = trans_point.z();
    point_out->intensity = int(point_in->intensity);
    point_out->normal_x = point_in->normal_x;
    point_out->normal_y = point_in->normal_y;
    point_out->normal_z = point_in->normal_z;
}

void LamProcess::computeCovMat(const pcl::PointCloud<PointType> &point_cloud, int closest_point_cnt,
                              const vector<int> &point_search_id)
{
    // 计算平面特征的协方差矩阵
    // 对应于论文中计算平均点坐标及协方差矩阵A
    cov_mat_ = Eigen::Matrix3d::Zero();
    center_coor_ = Vector3d(0.0, 0.0, 0.0);
    for (int j = 0; j < closest_point_cnt; j++)
    {
        Eigen::Vector3d tvec;
        tvec[0] = point_cloud[point_search_id[j]].x;
        tvec[1] = point_cloud[point_search_id[j]].y;
        tvec[2] = point_cloud[point_search_id[j]].z;
        cov_mat_ += tvec * tvec.transpose();
        center_coor_ += tvec;
    }

    center_coor_ /= closest_point_cnt;
    cov_mat_ -= closest_point_cnt * center_coor_ * center_coor_.transpose();
    cov_mat_ /= closest_point_cnt;
}
