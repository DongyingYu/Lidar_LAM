/**
 * @file lam_process.h
 * @author Dongying (yudong2817@sina.com)
 * @brief 定位与建图模块主函数，topic发送接受，回调函数
 * @version 1.0
 * @date 2022-04-11
 *
 * @copyright Copyright (c) 2022
 */

#pragma once

#include "localization_and_mapping.h"
#include <ros/ros.h>
#include "parameters.h"
#include <eigen3/Eigen/Dense>

class LamProcess
{
public:
    using Ptr = std::shared_ptr<LamProcess>;

    LamProcess();

    /**
   * @brief 将激光雷达点云做网格化处理
   * @param[in] 管理网格地图的哈希表数据
   * @param[in] 当前特征点云
   * @param[in] 位姿旋转
   * @param[in] 位姿平移
   * @param[in] 特征类型，0：surface, 1:corner
   * @param[in] 在滑窗中所处位置
   * @param[in] 滑动窗口容量大小，大于windows_size
   * @return void 
   */
    void cutVoxel(unordered_map<VoxelStructure, OctoTree *> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat,
                  Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity);

    /**
   * @brief 激光scan去畸变,将所有点云转换到起始点坐标系下(针对单点) 
   * @param[in] 输入点云，其指向的指针和指向的内容均不能被修改
   * @param[out] 转换后输出点云
   */
    void cloudDeskew(PointType const *const point_in, PointType *const point_out);

    /**
   * @brief 激光scan去畸变,将所有点云转换到起始点坐标系下(针对整个scan) 
   * @param[in/out] 输入/输出点云
   */
    void cloudDeskew(pcl::PointCloud<PointType> &point_in);

    /**
   * @brief 将当前激光雷达scan转换到下一帧雷达的起始位置(针对单点)
   * @note 初步去畸变之后最后的path轨迹贴合更为紧凑，说明有效果，在去畸变策略上可思考如何更好
   * @param[in] 输入点云，其指向的指针和指向的内容均不能被修改
   * @param[out] 转换后输出点云
   */
    void cloudPositionTrans(PointType const *const point_in, PointType *const point_out);

    /**
   * @brief 计算协方差矩阵
   * @param[in] 特征数据
   * @param[in] 最近邻点个数
   * @param[in] 存放最近邻点id的容器
   * @return void 
   */
    void computeCovMat(const pcl::PointCloud<PointType> &point_cloud, int closest_point_cnt,
                       const vector<int> &point_search_id);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool deskew_flag_ = true;
    double scan_period_ = 0.1;

    // Eigen::Map<Eigen::Quaterniond>
    // 使用Eigen::Map类型可以省去在函数中的不断赋值操作，其定义时初始变量值更新使，
    // 所定义的变量也会对应更新,其作用可以理解为引用
    // 参考:http://www.javashuo.com/article/p-uqyrhqcs-nw.html
    Eigen::Quaterniond qua_incre_;
    Eigen::Vector3d trans_incre_;

    // 初始化赋值方式
    // Eigen::Matrix3d cov_mat_ = Eigen::Matrix3d::Zero();;
    // Eigen::Vector3d center_coor_ = Vector3d(0.0, 0.0, 0.0);
    Eigen::Matrix3d cov_mat_;
    Eigen::Vector3d center_coor_;

private:
    /**
   * @brief 内部参数初始化
   */
    void init();
};