/**
 * @file localization_and_mapping.h
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
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <queue>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseArray.h>
#include "third_lib.h"
#include "utils.h"

// const
const double one_three = (1.0 / 3.0);

// 用在地图优化阶段
class SlidingWindowOpti
{
public:
    /**
   * @brief SlidingWindowOpti的默认构造函数，用以初始化类中参数
   * @param[in] 滑动窗口数量
   * @param[in] 滤波器数量
   * @param[in] 线程数量
   */
    SlidingWindowOpti(int ss, int fn, int thnum);

    /**
   * @brief 对当前滑窗中某帧原始点云数据做下采样处理
   * @param[in] 原始三维点
   * @param[in] 帧id
   * @param[out] 滤波处理后的网格点
   * @param[out] 存储当前帧id
   * @param[in] 滤波数量
   * @return void 
   */
    void downSample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame, vector<Eigen::Vector3d> &plvec_voxel,
                    vector<int> &slwd_num, int filternum2use);

    // Push voxel into optimizer
    void pushVoxel(vector<vector<Eigen::Vector3d> *> &plvec_orig, SigmaVector &sig_vec, int lam_type);

    /**
   * @brief 以更为精确的方式计算海森矩阵、雅克比矩阵及残差
   * @param[in] 旋转
   * @param[in] 平移
   * @param[in] 起始位置索引
   * @param[in] 结束位置索引
   * @param[out] 海森矩阵
   * @param[out] 雅克比矩阵
   * @param[out] 残差值
   * @return void 
   */
    void mapRefineEvaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head, int end, Eigen::MatrixXd &Hess,
                        Eigen::VectorXd &JacT, double &residual);

    /**
   * @brief 采用多线程方式计算海森矩阵、雅克比矩阵及残差
   * @param[in] 旋转
   * @param[in] 平移
   * @param[out] 海森矩阵
   * @param[out] 雅克比矩阵
   * @param[out] 残差值
   * @return void 
   */
    void divideThread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess,
                       Eigen::VectorXd &JacT, double &residual);

    /**
   * @brief 计算残差
   * @param[in] 旋转
   * @param[in] 平移
   * @param[out] 残差值
   * @return void 
   */
    void evaluateOnlyResidual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual);

    /**
   * @brief 采用滑窗非线性LM优化方式优化位姿，以构建较高精度地图
   * @return void 
   */
    void dampingIter();

    /**
   * @brief 读取地图优化状态
   * @return int 
   */
    int readRefineState();

    /**
   * @brief 设置地图优化状态
   * @return void 
   */
    void setRefineState(int tem);

    /**
   * @brief 销毁体素网格指针及数据
   * @return void 
   */
    void releaseVoxel();

public:
    int sw_size_;
    int filter_num_;
    int thd_num_;
    int jac_leng_;

    int iter_max_ = 20;
    double corn_less_;

    vector<SO3> so3_poses_, so3_poses_temp_;
    vector<Eigen::Vector3d> t_poses_, t_poses_temp_;

    // 0 surf, 1 line
    vector<int> lam_types_;
    vector<SigmaVector> sig_vecs_;
    vector<vector<Eigen::Vector3d> *> plvec_voxels_;
    vector<vector<int> *> sw_nums_;
    // 2：表示优化已完成； 0：表示可以执行优化；1：正在执行优化
    int map_refine_flag_;
    mutex my_mutex_;
};

class OctoTree
{
public:
    OctoTree(int ft, int capa);

    /**
   * @brief 用在recut()函数中 计算特征值比
   * @return void 
   */
    void calculateEigen();

    /**
   * @brief 将网格划分为更细的特征
   * @param[in] 八叉树划分的层数
   * @param[in] frame_head: 滑动窗口中最新scan的位置
   * @param[out] 根中心向量参数
   * @return void 
   */
    void recutVoxel(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map);

    // 将滑动窗口中的五个scan边缘化掉 (assume margi_size is 5)
    void scanMarginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses,
                     int window_base, pcl::PointCloud<PointType> &pl_feat_map);

    /**
   * @brief 计算特征值、特征向量及特征比率，针对于每个滑窗中的点来计算
   * @return void 
   */
    void traversalOptCalcEigen();

    /**
   * @brief 将网格地图放入优化器
   * @param[in] LM非线性优化对象句柄
   * @return void 
   */
    void traversalOpt(SlidingWindowOpti &opt_lsv);

public:
    static int voxel_windows_size_;
    vector<PointVector *> point_vec_orig_;
    vector<PointVector *> point_vec_tran_;
    // 0：树的终点位置 1：非终点位置
    int octo_state_;
    PointVector sig_vec_points_;
    SigmaVector sig_vec_;
    int ftype_;
    int points_size_, sw_points_size_;
    double feat_eigen_ratio_, feat_eigen_ratio_test_;
    PointType ap_centor_direct_;
    double voxel_center_[3];
    double quater_length_;
    OctoTree *leaves_[8];
    // 表示是否经过位姿优化
    bool is2opt_;
    int capacity_;
    pcl::PointCloud<PointType> root_centors_;
};

// 用在里程计求解阶段，Scam2map optimizer
class VoxelDistance
{
public:
    /**
   * @brief 将平面特征放入优化器
   * @param[in] 原始点
   * @param[in] 特征中心向量
   * @param[in] 平面特征法向向量
   * @param[in] 特征权重系数
   * @return void 
   */
    void pushSurf(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff);

    /**
   * @brief 将角点特征放入优化器
   * @param[in] 原始点
   * @param[in] 特征中心向量
   * @param[in] 线特征方向向量
   * @param[in] 特征权重系数
   * @return void 
   */
    void pushLine(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff);

    /**
   * @brief 计算目标函数的优化更新参数
   * @param[in] 位姿旋转
   * @param[in] 位姿平移
   * @param[out] 海森矩阵
   * @param[out] 优化增量
   * @param[out] 残差值
   * @return void 
   */
    void evaluateParameters(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g, double &residual);

    /**
   * @brief 计算目标函数的优化更新参数
   * @param[in] 位姿旋转
   * @param[in] 位姿平移
   * @param[out] 残差值
   * @return void 
   */
    void evaluateOnlyResidual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual);

    /**
   * @brief 阻尼优化方式进行位姿优化，采用LM优化算法
   * @return void 
   */
    void dampingIter();

public:
    SO3 so3_pose_, so3_temp_;
    Eigen::Vector3d t_pose_, t_temp_;
    PointVector surf_centor_, surf_direct_;
    PointVector corn_centor_, corn_direct_;
    PointVector surf_gather_, corn_gather_;
    vector<double> surf_coeffs_, corn_coeffs_;
};
