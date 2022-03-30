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

class SlidingWindowOpti
{
public:
    SlidingWindowOpti(int ss, int fn, int thnum);
    // Used by "push_voxel"
    void downsample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame, vector<Eigen::Vector3d> &plvec_voxel,
                    vector<int> &slwd_num, int filternum2use);

    // Push voxel into optimizer
    void push_voxel(vector<vector<Eigen::Vector3d> *> &plvec_orig, SigmaVector &sig_vec, int lam_type);

    // Calculate Hessian, Jacobian, residual
    void acc_t_evaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head, int end, Eigen::MatrixXd &Hess,
                        Eigen::VectorXd &JacT, double &residual);

    // Multithread for "acc_t_evaluate"
    void divide_thread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess,
                       Eigen::VectorXd &JacT, double &residual);

    // Calculate residual
    void evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual);

    // LM process
    void damping_iter();

    int read_refine_state();

    void set_refine_state(int tem);

    void free_voxel();

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
    int map_refine_flag_;
    mutex my_mutex_;
};

class OctoTree
{
public:
    OctoTree(int ft, int capa);

    // 用在recut()函数中 计算特征值比
    // Used by "recut"
    void calc_eigen();

    // 将网格划分为更细的特征
    // frame_head: 滑动窗口中最新scan的位置
    void recut(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map);

    // 将滑动窗口中的五个scan边缘化掉 (assume margi_size is 5)
    void marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses,
                     int window_base, pcl::PointCloud<PointType> &pl_feat_map);

    // Used by "traversal_opt"
    void traversal_opt_calc_eigen();

    // Push voxel into "opt_lsv" (LM optimizer)
    // Push voxel into "opt_lsv" (LM optimizer)
    void traversal_opt(SlidingWindowOpti &opt_lsv);

public:
    static int voxel_windows_size_;
    vector<PointVector *> point_vec_orig_;
    vector<PointVector *> point_vec_tran_;
    // 0 ：树的末端，1：非末端
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
    bool is2opt_;
    int capacity_;
    pcl::PointCloud<PointType> root_centors_;
};

// Scam2map optimizer
class VoxelDistance
{
public:
    void push_surf(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff);

    void push_line(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff);

    void evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g, double &residual);

    void evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual);

    void damping_iter();

public:
    SO3 so3_pose_, so3_temp_;
    Eigen::Vector3d t_pose_, t_temp_;
    PointVector surf_centor_, surf_direct_;
    PointVector corn_centor_, corn_direct_;
    PointVector surf_gather_, corn_gather_;
    vector<double> surf_coeffs_, corn_coeffs_;
};
