/**
 * @file localization_and_mapping.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */
#include "localization_and_mapping.h"

double feat_eigen_limit[2] = {3 * 3, 2 * 2};
double opt_feat_eigen_limit[2] = {4 * 4, 3 * 3};

SlidingWindowOpti::SlidingWindowOpti(int ss, int fn, int thnum) : sw_size_(ss), filter_num_(fn), thd_num_(thnum)
{
    so3_poses_.resize(ss);
    t_poses_.resize(ss);
    so3_poses_temp_.resize(ss);
    t_poses_temp_.resize(ss);
    jac_leng_ = 6 * ss;
    corn_less_ = 0.1;
    map_refine_flag_ = 0;
}

void SlidingWindowOpti::downsample(vector<Eigen::Vector3d> &plvec_orig, int cur_frame, vector<Eigen::Vector3d> &plvec_voxel,
                                   vector<int> &slwd_num, int filternum2use)
{
    uint plsize = plvec_orig.size();
    if (plsize <= (uint)filternum2use)
    {
        for (uint i = 0; i < plsize; i++)
        {
            plvec_voxel.push_back(plvec_orig[i]);
            slwd_num.push_back(cur_frame);
        }
        return;
    }

    Eigen::Vector3d center;
    double part = 1.0 * plsize / filternum2use;

    for (int i = 0; i < filternum2use; i++)
    {
        uint np = part * i;
        uint nn = part * (i + 1);
        center.setZero();
        for (uint j = np; j < nn; j++)
        {
            center += plvec_orig[j];
        }
        center = center / (nn - np);
        plvec_voxel.push_back(center);
        slwd_num.push_back(cur_frame);
    }
}

void SlidingWindowOpti::push_voxel(vector<vector<Eigen::Vector3d> *> &plvec_orig, SigmaVector &sig_vec, int lam_type)
{
    int process_points_size = 0;
    for (int i = 0; i < sw_size_; ++i)
    {
        if (!plvec_orig[i]->empty())
        {
            process_points_size++;
        }
    }

    // 若只有一帧点云数据
    if (process_points_size <= 1)
    {
        return;
    }

    int filternum2use = filter_num_;
    if (filter_num_ * process_points_size < MIN_PS)
    {
        filternum2use = MIN_PS / process_points_size + 1;
    }

    vector<Eigen::Vector3d> *plvec_voxel = new vector<Eigen::Vector3d>();

    // 记录plvec_voxel中每一个点所在滑动窗口中的帧数
    vector<int> *slwd_num = new vector<int>();
    plvec_voxel->reserve(filternum2use * sw_size_);
    slwd_num->reserve(filternum2use * sw_size_);

    // 一次扫描只保留一个点(可做修改)
    for (int i = 0; i < sw_size_; i++)
    {
        if (!plvec_orig[i]->empty())
        {
            downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
        }
    }

    // 将网格地图数据放入优化器
    plvec_voxels_.push_back(plvec_voxel);
    sw_nums_.push_back(slwd_num);
    lam_types_.push_back(lam_type);
    sig_vecs_.push_back(sig_vec);
}

void SlidingWindowOpti::acc_t_evaluate(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, int head,
                                       int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
{
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    Eigen::MatrixXd _hess(Hess);
    Eigen::MatrixXd _jact(JacT);

    // 在程序设计中, lambda_0 < lambda_1 < lambda_2，
    // 对于平面特征，残差为：lambda_0,对于线特征，残差为：lambda_0+lambda_1
    // 此处仅计算：lambda_1
    for (int a = head; a < end; a++)
    {
        // 0:平面特征, 1:线特征
        uint k = lam_types_[a]; 
        SigmaVector &sig_vec = sig_vecs_[a];
        vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels_[a];
        // 获取plvec_voxel中的点在滑动窗口中的位置
        vector<int> &slwd_num = *sw_nums_[a];
        uint backnum = plvec_voxel.size();

        Eigen::Vector3d vec_tran;
        vector<Eigen::Vector3d> plvec_back(backnum);
        // 对T (R, t)求导结果，即为论文中矩阵D的结果
        vector<Eigen::Matrix3d> point_xis(backnum);
        Eigen::Vector3d centor(Eigen::Vector3d::Zero());
        Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i];
            // 使用左乘代替右乘
            point_xis[i] = -SO3::hat(vec_tran);
            // 转换之后
            plvec_back[i] = vec_tran + t_ps[slwd_num[i]]; 
            centor += plvec_back[i];
            cov_mat += plvec_back[i] * plvec_back[i].transpose();
        }

        double N_points = backnum + sig_vec.sigma_size_;
        centor += sig_vec.sigma_vi_;
        cov_mat += sig_vec.sigma_vTv_;

        cov_mat = cov_mat - centor * centor.transpose() / N_points;
        cov_mat = cov_mat / N_points;
        centor = centor / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
        Eigen::Vector3d eigen_value = saes.eigenvalues();

        Eigen::Matrix3d U = saes.eigenvectors();
        // 存储特征向量
        Eigen::Vector3d u[3];
        for (int j = 0; j < 3; j++)
        {
            u[j] = U.block<3, 1>(0, j);
        }

        // 雅克比矩阵
        Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
        Eigen::Vector3d vec_Jt;
        for (uint i = 0; i < backnum; i++)
        {
            plvec_back[i] = plvec_back[i] - centor;
            // 因这里计算是考虑到所有点的情况，故与前述有所不同
            vec_Jt = 2.0 / N_points * ukukT * plvec_back[i];
            _jact.block<3, 1>(6 * slwd_num[i] + 3, 0) += vec_Jt;
            _jact.block<3, 1>(6 * slwd_num[i], 0) -= point_xis[i] * vec_Jt;
        }

        // 海森矩阵
        Eigen::Matrix3d Hessian33;
        Eigen::Matrix3d C_k;
        vector<Eigen::Matrix3d> C_k_np(3);
        for (uint i = 0; i < 3; i++)
        {
            if (i == k)
            {
                C_k_np[i].setZero();
                continue;
            }
            Hessian33 = u[i] * u[k].transpose();
            // 论文中F矩阵的某一部分
            C_k_np[i] = -1.0 / N_points / (eigen_value[i] - eigen_value[k]) * (Hessian33 + Hessian33.transpose());
        }

        Eigen::Matrix3d h33;
        uint rownum, colnum;
        for (uint j = 0; j < backnum; j++)
        {
            for (int f = 0; f < 3; f++)
            {
                C_k.block<1, 3>(f, 0) = plvec_back[j].transpose() * C_k_np[f];
            }
            C_k = U * C_k;
            colnum = 6 * slwd_num[j];
            // 矩阵块运算, 对照论文中公式(7)
            for (uint i = j; i < backnum; i++)
            {
                Hessian33 = u[k] * (plvec_back[i]).transpose() * C_k + u[k].dot(plvec_back[i]) * C_k;

                rownum = 6 * slwd_num[i];
                if (i == j)
                {
                    Hessian33 += (N_points - 1) / N_points * ukukT;
                }
                else
                {
                    Hessian33 -= 1.0 / N_points * ukukT;
                }
                // 关于lambda及地图点的海森矩阵
                Hessian33 = 2.0 / N_points * Hessian33; 

                // 关于lambda及位姿的海森矩阵
                if (rownum == colnum && i != j)
                {
                    _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33 + Hessian33.transpose();

                    h33 = -point_xis[i] * Hessian33;
                    _hess.block<3, 3>(rownum, colnum + 3) += h33;
                    _hess.block<3, 3>(rownum + 3, colnum) += h33.transpose();
                    h33 = Hessian33 * point_xis[j];
                    _hess.block<3, 3>(rownum + 3, colnum) += h33;
                    _hess.block<3, 3>(rownum, colnum + 3) += h33.transpose();
                    h33 = -point_xis[i] * h33;
                    _hess.block<3, 3>(rownum, colnum) += h33 + h33.transpose();
                }
                else
                {
                    _hess.block<3, 3>(rownum + 3, colnum + 3) += Hessian33;
                    h33 = Hessian33 * point_xis[j];
                    _hess.block<3, 3>(rownum + 3, colnum) += h33;
                    _hess.block<3, 3>(rownum, colnum + 3) -= point_xis[i] * Hessian33;
                    _hess.block<3, 3>(rownum, colnum) -= point_xis[i] * h33;
                }
            }
        }

        if (k == 1)
        {
            // 对线特征添加权重
            residual += corn_less_ * eigen_value[k];
            Hess += corn_less_ * _hess;
            JacT += corn_less_ * _jact;
        }
        else
        {
            // 对平面特征添加权重
            residual += eigen_value[k];
            Hess += _hess;
            JacT += _jact;
        }
        _hess.setZero();
        _jact.setZero();
    }

    // Hessian is symmetric, copy to save time
    for (int j = 0; j < jac_leng_; j += 6)
    {
        for (int i = j + 6; i < jac_leng_; i += 6)
        {
            Hess.block<6, 6>(j, i) = Hess.block<6, 6>(i, j).transpose();
        }
    }
}

void SlidingWindowOpti::divide_thread(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, Eigen::MatrixXd &Hess,
                                      Eigen::VectorXd &JacT, double &residual)
{
    Hess.setZero();
    JacT.setZero();
    residual = 0;

    vector<Eigen::MatrixXd> hessians(thd_num_, Hess);
    vector<Eigen::VectorXd> jacobians(thd_num_, JacT);
    vector<double> resis(thd_num_, 0);

    uint gps_size = plvec_voxels_.size();
    // 如果gps_size较小，只需要一个线程处理即可
    if (gps_size < (uint)thd_num_)
    {
        acc_t_evaluate(so3_ps, t_ps, 0, gps_size, Hess, JacT, residual);
        Hess = hessians[0];
        JacT = jacobians[0];
        residual = resis[0];
        return;
    }

    vector<thread *> mthreads(thd_num_);

    double part = 1.0 * (gps_size) / thd_num_;
    for (int i = 0; i < thd_num_; i++)
    {
        int np = part * i;
        int nn = part * (i + 1);
        // 默认传参是拷贝形式，这里需要传引用形式，使用std::ref()
        // 比如thread的方法传递引用的时候，必须外层用ref来进行引用传递，否则就是浅拷贝
        mthreads[i] = new thread(&SlidingWindowOpti::acc_t_evaluate, this, ref(so3_ps), ref(t_ps), np, nn, ref(hessians[i]), ref(jacobians[i]), ref(resis[i]));
    }

    for (int i = 0; i < thd_num_; i++)
    {
        mthreads[i]->join();
        Hess += hessians[i];
        JacT += jacobians[i];
        residual += resis[i];
        delete mthreads[i];
    }
}

void SlidingWindowOpti::evaluate_only_residual(vector<SO3> &so3_ps, vector<Eigen::Vector3d> &t_ps, double &residual)
{
    residual = 0;
    uint gps_size = plvec_voxels_.size();
    Eigen::Vector3d vec_tran;

    for (uint a = 0; a < gps_size; a++)
    {
        uint k = lam_types_[a];
        SigmaVector &sig_vec = sig_vecs_[a];
        vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels_[a];
        vector<int> &slwd_num = *sw_nums_[a];
        uint backnum = plvec_voxel.size();

        Eigen::Vector3d centor(Eigen::Vector3d::Zero());
        Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i] + t_ps[slwd_num[i]];
            centor += vec_tran;
            cov_mat += vec_tran * vec_tran.transpose();
        }

        double N_points = backnum + sig_vec.sigma_size_;
        centor += sig_vec.sigma_vi_;
        cov_mat += sig_vec.sigma_vTv_;

        cov_mat = cov_mat - centor * centor.transpose() / N_points;
        cov_mat = cov_mat / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
        Eigen::Vector3d eigen_value = saes.eigenvalues();

        if (k == 1)
        {
            residual += corn_less_ * eigen_value[k];
        }
        else
        {
            residual += eigen_value[k];
        }
    }
}

// 所采用的优化思想及步骤与scan2map模块相同
void SlidingWindowOpti::damping_iter()
{
    my_mutex_.lock();
    map_refine_flag_ = 1;
    my_mutex_.unlock();

    if (plvec_voxels_.size() != sw_nums_.size() || plvec_voxels_.size() != lam_types_.size() || plvec_voxels_.size() != sig_vecs_.size())
    {
        printf("size is not equal\n");
        exit(0);
    }

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng_, jac_leng_), Hess(jac_leng_, jac_leng_);
    Eigen::VectorXd JacT(jac_leng_), dxi(jac_leng_);

    Eigen::MatrixXd Hess2(jac_leng_, jac_leng_);
    Eigen::VectorXd JacT2(jac_leng_);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;

    cv::Mat matA(jac_leng_, jac_leng_, CV_64F, cv::Scalar::all(0));
    cv::Mat matB(jac_leng_, 1, CV_64F, cv::Scalar::all(0));
    cv::Mat matX(jac_leng_, 1, CV_64F, cv::Scalar::all(0));

    for (int i = 0; i < iter_max_; i++)
    {
        if (is_calc_hess)
        {
            // 计算海森矩阵、雅克比矩阵及残差
            divide_thread(so3_poses_, t_poses_, Hess, JacT, residual1);
        }

        D = Hess.diagonal().asDiagonal();
        Hess2 = Hess + u * D;

        // eigen中求解器要比opencv中求解方式慢一些
        // dxi = (Hess2).bdcSvd(Eigen::ComputeFullU|Eigen::ComputeFullV).solve(-JacT);
        for (int j = 0; j < jac_leng_; j++)
        {
            matB.at<double>(j, 0) = -JacT(j, 0);
            for (int f = 0; f < jac_leng_; f++)
            {
                matA.at<double>(j, f) = Hess2(j, f);
            }
        }
        cv::solve(matA, matB, matX, cv::DECOMP_QR);
        for (int j = 0; j < jac_leng_; j++)
        {
            dxi(j, 0) = matX.at<double>(j, 0);
        }

        for (int j = 0; j < sw_size_; j++)
        {
            // 左乘形式
            so3_poses_temp_[j] = SO3::exp(dxi.block<3, 1>(6 * (j), 0)) * so3_poses_[j];
            t_poses_temp_[j] = t_poses_[j] + dxi.block<3, 1>(6 * (j) + 3, 0);
        }

        double q1 = 0.5 * (dxi.transpose() * (u * D * dxi - JacT))[0];
        // double q1 = 0.5*dxi.dot(u*D*dxi-JacT);
        evaluate_only_residual(so3_poses_temp_, t_poses_temp_, residual2);

        q = (residual1 - residual2);
        // printf("residual%d: %lf u: %lf v: %lf q: %lf %lf %lf\n", i, residual1, u, v, q/q1, q1, q);

        if (q > 0)
        {
            so3_poses_ = so3_poses_temp_;
            t_poses_ = t_poses_temp_;
            q = q / q1;
            v = 2;
            q = 1 - pow(2 * q - 1, 3);
            u *= (q < one_three ? one_three : q);
            is_calc_hess = true;
        }
        else
        {
            u = u * v;
            v = 2 * v;
            is_calc_hess = false;
        }

        if (fabs(residual1 - residual2) < 1e-9)
        {
            break;
        }
    }

    my_mutex_.lock();
    map_refine_flag_ = 2;
    my_mutex_.unlock();
}

int SlidingWindowOpti::read_refine_state()
{
    int tem_flag;
    my_mutex_.lock();
    tem_flag = map_refine_flag_;
    my_mutex_.unlock();
    return tem_flag;
}

void SlidingWindowOpti::set_refine_state(int tem)
{
    my_mutex_.lock();
    map_refine_flag_ = tem;
    my_mutex_.unlock();
}

void SlidingWindowOpti::free_voxel()
{
    uint a_size = plvec_voxels_.size();
    for (uint i = 0; i < a_size; i++)
    {
        delete (plvec_voxels_[i]);
        delete (sw_nums_[i]);
    }

    plvec_voxels_.clear();
    sw_nums_.clear();
    sig_vecs_.clear();
    lam_types_.clear();
}

OctoTree::OctoTree(int ft, int capa) : ftype_(ft), capacity_(capa)
{
    octo_state_ = 0;
    for (int i = 0; i < 8; ++i)
    {
        leaves_[i] = nullptr;
    }
    for (int i = 0; i < capacity_; ++i)
    {
        point_vec_orig_.push_back(new PointVector());
        point_vec_tran_.push_back(new PointVector());
    }
    is2opt_ = true;
}

void OctoTree::calc_eigen()
{
    Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint vec_size;
    for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
    {
        vec_size = point_vec_tran_[i]->size();
        for (uint j = 0; j < vec_size; j++)
        {
            cov_mat += (*point_vec_tran_[i])[j] * (*point_vec_tran_[i])[j].transpose();
            center += (*point_vec_tran_[i])[j];
        }
    }

    cov_mat += sig_vec_.sigma_vTv_;
    center += sig_vec_.sigma_vi_;
    center /= points_size_;

    cov_mat = cov_mat / points_size_ - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
    feat_eigen_ratio_ = saes.eigenvalues()[2] / saes.eigenvalues()[ftype_];
    Eigen::Vector3d direct_vec = saes.eigenvectors().col(2 * ftype_);

    ap_centor_direct_.x = center.x();
    ap_centor_direct_.y = center.y();
    ap_centor_direct_.z = center.z();
    ap_centor_direct_.normal_x = direct_vec.x();
    ap_centor_direct_.normal_y = direct_vec.y();
    ap_centor_direct_.normal_z = direct_vec.z();
}

void OctoTree::recut(int layer, uint frame_head, pcl::PointCloud<PointType> &pl_feat_map)
{
    if (octo_state_ == 0)
    {
        points_size_ = 0;
        for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
        {
            points_size_ += point_vec_orig_[i]->size();
        }

        points_size_ += sig_vec_.sigma_size_;
        if (points_size_ < MIN_PS)
        {
            feat_eigen_ratio_ = -1;
            return;
        }

        // 计算特征值比率
        calc_eigen();

        if (isnan(feat_eigen_ratio_))
        {
            feat_eigen_ratio_ = -1;
            return;
        }

        if (feat_eigen_ratio_ >= feat_eigen_limit[ftype_])
        {
            pl_feat_map.push_back(ap_centor_direct_);
            return;
        }

        // if(layer == 3)
        if (layer == 4)
        {
            return;
        }

        octo_state_ = 1;
        // All points in slidingwindow should be put into subvoxel
        frame_head = 0;
    }

    int leafnum;
    uint vec_size;

    for (int i = frame_head; i < OctoTree::voxel_windows_size_; i++)
    {
        vec_size = point_vec_tran_[i]->size();
        for (uint j = 0; j < vec_size; j++)
        {
            int xyz[3] = {0, 0, 0};
            for (uint k = 0; k < 3; k++)
            {
                if ((*point_vec_tran_[i])[j][k] > voxel_center_[k])
                {
                    xyz[k] = 1;
                }
            }
            leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
            if (leaves_[leafnum] == nullptr)
            {
                leaves_[leafnum] = new OctoTree(ftype_, capacity_);
                leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
                leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            }
            leaves_[leafnum]->point_vec_orig_[i]->push_back((*point_vec_orig_[i])[j]);
            leaves_[leafnum]->point_vec_tran_[i]->push_back((*point_vec_tran_[i])[j]);
        }
    }

    if (layer != 0)
    {
        for (int i = frame_head; i < OctoTree::voxel_windows_size_; i++)
        {
            if (point_vec_orig_[i]->size() != 0)
            {
                vector<Eigen::Vector3d>().swap(*point_vec_orig_[i]);
                vector<Eigen::Vector3d>().swap(*point_vec_tran_[i]);
            }
        }
    }

    layer++;
    for (uint i = 0; i < 8; i++)
    {
        if (leaves_[i] != nullptr)
        {
            leaves_[i]->recut(layer, frame_head, pl_feat_map);
        }
    }
}

void OctoTree::marginalize(int layer, int margi_size, vector<Eigen::Quaterniond> &q_poses, vector<Eigen::Vector3d> &t_poses,
                           int window_base, pcl::PointCloud<PointType> &pl_feat_map)
{
    if (octo_state_ != 1 || layer == 0)
    {
        if (octo_state_ != 1)
        {
            for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
            {
                // Update points by new poses
                plvec_trans_func(*point_vec_orig_[i], *point_vec_tran_[i], q_poses[i + window_base].matrix(), t_poses[i + window_base]);
            }
        }

        // Push front 5 scans into P_fix
        uint a_size;
        if (feat_eigen_ratio_ > feat_eigen_limit[ftype_])
        {
            for (int i = 0; i < margi_size; i++)
            {
                sig_vec_points_.insert(sig_vec_points_.end(), point_vec_tran_[i]->begin(), point_vec_tran_[i]->end());
            }
            down_sampling_voxel(sig_vec_points_, quater_length_);

            a_size = sig_vec_points_.size();
            sig_vec_.toZero();
            sig_vec_.sigma_size_ = a_size;
            for (uint i = 0; i < a_size; i++)
            {
                sig_vec_.sigma_vTv_ += sig_vec_points_[i] * sig_vec_points_[i].transpose();
                sig_vec_.sigma_vi_ += sig_vec_points_[i];
            }
        }

        // Clear front 5 scans
        for (int i = 0; i < margi_size; i++)
        {
            PointVector().swap(*point_vec_orig_[i]);
            PointVector().swap(*point_vec_tran_[i]);
            // plvec_orig[i].clear(); plvec_orig[i].shrink_to_fit();
        }

        if (layer == 0)
        {
            a_size = 0;
            for (int i = margi_size; i < OctoTree::voxel_windows_size_; i++)
            {
                a_size += point_vec_orig_[i]->size();
            }
            if (a_size == 0)
            {
                // Voxel has no points in slidingwindow
                is2opt_ = false;
            }
        }

        for (int i = margi_size; i < OctoTree::voxel_windows_size_; i++)
        {
            point_vec_orig_[i]->swap(*point_vec_orig_[i - margi_size]);
            point_vec_tran_[i]->swap(*point_vec_tran_[i - margi_size]);
        }

        if (octo_state_ != 1)
        {
            points_size_ = 0;
            for (int i = 0; i < OctoTree::voxel_windows_size_ - margi_size; i++)
            {
                points_size_ += point_vec_orig_[i]->size();
            }
            points_size_ += sig_vec_.sigma_size_;
            if (points_size_ < MIN_PS)
            {
                feat_eigen_ratio_ = -1;
                return;
            }

            calc_eigen();

            if (isnan(feat_eigen_ratio_))
            {
                feat_eigen_ratio_ = -1;
                return;
            }
            if (feat_eigen_ratio_ >= feat_eigen_limit[ftype_])
            {
                pl_feat_map.push_back(ap_centor_direct_);
            }
        }
    }

    if (octo_state_ == 1)
    {
        layer++;
        for (int i = 0; i < 8; i++)
        {
            if (leaves_[i] != nullptr)
            {
                leaves_[i]->marginalize(layer, margi_size, q_poses, t_poses, window_base, pl_feat_map);
            }
        }
    }
}

void OctoTree::traversal_opt_calc_eigen()
{
    Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint vec_size;
    for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
    {
        vec_size = point_vec_tran_[i]->size();
        for (uint j = 0; j < vec_size; j++)
        {
            cov_mat += (*point_vec_tran_[i])[j] * (*point_vec_tran_[i])[j].transpose();
            center += (*point_vec_tran_[i])[j];
        }
    }

    cov_mat -= center * center.transpose() / sw_points_size_;
    cov_mat /= sw_points_size_;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
    feat_eigen_ratio_test_ = (saes.eigenvalues()[2] / saes.eigenvalues()[ftype_]);
}

void OctoTree::traversal_opt(SlidingWindowOpti &opt_lsv)
{
    if (octo_state_ != 1)
    {
        sw_points_size_ = 0;
        for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
        {
            sw_points_size_ += point_vec_orig_[i]->size();
        }
        if (sw_points_size_ < MIN_PS)
        {
            return;
        }
        traversal_opt_calc_eigen();

        if (isnan(feat_eigen_ratio_test_))
        {
            return;
        }

        if (feat_eigen_ratio_test_ > opt_feat_eigen_limit[ftype_])
        {
            opt_lsv.push_voxel(point_vec_orig_, sig_vec_, ftype_);
        }
    }
    else
    {
        for (int i = 0; i < 8; i++)
        {
            if (leaves_[i] != nullptr)
            {
                leaves_[i]->traversal_opt(opt_lsv);
            }
        }
    }
}

void VoxelDistance::push_surf(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
{
    direct.normalize();
    surf_direct_.push_back(direct);
    surf_centor_.push_back(centor);
    surf_gather_.push_back(orip);
    surf_coeffs_.push_back(coeff);
}

void VoxelDistance::push_line(Eigen::Vector3d &orip, Eigen::Vector3d &centor, Eigen::Vector3d &direct, double coeff)
{
    direct.normalize();
    corn_direct_.push_back(direct);
    corn_centor_.push_back(centor);
    corn_gather_.push_back(orip);
    corn_coeffs_.push_back(coeff);
}

void VoxelDistance::evaluate_para(SO3 &so3_p, Eigen::Vector3d &t_p, Eigen::Matrix<double, 6, 6> &Hess, Eigen::Matrix<double, 6, 1> &g, double &residual)
{
    Hess.setZero();
    g.setZero();
    residual = 0;
    uint a_size = surf_gather_.size();
    for (uint i = 0; i < a_size; i++)
    {
        // 由平面的方向向量获得雅克比矩阵，对照论文中公式(10)～(12)，这里_jac的计算形式有所简化
        Eigen::Matrix3d _jac = surf_direct_[i] * surf_direct_[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather_[i];
        Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
        vec_tran += t_p;
        // 经位姿转换后的点与特征中心的偏差向量
        Eigen::Vector3d v_ac = vec_tran - surf_centor_[i];
        // 定义三维的误差向量，雅克比乘以误差项，得到残差向量
        Eigen::Vector3d d_vec = _jac * v_ac;
        Eigen::Matrix<double, 3, 6> jacob;
        // 对照论文中公式，雅克比矩阵块所对应的计算方法
        jacob.block<3, 3>(0, 0) = _jac * point_xi;
        jacob.block<3, 3>(0, 3) = _jac;

        // 残差项的计算方法，每个残差项均有其对应的系数
        residual += surf_coeffs_[i] * d_vec.dot(d_vec);
        // 海森矩阵与雅克比矩阵近似关系，H=J^T * J
        Hess += surf_coeffs_[i] * jacob.transpose() * jacob;
        // H * delte_x = g, 这里计算的是增量g
        g += surf_coeffs_[i] * jacob.transpose() * d_vec;
    }

    a_size = corn_gather_.size();
    for (uint i = 0; i < a_size; i++)
    {
        // 对于corn特征，其_jac的计算方式不同，其余部分相同
        Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct_[i] * corn_direct_[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather_[i];
        Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - corn_centor_[i];
        Eigen::Vector3d d_vec = _jac * v_ac;
        Eigen::Matrix<double, 3, 6> jacob;
        jacob.block<3, 3>(0, 0) = _jac * point_xi;
        jacob.block<3, 3>(0, 3) = _jac;

        residual += corn_coeffs_[i] * d_vec.dot(d_vec);
        Hess += corn_coeffs_[i] * jacob.transpose() * jacob;
        g += corn_coeffs_[i] * jacob.transpose() * d_vec;
    }
}

void VoxelDistance::evaluate_only_residual(SO3 &so3_p, Eigen::Vector3d &t_p, double &residual)
{
    residual = 0;
    uint a_size = surf_gather_.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = surf_direct_[i] * surf_direct_[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather_[i];
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - surf_centor_[i];
        Eigen::Vector3d d_vec = _jac * v_ac;

        residual += surf_coeffs_[i] * d_vec.dot(d_vec);
    }

    a_size = corn_gather_.size();
    for (uint i = 0; i < a_size; i++)
    {
        Eigen::Matrix3d _jac = Eigen::Matrix3d::Identity() - corn_direct_[i] * corn_direct_[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * corn_gather_[i];
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - corn_centor_[i];
        Eigen::Vector3d d_vec = _jac * v_ac;

        residual += corn_coeffs_[i] * d_vec.dot(d_vec);
    }
}

void VoxelDistance::damping_iter()
{
    double u = 0.01, v = 2;
    Eigen::Matrix<double, 6, 6> D;
    D.setIdentity();
    Eigen::Matrix<double, 6, 6> Hess, Hess2;
    Eigen::Matrix<double, 6, 1> g;
    Eigen::Matrix<double, 6, 1> dxi;
    double residual1, residual2;

    cv::Mat matA(6, 6, CV_64F, cv::Scalar::all(0));
    cv::Mat matB(6, 1, CV_64F, cv::Scalar::all(0));
    cv::Mat matX(6, 1, CV_64F, cv::Scalar::all(0));

    // 最多执行20次迭代求解，当残差项满足要求是退出循环
    for (int i = 0; i < 20; i++)
    {
        evaluate_para(so3_pose_, t_pose_, Hess, g, residual1);
        // D为系数矩阵，为非负对角阵
        D = Hess.diagonal().asDiagonal();
        // LM求解的矩阵方程形式
        // dxi = (Hess + u*D).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-g);
        // 构造矩阵方程的求解形式
        Hess2 = Hess + u * D;
        for (int j = 0; j < 6; j++)
        {
            matB.at<double>(j, 0) = -g(j, 0);
            for (int f = 0; f < 6; f++)
            {
                matA.at<double>(j, f) = Hess2(j, f);
            }
        }
        // OpenCV中求解非线性最小二乘问题的方法，采用QR分解的方式
        // 使用opencv中的求解方式，比使用g2o或ceres要简便一些，但是总体思路是一样了的，
        // 这里需要手动构建残差项，设计优化步长进行逐次求解迭代；
        cv::solve(matA, matB, matX, cv::DECOMP_QR);
        for (int j = 0; j < 6; j++)
        {
            dxi(j, 0) = matX.at<double>(j, 0);
        }

        so3_temp_ = SO3::exp(dxi.block<3, 1>(0, 0)) * so3_pose_;
        t_temp_ = t_pose_ + dxi.block<3, 1>(3, 0);
        // 对新求出的位姿评估残差情况
        evaluate_only_residual(so3_temp_, t_temp_, residual2);
        // 使用q/q1来刻画近似程度的好坏，LM算法中详细给出
        // 求解出来的x值，点乘(步长*系数矩阵*x时刻值，减去g),其含义即是求解出相对增量
        double q1 = dxi.dot(u * D * dxi - g);
        // residual1 residual2表征两时刻的位姿数据
        double q = residual1 - residual2;
        // printf("residual: %lf u: %lf v: %lf q: %lf %lf %lf\n", residual1, u, v, q/q1, q1, q);
        // 不断更新优化半径
        // q大于0,说明求解有效，减小迭代步长，细化计算结果
        if (q > 0)
        {
            so3_pose_ = so3_temp_;
            t_pose_ = t_temp_;
            q = q / q1;
            v = 2;
            q = 1 - pow(2 * q - 1, 3);
            u *= (q < one_three ? one_three : q);
        }
        else
        {
            // 如果残差之差为负数，增大步长
            u = u * v;
            v = 2 * v;
        }

        if (fabs(residual1 - residual2) < 1e-9)
        {
            break;
        }
    }
}

int OctoTree::voxel_windows_size_ = 0;