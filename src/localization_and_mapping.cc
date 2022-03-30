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

    // Only one scan
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

    // Frame num in sliding window for each point in "plvec_voxel"
    vector<int> *slwd_num = new vector<int>();
    plvec_voxel->reserve(filternum2use * sw_size_);
    slwd_num->reserve(filternum2use * sw_size_);

    // retain one point for one scan (you can modify)
    for (int i = 0; i < sw_size_; i++)
    {
        if (!plvec_orig[i]->empty())
        {
            downsample(*plvec_orig[i], i, *plvec_voxel, *slwd_num, filternum2use);
        }
    }

    // Push a voxel into optimizer
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

    // In program, lambda_0 < lambda_1 < lambda_2
    // For plane, the residual is lambda_0
    // For line, the residual is lambda_0+lambda_1
    // We only calculate lambda_1 here
    for (int a = head; a < end; a++)
    {
        uint k = lam_types_[a]; // 0 is surf, 1 is line
        SigmaVector &sig_vec = sig_vecs_[a];
        vector<Eigen::Vector3d> &plvec_voxel = *plvec_voxels_[a];
        // Position in slidingwindow for each point in "plvec_voxel"
        vector<int> &slwd_num = *sw_nums_[a];
        uint backnum = plvec_voxel.size();

        Eigen::Vector3d vec_tran;
        vector<Eigen::Vector3d> plvec_back(backnum);
        // derivative point to T (R, t)
        vector<Eigen::Matrix3d> point_xis(backnum);
        Eigen::Vector3d centor(Eigen::Vector3d::Zero());
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i];
            // left multiplication instead of right muliplication in paper
            point_xis[i] = -SO3::hat(vec_tran);
            plvec_back[i] = vec_tran + t_ps[slwd_num[i]]; // after trans

            centor += plvec_back[i];
            covMat += plvec_back[i] * plvec_back[i].transpose();
        }

        double N_points = backnum + sig_vec.sigma_size_;
        centor += sig_vec.sigma_vi_;
        covMat += sig_vec.sigma_vTv_;

        covMat = covMat - centor * centor.transpose() / N_points;
        covMat = covMat / N_points;
        centor = centor / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        Eigen::Vector3d eigen_value = saes.eigenvalues();

        Eigen::Matrix3d U = saes.eigenvectors();
        Eigen::Vector3d u[3]; // eigenvectors
        for (int j = 0; j < 3; j++)
        {
            u[j] = U.block<3, 1>(0, j);
        }

        // Jacobian matrix
        Eigen::Matrix3d ukukT = u[k] * u[k].transpose();
        Eigen::Vector3d vec_Jt;
        for (uint i = 0; i < backnum; i++)
        {
            plvec_back[i] = plvec_back[i] - centor;
            vec_Jt = 2.0 / N_points * ukukT * plvec_back[i];
            _jact.block<3, 1>(6 * slwd_num[i] + 3, 0) += vec_Jt;
            _jact.block<3, 1>(6 * slwd_num[i], 0) -= point_xis[i] * vec_Jt;
        }

        // Hessian matrix
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
            // part of F matrix in paper
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
            // block matrix operation, half Hessian matrix
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
                Hessian33 = 2.0 / N_points * Hessian33; // Hessian matrix of lambda and point

                // Hessian matrix of lambda and pose
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
            // add weight for line feature
            residual += corn_less_ * eigen_value[k];
            Hess += corn_less_ * _hess;
            JacT += corn_less_ * _jact;
        }
        else
        {
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
        Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());

        for (uint i = 0; i < backnum; i++)
        {
            vec_tran = so3_ps[slwd_num[i]].matrix() * plvec_voxel[i] + t_ps[slwd_num[i]];
            centor += vec_tran;
            covMat += vec_tran * vec_tran.transpose();
        }

        double N_points = backnum + sig_vec.sigma_size_;
        centor += sig_vec.sigma_vi_;
        covMat += sig_vec.sigma_vTv_;

        covMat = covMat - centor * centor.transpose() / N_points;
        covMat = covMat / N_points;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
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
            // calculate Hessian, Jacobian, residual
            divide_thread(so3_poses_, t_poses_, Hess, JacT, residual1);
        }

        D = Hess.diagonal().asDiagonal();
        Hess2 = Hess + u * D;

        // The eigen solver is slower than opencv solver

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
            // left multiplication
            so3_poses_temp_[j] = SO3::exp(dxi.block<3, 1>(6 * (j), 0)) * so3_poses_[j];
            t_poses_temp_[j] = t_poses_[j] + dxi.block<3, 1>(6 * (j) + 3, 0);
        }

        // LM
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
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint asize;
    for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
    {
        asize = point_vec_tran_[i]->size();
        for (uint j = 0; j < asize; j++)
        {
            covMat += (*point_vec_tran_[i])[j] * (*point_vec_tran_[i])[j].transpose();
            center += (*point_vec_tran_[i])[j];
        }
    }

    covMat += sig_vec_.sigma_vTv_;
    center += sig_vec_.sigma_vi_;
    center /= points_size_;

    covMat = covMat / points_size_ - center * center.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
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

        calc_eigen(); // calculate eigenvalue ratio

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
    uint a_size;

    for (int i = frame_head; i < OctoTree::voxel_windows_size_; i++)
    {
        a_size = point_vec_tran_[i]->size();
        for (uint j = 0; j < a_size; j++)
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
    Eigen::Matrix3d covMat(Eigen::Matrix3d::Zero());
    Eigen::Vector3d center(0, 0, 0);

    uint asize;
    for (int i = 0; i < OctoTree::voxel_windows_size_; i++)
    {
        asize = point_vec_tran_[i]->size();
        for (uint j = 0; j < asize; j++)
        {
            covMat += (*point_vec_tran_[i])[j] * (*point_vec_tran_[i])[j].transpose();
            center += (*point_vec_tran_[i])[j];
        }
    }

    covMat -= center * center.transpose() / sw_points_size_;
    covMat /= sw_points_size_;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
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
        Eigen::Matrix3d _jac = surf_direct_[i] * surf_direct_[i].transpose();
        Eigen::Vector3d vec_tran = so3_p.matrix() * surf_gather_[i];
        Eigen::Matrix3d point_xi = -SO3::hat(vec_tran);
        vec_tran += t_p;

        Eigen::Vector3d v_ac = vec_tran - surf_centor_[i];
        Eigen::Vector3d d_vec = _jac * v_ac;
        Eigen::Matrix<double, 3, 6> jacob;
        jacob.block<3, 3>(0, 0) = _jac * point_xi;
        jacob.block<3, 3>(0, 3) = _jac;

        residual += surf_coeffs_[i] * d_vec.dot(d_vec);
        Hess += surf_coeffs_[i] * jacob.transpose() * jacob;
        g += surf_coeffs_[i] * jacob.transpose() * d_vec;
    }

    a_size = corn_gather_.size();
    for (uint i = 0; i < a_size; i++)
    {
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

    for (int i = 0; i < 20; i++)
    {
        evaluate_para(so3_pose_, t_pose_, Hess, g, residual1);
        D = Hess.diagonal().asDiagonal();

        // dxi = (Hess + u*D).bdcSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(-g);

        Hess2 = Hess + u * D;
        for (int j = 0; j < 6; j++)
        {
            matB.at<double>(j, 0) = -g(j, 0);
            for (int f = 0; f < 6; f++)
            {
                matA.at<double>(j, f) = Hess2(j, f);
            }
        }
        cv::solve(matA, matB, matX, cv::DECOMP_QR);
        for (int j = 0; j < 6; j++)
        {
            dxi(j, 0) = matX.at<double>(j, 0);
        }

        so3_temp_ = SO3::exp(dxi.block<3, 1>(0, 0)) * so3_pose_;
        t_temp_ = t_pose_ + dxi.block<3, 1>(3, 0);
        evaluate_only_residual(so3_temp_, t_temp_, residual2);
        double q1 = dxi.dot(u * D * dxi - g);
        double q = residual1 - residual2;
        // printf("residual: %lf u: %lf v: %lf q: %lf %lf %lf\n", residual1, u, v, q/q1, q1, q);
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