/**
 * @file lam_ros.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 定位与建图模块主函数，topic发送接受，回调函数
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */
#include "localization_and_mapping.h"
#include <ros/ros.h>
#include "paramaters.h"

std::mutex mutex_data_buf;

queue<sensor_msgs::PointCloud2ConstPtr> corn_buf;
queue<sensor_msgs::PointCloud2ConstPtr> surf_buf;
queue<sensor_msgs::PointCloud2ConstPtr> full_buf;

/**
   * @brief 点、平面特征及完整点云的回调函数
   * @param[in] 特征信息及点云数据
   * @return void 
   */
void cornerFeatureHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    corn_buf.push(msg);
    mutex_data_buf.unlock();
}

void surfaceFeatureHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    surf_buf.push(msg);
    mutex_data_buf.unlock();
}

void combineCloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    full_buf.push(msg);
    mutex_data_buf.unlock();
}

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
void cut_voxel(unordered_map<VoxelStrcuture, OctoTree *> &feat_map, pcl::PointCloud<PointType>::Ptr pl_feat,
               Eigen::Matrix3d R_p, Eigen::Vector3d t_p, int feattype, int fnum, int capacity)
{
    uint plsize = pl_feat->size();
    for (uint i = 0; i < plsize; i++)
    {
        // 将特征点云转换至世界坐标系
        PointType &p_c = pl_feat->points[i];
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = R_p * pvec_orig + t_p;

        // 确定哈希表键值
        float loc_xyz[3];
        for (int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pvec_tran[j] / voxel_size[feattype];
            if (loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }
        VoxelStrcuture position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

        // 找到对应的网格位置
        auto iter = feat_map.find(position);
        if (iter != feat_map.end())
        {
            iter->second->point_vec_orig_[fnum]->push_back(pvec_orig);
            iter->second->point_vec_tran_[fnum]->push_back(pvec_tran);
            iter->second->is2opt_ = true;
        }
        else
        {
            // 新建网格数据
            OctoTree *ot = new OctoTree(feattype, capacity);
            ot->point_vec_orig_[fnum]->push_back(pvec_orig);
            ot->point_vec_tran_[fnum]->push_back(pvec_tran);

            // 网格中心坐标
            ot->voxel_center_[0] = (0.5 + position.x_) * voxel_size[feattype];
            ot->voxel_center_[1] = (0.5 + position.y_) * voxel_size[feattype];
            ot->voxel_center_[2] = (0.5 + position.z_) * voxel_size[feattype];
            // 取边长的四分之一值
            ot->quater_length_ = voxel_size[feattype] / 4.0;
            feat_map[position] = ot;
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_lam");
    ros::NodeHandle nh;

    ROS_INFO("Start the lam module ...");

    ros::Subscriber corner_feature_sub = nh.subscribe<sensor_msgs::PointCloud2>("/corner_feature_sharp", 100, cornerFeatureHandler);
    ros::Subscriber surface_feature_sub = nh.subscribe<sensor_msgs::PointCloud2>("/surface_feature_flat", 100, surfaceFeatureHandler);
    ros::Subscriber combine_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/combine_cloud", 100, combineCloudHandler);

    ros::Publisher pub_full = nh.advertise<sensor_msgs::PointCloud2>("/global_refined_map", 10);
    ros::Publisher pub_test = nh.advertise<sensor_msgs::PointCloud2>("/local_map", 10);
    ros::Publisher pub_odom = nh.advertise<nav_msgs::Odometry>("/odom_mark", 10);
    ros::Publisher pub_pose = nh.advertise<geometry_msgs::PoseArray>("/pose_array", 10);

    Eigen::Quaterniond q_curr(1, 0, 0, 0);
    Eigen::Vector3d t_curr(0, 0, 0);

    pcl::PointCloud<PointType>::Ptr pl_corn(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr pl_surf(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> pl_full_buf;

    // 记录接收到的点云scan数量
    int plcount = 0;
    // 记录滑动窗口中第一帧的数量，每次边缘化后递增
    int window_base = 0;

    Eigen::Quaterniond delta_q(1, 0, 0, 0);
    Eigen::Vector3d delta_t(0, 0, 0);

    // 使用哈希网格表存储特征
    unordered_map<VoxelStrcuture, OctoTree *> surf_map, corn_map;

    vector<Eigen::Quaterniond> delta_q_buf, q_buf;
    vector<Eigen::Vector3d> delta_t_buf, t_buf;

    // 滑窗优化，用以地图细化阶段
    SlidingWindowOpti opt_lsv(window_size, filter_num, thread_num);

    pcl::PointCloud<PointType> pl_surf_centor_map, pl_corn_centor_map;
    pcl::PointCloud<PointType> pl_corn_fil_map, pl_surf_fil_map;

    pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf(new pcl::KdTreeFLANN<PointType>());
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_corn(new pcl::KdTreeFLANN<PointType>());

    pcl::PointCloud<PointType> pl_send;
    // 原始坐标点、位姿转换后坐标点、核心位置向量 、方向向量、位姿转换后点距临近点中心的距离向量
    Eigen::Vector3d point_orig, point_aft_tran, ker_vec, direct_vector, dist_vec;
    uint a_size;
    PointType point_tmp;
    vector<int> point_search_id;
    vector<float> point_search_sqrt_dis;
    double range;
    Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
    ros::Time ct;
    geometry_msgs::PoseArray parray;
    parray.header.frame_id = "laser_init";
    thread *map_refine_thread = nullptr;

    while (nh.ok())
    {
        ros::spinOnce();

        // 地图优化完成后，执行此操作，包含边缘化及话题发布
        if (opt_lsv.read_refine_state() == 2)
        {
            ROS_INFO("After map refine . ");
            nav_msgs::Odometry laser_odom;
            laser_odom.header.frame_id = "laser_init";
            laser_odom.header.stamp = ct;

            // 发布位姿固定的点云数据
            pl_send.clear();
            ROS_INFO("The value of margi_size: %d", margi_size);
            for (int i = 0; i < margi_size; i += pub_skip)
            {
                ROS_INFO("Test one ...");
                trans.block<3, 3>(0, 0) = opt_lsv.so3_poses_[i].matrix();
                trans.block<3, 1>(0, 3) = opt_lsv.t_poses_[i];
                pcl::PointCloud<PointType> pcloud;
                pcl::transformPointCloud(*pl_full_buf[window_base + i], pcloud, trans);

                pl_send += pcloud;
            }
            pub_func(pl_send, pub_full, ct);

            for (int i = 0; i < margi_size; i++)
            {
                pl_full_buf[window_base + i] = nullptr;
            }

            for (int i = 0; i < window_size; i++)
            {
                q_buf[window_base + i] = opt_lsv.so3_poses_[i].unit_quaternion();
                t_buf[window_base + i] = opt_lsv.t_poses_[i];
            }

            // 发布位姿
            for (int i = window_base; i < plcount; i++)
            {
                parray.poses[i].orientation.w = q_buf[i].w();
                parray.poses[i].orientation.x = q_buf[i].x();
                parray.poses[i].orientation.y = q_buf[i].y();
                parray.poses[i].orientation.z = q_buf[i].z();
                parray.poses[i].position.x = t_buf[i].x();
                parray.poses[i].position.x = t_buf[i].x();
                parray.poses[i].position.x = t_buf[i].x();
            }
            pub_pose.publish(parray);

            // 边缘化操作并更新网格地图
            pl_surf_centor_map.clear();
            pl_corn_centor_map.clear();
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
            {
                if (iter->second->is2opt_)
                {
                    iter->second->root_centors_.clear();
                    iter->second->marginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors_);
                }
                pl_surf_centor_map += iter->second->root_centors_;
            }

            for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter)
            {
                if (iter->second->is2opt_)
                {
                    iter->second->root_centors_.clear();
                    iter->second->marginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors_);
                }
                pl_corn_centor_map += iter->second->root_centors_;
            }
            ROS_INFO("Test two ...");
            // 多线程模式，销毁指针
            if (map_refine_thread != nullptr)
            {
                delete map_refine_thread;
                map_refine_thread = nullptr;
            }

            // 更新voxel_windows_size
            OctoTree::voxel_windows_size_ -= margi_size;

            opt_lsv.free_voxel();
            window_base += margi_size;
            opt_lsv.set_refine_state(0);
            ROS_INFO("After map refine done. ");
        }

        // 各特征数据读取，筛选条件
        if (surf_buf.empty() || corn_buf.empty() || full_buf.empty())
        {
            continue;
        }

        mutex_data_buf.lock();
        uint64_t time_surf = surf_buf.front()->header.stamp.toNSec();
        uint64_t time_corn = corn_buf.front()->header.stamp.toNSec();
        uint64_t time_full = full_buf.front()->header.stamp.toNSec();

        if (time_corn != time_surf)
        {
            time_corn < time_surf ? corn_buf.pop() : surf_buf.pop();
            mutex_data_buf.unlock();
            continue;
        }
        if (time_corn != time_full)
        {
            time_corn < time_full ? corn_buf.pop() : full_buf.pop();
            mutex_data_buf.unlock();
            continue;
        }
        pcl::PointCloud<PointType>::Ptr pl_full(new pcl::PointCloud<PointType>);

        ct = full_buf.front()->header.stamp;

        // ROS中点云类型，转换至PCL中数据格式
        rosmsg2ptype(*surf_buf.front(), *pl_surf);
        rosmsg2ptype(*corn_buf.front(), *pl_corn);
        rosmsg2ptype(*full_buf.front(), *pl_full);
        corn_buf.pop();
        surf_buf.pop();
        full_buf.pop();

        // 全局雷达点数过少，丢弃该帧数据
        if (pl_full->size() < 5000)
        {
            mutex_data_buf.unlock();
            continue;
        }
        pl_full_buf.push_back(pl_full);
        mutex_data_buf.unlock();
        plcount++;
        // 记录滑窗口中scan点云帧数量
        OctoTree::voxel_windows_size_ = plcount - window_base;
        // 对corner特征下采样
        down_sampling_voxel(*pl_corn, corn_filter_length);

        double time_scan2map = ros::Time::now().toSec();
        // Scan2map方式求取位姿，里程计部分
        // 当有两个scan时执行该操作，第一帧为默认初始位姿
        if (plcount > accumulate_window)
        {
            ROS_INFO("Run The scan2map module. ");
            down_sampling_voxel(*pl_surf, surf_filter_length);

            // 需要几帧数据初始化雷达
            // 在不同的阶段使用不同的特征数据
            if (plcount <= scan2map_on)
            {
                // 与loam建图类似
                kdtree_surf->setInputCloud(pl_surf_fil_map.makeShared());
                kdtree_corn->setInputCloud(pl_corn_fil_map.makeShared());
            }
            else
            {
                // 开启新的scan2map
                kdtree_surf->setInputCloud(pl_surf_centor_map.makeShared());
                kdtree_corn->setInputCloud(pl_corn_centor_map.makeShared());
            }

            // 两次迭代求取位姿
            for (int itercount = 0; itercount < 2; itercount++)
            {
                // 采用LM优化算法，依据最近距离做判定
                VoxelDistance sld;
                sld.so3_pose_.setQuaternion(q_curr);
                sld.t_pose_ = t_curr;

                a_size = pl_surf->size();
                // 定位分为两个阶段，依据plcount 与 scan2map_on大小关系，采用两种方法进行位姿估计
                if (plcount <= scan2map_on)
                {
                    // 与LOAM定位建图方法类似
                    for (uint i = 0; i < a_size; i++)
                    {
                        int closest_point_num = 5;
                        point_orig << (*pl_surf)[i].x, (*pl_surf)[i].y, (*pl_surf)[i].z;
                        point_aft_tran = q_curr * point_orig + t_curr;
                        point_tmp.x = point_aft_tran[0];
                        point_tmp.y = point_aft_tran[1];
                        point_tmp.z = point_aft_tran[2];

                        /**
                         * @brief 最近邻查找
                         * @param[in] 基准点
                         * @param[in] 选取距基准点最近点数量
                         * @param[out] 所选取点的下标
                         * @param[out] 所选点相对于基准点的距离
                         * @return int 
                         */
                        kdtree_surf->nearestKSearch(point_tmp, closest_point_num, point_search_id, point_search_sqrt_dis);

                        if (point_search_sqrt_dis[closest_point_num - 1] > 5)
                        {
                            continue;
                        }
                        // 计算平面特征的协方差矩阵
                        Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
                        Eigen::Vector3d center_coor(0, 0, 0);
                        for (int j = 0; j < closest_point_num; j++)
                        {
                            Eigen::Vector3d tvec;
                            tvec[0] = pl_surf_fil_map[point_search_id[j]].x;
                            tvec[1] = pl_surf_fil_map[point_search_id[j]].y;
                            tvec[2] = pl_surf_fil_map[point_search_id[j]].z;
                            cov_mat += tvec * tvec.transpose();
                            center_coor += tvec;
                        }

                        center_coor /= closest_point_num;
                        cov_mat -= closest_point_num * center_coor * center_coor.transpose();
                        cov_mat /= closest_point_num;
                        // 计算特征值特征向量
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
                        // 对于平面特征，要求其中一个特征值明显小于其他两个，与较小特征值相对应的特征向量代表平面的方向；
                        // eigenvalues()[0]为较小的特征值，确保该特征协方差矩阵特征值满足要求
                        if (saes.eigenvalues()[2] < 25 * saes.eigenvalues()[0])
                        {
                            continue;
                        }

                        ker_vec = center_coor;
                        // 平面方向向量
                        direct_vector = saes.eigenvectors().col(0);
                        // 进行矩阵点乘，依据点到平面距离公式，简化计算判定参数
                        range = fabs(direct_vector.dot(point_aft_tran - ker_vec));

                        if (range > 1)
                        {
                            continue;
                        }

                        sld.push_surf(point_orig, ker_vec, direct_vector, (1 - 0.75 * range));
                    }

                    a_size = pl_corn->size();
                    for (uint i = 0; i < a_size; i++)
                    {
                        int closest_point_num = 5;
                        point_orig << (*pl_corn)[i].x, (*pl_corn)[i].y, (*pl_corn)[i].z;
                        point_aft_tran = q_curr * point_orig + t_curr;

                        point_tmp.x = point_aft_tran[0];
                        point_tmp.y = point_aft_tran[1];
                        point_tmp.z = point_aft_tran[2];
                        kdtree_corn->nearestKSearch(point_tmp, closest_point_num, point_search_id, point_search_sqrt_dis);

                        if ((point_search_sqrt_dis[closest_point_num - 1]) > 5)
                        {
                            continue;
                        }

                        Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
                        Eigen::Vector3d center_coor(0, 0, 0);
                        // 计算corner特征协方差矩阵
                        for (int j = 0; j < closest_point_num; j++)
                        {
                            Eigen::Vector3d tvec;
                            tvec[0] = pl_corn_fil_map[point_search_id[j]].x;
                            tvec[1] = pl_corn_fil_map[point_search_id[j]].y;
                            tvec[2] = pl_corn_fil_map[point_search_id[j]].z;
                            cov_mat += tvec * tvec.transpose();
                            center_coor += tvec;
                        }
                        center_coor /= closest_point_num;
                        cov_mat -= closest_point_num * center_coor * center_coor.transpose();
                        cov_mat /= closest_point_num;
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
                        // 确保eigenvalues()[2]足够大，才可进行后续运算
                        if (saes.eigenvalues()[2] < 4 * saes.eigenvalues()[1])
                        {
                            continue;
                        }

                        ker_vec = center_coor;
                        // 对于线特征的提取，要求其中一个特征值明显大于其他两个，与最大特征值对应的特征向量表示边缘线的方向
                        direct_vector = saes.eigenvectors().col(2);
                        dist_vec = point_aft_tran - ker_vec;
                        // norm()表示返回二范数；计算公式为点到直线距离公式的向量形式。
                        range = (dist_vec - direct_vector * direct_vector.transpose() * dist_vec).norm();
                        if (range > 1.0)
                        {
                            continue;
                        }
                        sld.push_line(point_orig, ker_vec, direct_vector, 0.5 * (1 - 0.75 * range));
                    }
                }
                else
                {
                    // BALM的方法，本质上与LOAM mapping方法还是相同的
                    for (uint i = 0; i < a_size; i++)
                    {
                        int closest_point_num = 5;
                        point_orig << (*pl_surf)[i].x, (*pl_surf)[i].y, (*pl_surf)[i].z;
                        point_aft_tran = q_curr * point_orig + t_curr;
                        point_tmp.x = point_aft_tran[0];
                        point_tmp.y = point_aft_tran[1];
                        point_tmp.z = point_aft_tran[2];

                        kdtree_surf->nearestKSearch(point_tmp, closest_point_num, point_search_id, point_search_sqrt_dis);

                        if ((point_search_sqrt_dis[0]) > 1.0)
                        {
                            continue;
                        }

                        // 找到距离最近的平面
                        range = 10;
                        for (int j = 0; j < closest_point_num; j++)
                        {
                            PointType &point_select = pl_surf_centor_map[point_search_id[j]];
                            Eigen::Vector3d center_coor(point_select.x, point_select.y, point_select.z);
                            Eigen::Vector3d direct(point_select.normal_x, point_select.normal_y, point_select.normal_z);
                            double dista = fabs(direct.dot(point_aft_tran - center_coor));
                            if (dista <= range && point_search_sqrt_dis[j] < 4.0)
                            {
                                ker_vec = center_coor;
                                direct_vector = direct;
                                range = dista;
                            }
                        }

                        // 将点存入优化器
                        sld.push_surf(point_orig, ker_vec, direct_vector, (1 - 0.75 * range));
                    }

                    // Corn特征处理
                    a_size = pl_corn->size();
                    for (uint i = 0; i < a_size; i++)
                    {
                        int closest_point_num = 3;
                        point_orig << (*pl_corn)[i].x, (*pl_corn)[i].y, (*pl_corn)[i].z;
                        point_aft_tran = q_curr * point_orig + t_curr;
                        point_tmp.x = point_aft_tran[0];
                        point_tmp.y = point_aft_tran[1];
                        point_tmp.z = point_aft_tran[2];

                        kdtree_corn->nearestKSearch(point_tmp, closest_point_num, point_search_id, point_search_sqrt_dis);
                        if ((point_search_sqrt_dis[0]) > 1)
                        {
                            continue;
                        }

                        range = 10;
                        double dis_record = 10;
                        for (int j = 0; j < closest_point_num; j++)
                        {
                            PointType &point_select = pl_corn_centor_map[point_search_id[j]];
                            Eigen::Vector3d center_coor(point_select.x, point_select.y, point_select.z);
                            Eigen::Vector3d direct(point_select.normal_x, point_select.normal_y, point_select.normal_z);
                            dist_vec = point_aft_tran - center_coor;
                            double dista = (dist_vec - direct * direct.transpose() * dist_vec).norm();
                            if (dista <= range)
                            {
                                ker_vec = center_coor;
                                direct_vector = direct;
                                range = dista;
                                dis_record = point_search_sqrt_dis[j];
                            }
                        }

                        if (range < 0.2 && sqrt(dis_record) < 1)
                        {
                            sld.push_line(point_orig, ker_vec, direct_vector, (1 - 0.75 * range));
                        }
                    }
                }

                sld.damping_iter();
                q_curr = sld.so3_pose_.unit_quaternion();
                t_curr = sld.t_pose_;
            }
        }
        // 统计优化执行耗时
        time_scan2map = ros::Time::now().toSec() - time_scan2map;
        // printf("Scan2map time: %lfs\n", ros::Time::now().toSec()-time_scan2map);

        if (plcount <= scan2map_on)
        {
            // 激光雷达帧小于scan2map_on，提取pl_surf_fil_map、pl_corn_fil_map特征
            ROS_INFO("test ...")时，
            trans.block<3, 3>(0, 0) = q_curr.matrix();
            trans.block<3, 1>(0, 3) = t_curr;

            pcl::transformPointCloud(*pl_surf, pl_send, trans);
            pl_surf_fil_map += pl_send;
            pcl::transformPointCloud(*pl_corn, pl_send, trans);
            pl_corn_fil_map += pl_send;
            down_sampling_voxel(pl_surf_fil_map, 0.2);
            down_sampling_voxel(pl_corn_fil_map, 0.2);
        }

        // 将新的位姿放入位姿阵列
        parray.header.stamp = ct;
        geometry_msgs::Pose apose;
        apose.orientation.w = q_curr.w();
        apose.orientation.x = q_curr.x();
        apose.orientation.y = q_curr.y();
        apose.orientation.z = q_curr.z();
        apose.position.x = t_curr.x();
        apose.position.y = t_curr.y();
        apose.position.z = t_curr.z();
        parray.poses.push_back(apose);
        pub_pose.publish(parray);

        // 获取里程计数据，与位姿阵列使用相同的数据，注意数据格式的不同
        nav_msgs::Odometry laser_odom;
        laser_odom.header.frame_id = "laser_init";
        laser_odom.header.stamp = ct;
        laser_odom.pose.pose.orientation.x = q_curr.x();
        laser_odom.pose.pose.orientation.y = q_curr.y();
        laser_odom.pose.pose.orientation.z = q_curr.z();
        laser_odom.pose.pose.orientation.w = q_curr.w();
        laser_odom.pose.pose.position.x = t_curr.x();
        laser_odom.pose.pose.position.y = t_curr.y();
        laser_odom.pose.pose.position.z = t_curr.z();
        pub_odom.publish(laser_odom);

        // 发布位姿变换关系
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(t_curr.x(), t_curr.y(), t_curr.z()));
        q.setW(q_curr.w());
        q.setX(q_curr.x());
        q.setY(q_curr.y());
        q.setZ(q_curr.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, laser_odom.header.stamp, "/laser_init", "/aft_mapped"));

        // 发布当前的full点云特征
        trans.block<3, 3>(0, 0) = q_curr.matrix();
        trans.block<3, 1>(0, 3) = t_curr;
        pcl::transformPointCloud(*pl_full, pl_send, trans);
        pub_func(pl_send, pub_test, ct);

        // 获取位姿变化量
        if (plcount > 1)
        {
            delta_t = q_buf[plcount - 2].matrix().transpose() * (t_curr - t_buf[plcount - 2]);
            delta_q = q_buf[plcount - 2].matrix().transpose() * q_curr.matrix();
        }

        q_buf.push_back(q_curr);
        t_buf.push_back(t_curr);
        delta_q_buf.push_back(delta_q);
        delta_t_buf.push_back(delta_t);

        // 多线程运行情况下，如果计算较慢会导致超内存，10由cut_voxel决定
        if (plcount - window_base - window_size > 10)
        {
            printf("Out of size\n");
            exit(0);
        }

        // Put current feature points into root voxel node
        cut_voxel(surf_map, pl_surf, q_curr.matrix(), t_curr, 0, plcount - 1 - window_base, window_size + 10);
        cut_voxel(corn_map, pl_corn, q_curr.matrix(), t_curr, 1, plcount - 1 - window_base, window_size + 10);

        // The center point of surf points and corn points
        // The normal_x(yz) in each point is normal vector for plane
        // or direction vector for line.
        pl_surf_centor_map.clear();
        pl_corn_centor_map.clear();
        ROS_INFO("Test 610");
        // Points in new scan have been distributed in corresponding root node voxel
        // Then continue to cut the root voxel until right size
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        {
            if (iter->second->is2opt_) // Sliding window of root voxel should have points
            {
                iter->second->root_centors_.clear();
                iter->second->recut(0, plcount - 1 - window_base, iter->second->root_centors_);
            }

            // Add up surf centor points.
            pl_surf_centor_map += iter->second->root_centors_;
            // You can add some distance restrictions in case that pl_surf_centor_map is too large.
            // You can also push points in root voxel into kdtree (loam mapping)
            // You can use "surf_map.erase(iter++)" to erase voxel for saving memory
        }
        ROS_INFO("Test 627");
        for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter)
        {
            if (iter->second->is2opt_)
            {
                iter->second->root_centors_.clear();
                iter->second->recut(0, plcount - 1 - window_base, iter->second->root_centors_);
            }
            pl_corn_centor_map += iter->second->root_centors_;
        }
        ROS_INFO("Test 637");
        // Begin map refine module
        ROS_INFO("The value of plcount: %d", plcount);
        if (plcount >= window_base + window_size && opt_lsv.read_refine_state() == 0)
        {
            ROS_INFO("Run the map refine module. ");
            for (int i = 0; i < window_size; i++)
            {
                opt_lsv.so3_poses_[i].setQuaternion(q_buf[window_base + i]);
                opt_lsv.t_poses_[i] = t_buf[window_base + i];
            }
            // Do not optimize first sliding window
            if (window_base == 0)
            {
                opt_lsv.set_refine_state(2);
            }
            else
            {
                // Push voxel map into optimizer
                for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
                {
                    if (iter->second->is2opt_)
                    {
                        iter->second->traversal_opt(opt_lsv);
                    }
                }

                for (auto iter = corn_map.begin(); iter != corn_map.end(); ++iter)
                {
                    if (iter->second->is2opt_)
                    {
                        iter->second->traversal_opt(opt_lsv);
                    }
                }

                // Begin iterative optimization
                // You can use multithreading or not.
                // We do not recommend use multiple thread on computer with poor performance

                // multithreading
                // map_refine_thread = new thread(&LM_SLWD_VOXEL::damping_iter, &opt_lsv);
                // map_refine_thread->detach();

                // non multithreading
                opt_lsv.damping_iter();
            }
        }

        // pose prediction
        t_curr = t_curr + q_curr * delta_t;
        q_curr = q_curr * delta_q;
    }
}
