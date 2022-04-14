/**
 * @file lam_ros.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 定位与建图模块主函数，topic发送接受，回调函数
 * @version 1.0
 * @date 2022-03-25
 *
 * @copyright Copyright (c) 2022
 */

#include "lam_process.h"
#include "parameters.h"

// 建立点云特征数据存储队列
queue<sensor_msgs::PointCloud2ConstPtr> corner_feature_buf;
queue<sensor_msgs::PointCloud2ConstPtr> surface_feature_buf;
queue<sensor_msgs::PointCloud2ConstPtr> combine_cloud_buf;

std::mutex mutex_data_buf;

/**
   * @brief 点、平面特征及完整点云的回调函数
   * @param[in] 特征信息及点云数据
   * @return void 
   */
void cornerFeatureHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    corner_feature_buf.push(msg);
    mutex_data_buf.unlock();
}

void surfaceFeatureHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    surface_feature_buf.push(msg);
    mutex_data_buf.unlock();
}

void combineCloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    mutex_data_buf.lock();
    combine_cloud_buf.push(msg);
    mutex_data_buf.unlock();
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

    initialParameters(nh);

    LamProcess::Ptr localizeThread = std::make_shared<LamProcess>();

    // 四元数的几种初始化方式：https://www.cnblogs.com/lovebay/p/13820750.html
    Eigen::Quaterniond q_curr(1, 0, 0, 0);
    Eigen::Vector3d t_curr(0, 0, 0);

    pcl::PointCloud<PointType>::Ptr corn_cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr surf_cloud(new pcl::PointCloud<PointType>);
    vector<pcl::PointCloud<PointType>::Ptr> full_cloud_buf;

    // 记录接收到的点云scan数量
    int cloud_frame_cnt = 0;
    // 记录滑动窗口中第一帧的数量，每次边缘化后递增
    int window_base = 0;

    Eigen::Quaterniond delta_q(1, 0, 0, 0);
    Eigen::Vector3d delta_t(0, 0, 0);

    // 使用哈希网格表存储特征
    unordered_map<VoxelStructure, OctoTree *> surf_octo_map, corn_octo_map;

    vector<Eigen::Quaterniond> delta_q_buf, q_buf;
    vector<Eigen::Vector3d> delta_t_buf, t_buf;

    // 滑窗优化，用以地图细化阶段
    SlidingWindowOpti opt_lsv(window_size, filter_num, thread_num);

    // surf_cloud_centor_map、corn_..._map数据，每次来一个scan就会进行更新，然后均存入八叉树中
    pcl::PointCloud<PointType> surf_cloud_centor_map, corn_cloud_centor_map;
    // 存入数据不清空，每次有新数据存入，均进行一次下采样滤波
    pcl::PointCloud<PointType> corn_cloud_filter_map, surf_cloud_filter_map;

    pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf(new pcl::KdTreeFLANN<PointType>());
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_corn(new pcl::KdTreeFLANN<PointType>());

    pcl::PointCloud<PointType> cloud_send;
    // 原始坐标点、位姿转换后坐标点、核心位置向量 、方向向量、位姿转换后点距临近点中心的距离向量
    Eigen::Vector3d point_orig, point_aft_tran, centor_vec, direct_vector, dist_vec;
    uint feature_size;
    PointType point_tmp;
    vector<int> point_search_id;
    vector<float> point_search_sqrt_dis;
    double range;
    Eigen::Matrix4d trans(Eigen::Matrix4d::Identity());
    ros::Time cur_time;
    geometry_msgs::PoseArray pose_array;
    pose_array.header.frame_id = "laser_init";
    thread *map_refine_thread = nullptr;

    while (nh.ok())
    {
        ros::spinOnce();

        localizeThread->qua_incre_ = delta_q;
        localizeThread->trans_incre_ = delta_t;
        // 地图优化完成后，执行此操作，包含边缘化及话题发布
        if (opt_lsv.readRefineState() == 2)
        {
            ROS_INFO("After map refine . ");
            nav_msgs::Odometry laser_odom;
            laser_odom.header.frame_id = "laser_init";
            laser_odom.header.stamp = cur_time;

            // 发布位姿固定的点云数据
            cloud_send.clear();
            ROS_INFO("The value of margi_size: %d", margi_size);
            // 发布点云时需跳过的帧数，可以灵活调整（设置为5，太小可视化显示卡顿），依据可视化效果，这里只用在可视化上，
            // 影响不大，若需保存全局地图，需要重新添加代码
            for (int i = 0; i < margi_size; i += pub_skip)
            {
                ROS_INFO("Test one ...");
                trans.block<3, 3>(0, 0) = opt_lsv.so3_poses_[i].matrix();
                trans.block<3, 1>(0, 3) = opt_lsv.t_poses_[i];
                pcl::PointCloud<PointType> pcloud;
                pcl::transformPointCloud(*full_cloud_buf[window_base + i], pcloud, trans);

                cloud_send += pcloud;
            }
            pubFunction(cloud_send, pub_full, cur_time);

            // 清空所需边缘化帧的数据
            for (int i = 0; i < margi_size; i++)
            {
                full_cloud_buf[window_base + i] = nullptr;
            }

            for (int i = 0; i < window_size; i++)
            {
                q_buf[window_base + i] = opt_lsv.so3_poses_[i].unit_quaternion();
                t_buf[window_base + i] = opt_lsv.t_poses_[i];
            }

            // 发布位姿,在这里发布位姿array会对先前发布的数据进行更新，在rviz展示上会变化
            for (int i = window_base; i < cloud_frame_cnt; i++)
            {
                pose_array.poses[i].orientation.w = q_buf[i].w();
                pose_array.poses[i].orientation.x = q_buf[i].x();
                pose_array.poses[i].orientation.y = q_buf[i].y();
                pose_array.poses[i].orientation.z = q_buf[i].z();
                pose_array.poses[i].position.x = t_buf[i].x();
                pose_array.poses[i].position.x = t_buf[i].x();
                pose_array.poses[i].position.x = t_buf[i].x();
            }
            pub_pose.publish(pose_array);

            // 边缘化操作并更新网格地图
            surf_cloud_centor_map.clear();
            corn_cloud_centor_map.clear();
            for (auto iter = surf_octo_map.begin(); iter != surf_octo_map.end(); ++iter)
            {
                if (iter->second->is2opt_)
                {
                    iter->second->root_centors_.clear();
                    iter->second->scanMarginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors_);
                }
                surf_cloud_centor_map += iter->second->root_centors_;
            }

            for (auto iter = corn_octo_map.begin(); iter != corn_octo_map.end(); ++iter)
            {
                if (iter->second->is2opt_)
                {
                    iter->second->root_centors_.clear();
                    iter->second->scanMarginalize(0, margi_size, q_buf, t_buf, window_base, iter->second->root_centors_);
                }
                corn_cloud_centor_map += iter->second->root_centors_;
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

            opt_lsv.releaseVoxel();
            window_base += margi_size;
            opt_lsv.setRefineState(0);
            ROS_INFO("After map refine done. ");
        }

        // 各特征数据读取，筛选条件
        if (surface_feature_buf.empty() || corner_feature_buf.empty() || combine_cloud_buf.empty())
        {
            continue;
        }

        mutex_data_buf.lock();
        uint64_t time_surf = surface_feature_buf.front()->header.stamp.toNSec();
        uint64_t time_corn = corner_feature_buf.front()->header.stamp.toNSec();
        uint64_t time_full = combine_cloud_buf.front()->header.stamp.toNSec();

        if (time_corn != time_surf)
        {
            time_corn < time_surf ? corner_feature_buf.pop() : surface_feature_buf.pop();
            mutex_data_buf.unlock();
            continue;
        }
        if (time_corn != time_full)
        {
            time_corn < time_full ? corner_feature_buf.pop() : combine_cloud_buf.pop();
            mutex_data_buf.unlock();
            continue;
        }
        pcl::PointCloud<PointType>::Ptr full_cloud(new pcl::PointCloud<PointType>);

        cur_time = combine_cloud_buf.front()->header.stamp;

        // ROS中点云类型，转换至PCL中数据格式
        rosmsgToPointtype(*surface_feature_buf.front(), *surf_cloud);
        rosmsgToPointtype(*corner_feature_buf.front(), *corn_cloud);
        rosmsgToPointtype(*combine_cloud_buf.front(), *full_cloud);
        corner_feature_buf.pop();
        surface_feature_buf.pop();
        combine_cloud_buf.pop();

        // 全局雷达点数过少，丢弃该帧数据
        if (full_cloud->size() < 5000)
        {
            mutex_data_buf.unlock();
            continue;
        }
        full_cloud_buf.push_back(full_cloud);
        mutex_data_buf.unlock();
        cloud_frame_cnt++;
        // 记录滑窗口中scan点云帧数量
        OctoTree::voxel_windows_size_ = cloud_frame_cnt - window_base;

        // 对corn、surf特征做矫正处理
        if (cloud_frame_cnt > 1)
        {
            localizeThread->cloudDeskew(*corn_cloud);
            localizeThread->cloudDeskew(*surf_cloud);
        }

        // 对corner特征下采样
        downSamplingVoxel(*corn_cloud, corn_filter_length);
        downSamplingVoxel(*surf_cloud, surf_filter_length);

        double time_scan2map = ros::Time::now().toSec();
        // Scan2map方式求取位姿，里程计部分
        // 当有两个scan时执行该操作，第一帧为默认初始位姿
        // 每来一个scan求取一次位姿。
        if (cloud_frame_cnt > accumulate_window)
        {
            ROS_INFO("Run The scan2map module. ");

            // 需要几帧数据初始化雷达
            // 在不同的阶段使用不同的特征数据
            // 将无序点云存入八叉树，后续便可基于领域的方式进行查找处理
            if (cloud_frame_cnt <= scan2map_on)
            {
                // 与loam建图类似
                kdtree_surf->setInputCloud(surf_cloud_filter_map.makeShared());
                kdtree_corn->setInputCloud(corn_cloud_filter_map.makeShared());
            }
            else
            {
                // 开启新的scan2map
                kdtree_surf->setInputCloud(surf_cloud_centor_map.makeShared());
                kdtree_corn->setInputCloud(corn_cloud_centor_map.makeShared());
            }

            // 两次迭代求取位姿：q_curr,t_curr
            for (int itercount = 0; itercount < 2; itercount++)
            {
                // 采用LM优化算法，依据最近距离做判定
                VoxelDistance sld;
                sld.so3_pose_.setQuaternion(q_curr);
                sld.t_pose_ = t_curr;

                feature_size = surf_cloud->size();
                // 采用了一种新的平面特征、线特征表示形式，同时残差计算公式也不同，简化了计算，提升了算法效果
                // 定位分为两个阶段，依据cloud_frame_cnt 与 scan2map_on大小关系，采用两种方法进行位姿估计
                if (cloud_frame_cnt <= scan2map_on)
                {
                    // 与LOAM定位建图方法类似
                    for (uint i = 0; i < feature_size; i++)
                    {
                        int closest_point_num = 5;
                        // 这里通过位姿变换是为点云的最近邻搜索提供依据
                        point_orig << (*surf_cloud)[i].x, (*surf_cloud)[i].y, (*surf_cloud)[i].z;
                        point_aft_tran = q_curr * point_orig + t_curr;
                        point_tmp.x = point_aft_tran[0];
                        point_tmp.y = point_aft_tran[1];
                        point_tmp.z = point_aft_tran[2];
                        // 将经初始位姿转换后的点，在八叉树地图中查找得到最近点。
                        /**
                         * @brief 最近邻查找
                         * @param[in] 基准点
                         * @param[in] 选取距基准点最近点数量
                         * @param[out] 所选取点的下标
                         * @param[out] 所选点相对于基准点的距离
                         * @return ptr 
                         */
                        kdtree_surf->nearestKSearch(point_tmp, closest_point_num, point_search_id, point_search_sqrt_dis);

                        if (point_search_sqrt_dis[closest_point_num - 1] > 5)
                        {
                            continue;
                        }
                        // 计算平面特征的协方差矩阵
                        // 对应于论文中计算平均点坐标及协方差矩阵A
                        Eigen::Matrix3d cov_mat(Eigen::Matrix3d::Zero());
                        Eigen::Vector3d center_coor(0, 0, 0);
                        for (int j = 0; j < closest_point_num; j++)
                        {
                            Eigen::Vector3d tvec;
                            tvec[0] = surf_cloud_filter_map[point_search_id[j]].x;
                            tvec[1] = surf_cloud_filter_map[point_search_id[j]].y;
                            tvec[2] = surf_cloud_filter_map[point_search_id[j]].z;
                            cov_mat += tvec * tvec.transpose();
                            center_coor += tvec;
                        }

                        center_coor /= closest_point_num;
                        cov_mat -= closest_point_num * center_coor * center_coor.transpose();
                        cov_mat /= closest_point_num;
                        // 计算特征值、特征向量
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat);
                        // 对于平面特征，要求其中一个特征值明显小于其他两个，与较小特征值相对应的特征向量代表平面的法向方向；
                        // eigenvalues()[0]为较小的特征值，确保该特征协方差矩阵特征值满足要求
                        if (saes.eigenvalues()[2] < 25 * saes.eigenvalues()[0])
                        {
                            continue;
                        }

                        centor_vec = center_coor;
                        // 平面方向向量
                        direct_vector = saes.eigenvectors().col(0);
                        // 进行矩阵点乘，依据点到平面距离公式，简化计算判定参数
                        range = fabs(direct_vector.dot(point_aft_tran - centor_vec));

                        if (range > 1)
                        {
                            continue;
                        }
                        // 使用中心点及方向向量来唯一表示一个平面特征
                        sld.pushSurf(point_orig, centor_vec, direct_vector, (1 - 0.75 * range));
                    }

                    feature_size = corn_cloud->size();
                    for (uint i = 0; i < feature_size; i++)
                    {
                        int closest_point_num = 5;
                        point_orig << (*corn_cloud)[i].x, (*corn_cloud)[i].y, (*corn_cloud)[i].z;
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
                            tvec[0] = corn_cloud_filter_map[point_search_id[j]].x;
                            tvec[1] = corn_cloud_filter_map[point_search_id[j]].y;
                            tvec[2] = corn_cloud_filter_map[point_search_id[j]].z;
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

                        centor_vec = center_coor;
                        // 对于线特征的提取，要求其中一个特征值明显大于其他两个，与最大特征值对应的特征向量表示边缘线的方向
                        direct_vector = saes.eigenvectors().col(2);
                        // 表示距离向量
                        dist_vec = point_aft_tran - centor_vec;
                        // norm()表示返回二范数；计算公式为点到直线距离公式的向量形式。
                        // 按照常识vector3d表示的是，三行一列的向量
                        // direct_vector * direct_vector.transpose()为方向向量的投影矩阵，因其行列式平方和为1，
                        // 故不需要进行归一化处理，
                        range = (dist_vec - direct_vector * direct_vector.transpose() * dist_vec).norm();
                        if (range > 1.0)
                        {
                            continue;
                        }
                        sld.pushLine(point_orig, centor_vec, direct_vector, 0.5 * (1 - 0.75 * range));
                    }
                }
                else
                {
                    // BALM的方法，本质上与LOAM mapping方法还是相同的
                    for (uint i = 0; i < feature_size; i++)
                    {
                        int closest_point_num = 5;
                        point_orig << (*surf_cloud)[i].x, (*surf_cloud)[i].y, (*surf_cloud)[i].z;
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
                            PointType &point_select = surf_cloud_centor_map[point_search_id[j]];
                            Eigen::Vector3d center_coor(point_select.x, point_select.y, point_select.z);
                            Eigen::Vector3d direct(point_select.normal_x, point_select.normal_y, point_select.normal_z);
                            double dista = fabs(direct.dot(point_aft_tran - center_coor));
                            if (dista <= range && point_search_sqrt_dis[j] < 4.0)
                            {
                                centor_vec = center_coor;
                                direct_vector = direct;
                                range = dista;
                            }
                        }

                        // 将点存入优化器
                        sld.pushSurf(point_orig, centor_vec, direct_vector, (1 - 0.75 * range));
                    }

                    // Corn特征处理
                    feature_size = corn_cloud->size();
                    for (uint i = 0; i < feature_size; i++)
                    {
                        int closest_point_num = 3;
                        point_orig << (*corn_cloud)[i].x, (*corn_cloud)[i].y, (*corn_cloud)[i].z;
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
                            PointType &point_select = corn_cloud_centor_map[point_search_id[j]];
                            Eigen::Vector3d center_coor(point_select.x, point_select.y, point_select.z);
                            Eigen::Vector3d direct(point_select.normal_x, point_select.normal_y, point_select.normal_z);
                            dist_vec = point_aft_tran - center_coor;
                            double dista = (dist_vec - direct * direct.transpose() * dist_vec).norm();
                            if (dista <= range)
                            {
                                centor_vec = center_coor;
                                direct_vector = direct;
                                range = dista;
                                dis_record = point_search_sqrt_dis[j];
                            }
                        }

                        if (range < 0.2 && sqrt(dis_record) < 1)
                        {
                            sld.pushLine(point_orig, centor_vec, direct_vector, (1 - 0.75 * range));
                        }
                    }
                }

                sld.dampingIter();
                q_curr = sld.so3_pose_.unit_quaternion();
                t_curr = sld.t_pose_;
            }
        }
        // 统计优化执行耗时
        time_scan2map = ros::Time::now().toSec() - time_scan2map;
        // printf("Scan2map time: %lfs\n", ros::Time::now().toSec()-time_scan2map);

        if (cloud_frame_cnt <= scan2map_on)
        {
            // 激光雷达帧小于scan2map_on，提取surf_cloud_filter_map、corn_cloud_filter_map特征
            ROS_INFO("test ...");
            trans.block<3, 3>(0, 0) = q_curr.matrix();
            trans.block<3, 1>(0, 3) = t_curr;

            pcl::transformPointCloud(*surf_cloud, cloud_send, trans);
            surf_cloud_filter_map += cloud_send;
            pcl::transformPointCloud(*corn_cloud, cloud_send, trans);
            corn_cloud_filter_map += cloud_send;
            downSamplingVoxel(surf_cloud_filter_map, 0.2);
            downSamplingVoxel(corn_cloud_filter_map, 0.2);
        }

        // 将新的位姿放入位姿阵列
        pose_array.header.stamp = cur_time;
        geometry_msgs::Pose current_pose;
        current_pose.orientation.w = q_curr.w();
        current_pose.orientation.x = q_curr.x();
        current_pose.orientation.y = q_curr.y();
        current_pose.orientation.z = q_curr.z();
        current_pose.position.x = t_curr.x();
        current_pose.position.y = t_curr.y();
        current_pose.position.z = t_curr.z();
        pose_array.poses.push_back(current_pose);
        pub_pose.publish(pose_array);

        // 获取里程计数据，与位姿阵列使用相同的数据，注意数据格式的不同
        nav_msgs::Odometry laser_odom;
        laser_odom.header.frame_id = "laser_init";
        laser_odom.header.stamp = cur_time;
        laser_odom.pose.pose.orientation.x = q_curr.x();
        laser_odom.pose.pose.orientation.y = q_curr.y();
        laser_odom.pose.pose.orientation.z = q_curr.z();
        laser_odom.pose.pose.orientation.w = q_curr.w();
        laser_odom.pose.pose.position.x = t_curr.x();
        laser_odom.pose.pose.position.y = t_curr.y();
        laser_odom.pose.pose.position.z = t_curr.z();
        pub_odom.publish(laser_odom);

        // 发布当前帧位姿变换关系
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

        // 发布当前帧full点云特征
        trans.block<3, 3>(0, 0) = q_curr.matrix();
        trans.block<3, 1>(0, 3) = t_curr;
        pcl::transformPointCloud(*full_cloud, cloud_send, trans);
        pubFunction(cloud_send, pub_test, cur_time);

        // 帧数大于等于2时，获取位姿间变化量，新帧的位姿由scan2map部分计算
        if (cloud_frame_cnt > 1)
        {
            // 可理解为维护两帧数据间的位姿增量
            delta_t = q_buf[cloud_frame_cnt - 2].matrix().transpose() * (t_curr - t_buf[cloud_frame_cnt - 2]);
            delta_q = q_buf[cloud_frame_cnt - 2].matrix().transpose() * q_curr.matrix();
        }

        // 将位姿及增量放入缓存buf
        q_buf.push_back(q_curr);
        t_buf.push_back(t_curr);
        delta_q_buf.push_back(delta_q);
        delta_t_buf.push_back(delta_t);

        // 多线程运行情况下，如果计算较慢会导致超内存，10由cutVoxel决定
        if (cloud_frame_cnt - window_base - window_size > 10)
        {
            printf("Out of size\n");
            exit(0);
        }

        // 将边、面特征做网格划分
        localizeThread->cutVoxel(surf_octo_map, surf_cloud, q_curr.matrix(), t_curr, 0, cloud_frame_cnt - 1 - window_base, window_size + 10);
        localizeThread->cutVoxel(corn_octo_map, corn_cloud, q_curr.matrix(), t_curr, 1, cloud_frame_cnt - 1 - window_base, window_size + 10);

        // 角点特征以及平面特征的中心点
        // normal_x(yz)表示平面特征的法向量或线特征的方向向量
        surf_cloud_centor_map.clear();
        corn_cloud_centor_map.clear();
        // 新的scan被划分为对应的网格地图中，需要进一步将网格划分，直到获得对应的尺度
        // 对当前新的线、面平面特征做网格划分处理，起始层为0
        // 对每一个特征所处的区域进行自适应划分
        for (auto iter = surf_octo_map.begin(); iter != surf_octo_map.end(); ++iter)
        {
            if (iter->second->is2opt_)
            {
                iter->second->root_centors_.clear();
                // point_vec_tran_由cutVoxel函数生成
                iter->second->recutVoxel(0, cloud_frame_cnt - 1 - window_base, iter->second->root_centors_);
            }

            // 添加平面中心点
            surf_cloud_centor_map += iter->second->root_centors_;
            // 可以添加一些距离约束，以防止平面中心地图过大；
            // 也可将存入网格中的点，存入八叉树中，类似于loam
            // 也可使用 surf_octo_map.erase(iter++)  删除网格，以节省内存空间
        }
        for (auto iter = corn_octo_map.begin(); iter != corn_octo_map.end(); ++iter)
        {
            if (iter->second->is2opt_)
            {
                iter->second->root_centors_.clear();
                iter->second->recutVoxel(0, cloud_frame_cnt - 1 - window_base, iter->second->root_centors_);
            }
            corn_cloud_centor_map += iter->second->root_centors_;
        }
        // 显示所处文件名及行号
        ROS_INFO("The name of current file is : %s", __FILE__);
        ROS_INFO("The current line number : %d", __LINE__);
        // 开启地图优化模块
        ROS_INFO("The value of cloud_frame_cnt: %d", cloud_frame_cnt);
        // 对位姿buf中的数据执行优化
        if (cloud_frame_cnt >= window_base + window_size && opt_lsv.readRefineState() == 0)
        {
            ROS_INFO("Run the map refine module. ");
            for (int i = 0; i < window_size; i++)
            {
                opt_lsv.so3_poses_[i].setQuaternion(q_buf[window_base + i]);
                opt_lsv.t_poses_[i] = t_buf[window_base + i];
            }
            // 第一个滑动窗口不做优化
            if (window_base == 0)
            {
                opt_lsv.setRefineState(2);
            }
            else
            {
                // 将网格地图信息加入优化器
                for (auto iter = surf_octo_map.begin(); iter != surf_octo_map.end(); ++iter)
                {
                    if (iter->second->is2opt_)
                    {
                        iter->second->traversalOpt(opt_lsv);
                    }
                }

                for (auto iter = corn_octo_map.begin(); iter != corn_octo_map.end(); ++iter)
                {
                    if (iter->second->is2opt_)
                    {
                        iter->second->traversalOpt(opt_lsv);
                    }
                }

                // 开启迭代优化
                // 多线程优化方式， 对电脑性能要求较高
                // map_refine_thread = new thread(&LM_SLWD_VOXEL::dampingIter, &opt_lsv);
                // map_refine_thread->detach();

                // 非多线程
                opt_lsv.dampingIter();
            }
        }

        // 位姿预测计算
        t_curr = t_curr + q_curr * delta_t;
        q_curr = q_curr * delta_q;
    }
}
