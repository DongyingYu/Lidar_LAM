/**
 * @file system_ros.cc
 * @author Dongying (yudong2817@sina.com)
 * @brief 接收原始点云，提取并发布特征
 * @version 1.0
 * @date 2022-03-22
 *
 * @copyright Copyright (c) 2022
 */


#include "feature_extract.h"

ros::Publisher pub_laser_cloud;
ros::Publisher pub_combine_cloud;
ros::Publisher pub_corner_feature_sharp;
ros::Publisher pub_corner_feature_less_sharp;
ros::Publisher pub_surface_feature_flat;
ros::Publisher pub_surface_feature_less_flat;

// FeatureExtract::Ptr cloud_process = std::make_shared<FeatureExtract>();

void initialCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laser_cloud)
{
    ROS_INFO("Run the initialCloudHandler function.");
    pcl::PointCloud<pcl::PointXYZ> initial_cloud;
    pcl::fromROSMsg(*laser_cloud, initial_cloud);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(initial_cloud, initial_cloud, indices);

    // 阈值的设置后续需提取
    filterCloseDistance(initial_cloud,initial_cloud,0.3);
    FeatureExtract::Ptr cloud_process = std::make_shared<FeatureExtract>();
    cloud_process->computeCloudYawAngle(initial_cloud);
    pcl::PointCloud<PointType> combine_cloud;
    cloud_process->cloudDevideToScan(initial_cloud);
    combine_cloud = * cloud_process->getCombineCloud();

    int cloud_size = initial_cloud.points.size();
    ROS_INFO("The size of initial cloud: %d", cloud_size);

    cloud_process->computeCurvature();
    cloud_process->extractFeature();

    // 发布角点特征
    sensor_msgs::PointCloud2 corner_feature_sharp_msg;
    pcl::PointCloud<PointType> corner_feature_sharp;
    corner_feature_sharp = * cloud_process->getCornerFeature();
    pcl::toROSMsg(corner_feature_sharp, corner_feature_sharp_msg);
    corner_feature_sharp_msg.header.stamp = laser_cloud->header.stamp;
    corner_feature_sharp_msg.header.frame_id = "laser_init";
    pub_corner_feature_sharp.publish(corner_feature_sharp_msg); 

    // 发布less角点特征
    sensor_msgs::PointCloud2 corner_feature_less_sharp_msg;
    pcl::PointCloud<PointType> corner_feature_less_sharp;
    corner_feature_less_sharp = * cloud_process->getCornerFeatureLess();
    pcl::toROSMsg(corner_feature_less_sharp, corner_feature_less_sharp_msg);
    corner_feature_less_sharp_msg.header.stamp = laser_cloud->header.stamp;
    corner_feature_less_sharp_msg.header.frame_id = "laser_init";
    pub_corner_feature_less_sharp.publish(corner_feature_less_sharp_msg); 

    // 发布平面特征
    sensor_msgs::PointCloud2 surface_feature_flat_msg;
    pcl::PointCloud<PointType> surface_feature_flat;
    surface_feature_flat = * cloud_process->getSurfaceFeature();
    pcl::toROSMsg(surface_feature_flat, surface_feature_flat_msg);
    surface_feature_flat_msg.header.stamp = laser_cloud->header.stamp;
    surface_feature_flat_msg.header.frame_id = "laser_init";
    pub_surface_feature_flat.publish(surface_feature_flat_msg); 

    // 发布less平面特征
    sensor_msgs::PointCloud2 surface_feature_less_flat_msg;
    pcl::PointCloud<PointType> surface_feature_less_flat;
    surface_feature_less_flat = * cloud_process->getSurfaceFeatureLess();
    pcl::toROSMsg(surface_feature_less_flat, surface_feature_less_flat_msg);
    surface_feature_less_flat_msg.header.stamp = laser_cloud->header.stamp;
    surface_feature_less_flat_msg.header.frame_id = "laser_init";
    pub_surface_feature_less_flat.publish(surface_feature_less_flat_msg); 

    // 发布初始激光点云
    sensor_msgs::PointCloud2 initial_cloud_output;
    pcl::toROSMsg(initial_cloud, initial_cloud_output);
    initial_cloud_output.header.stamp = laser_cloud->header.stamp;
    initial_cloud_output.header.frame_id = "laser_init";
    pub_laser_cloud.publish(initial_cloud_output); 

    // 发布组合后的激光点云
    sensor_msgs::PointCloud2 combine_cloud_output;
    pcl::toROSMsg(combine_cloud, combine_cloud_output);
    combine_cloud_output.header.stamp = laser_cloud->header.stamp;
    combine_cloud_output.header.frame_id = "laser_init";
    pub_combine_cloud.publish(combine_cloud_output); 

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_feature");
    ros::NodeHandle nh;

    ROS_INFO("Start to process data ...");

    // 接收订阅的为16线整体点云
    ros::Subscriber initial_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, initialCloudHandler);
    pub_laser_cloud = nh.advertise<sensor_msgs::PointCloud2>("/initial_cloud", 100);
    pub_combine_cloud = nh.advertise<sensor_msgs::PointCloud2>("/combine_cloud", 100);
    pub_corner_feature_sharp = nh.advertise<sensor_msgs::PointCloud2>("/corner_feature_sharp", 100);
    pub_corner_feature_less_sharp = nh.advertise<sensor_msgs::PointCloud2>("/corner_feature_less_sharp", 100);
    pub_surface_feature_flat = nh.advertise<sensor_msgs::PointCloud2>("/surface_feature_flat", 100);
    pub_surface_feature_less_flat = nh.advertise<sensor_msgs::PointCloud2>("/surface_feature_less_flat", 100);

    ros::spin();

    return 0;
}
