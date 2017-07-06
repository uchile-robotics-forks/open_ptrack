#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <ros/package.h>
#include "std_msgs/String.h"
#include "sensor_msgs/image_encodings.h"
#include <cv_bridge/cv_bridge.h>

#include "opt_msgs/DetectionArray.h"
#include "opt_msgs/Detection.h"
#include "opt_msgs/BoundingBox2D.h"
#include "open_ptrack/opt_utils/conversions.h"

#include <fstream>
#include <iostream>

#include <sstream>

using namespace std;


ofstream outputFile;
open_ptrack::opt_utils::Conversions converter;

void detectionsCb(const opt_msgs::DetectionArray::ConstPtr& msg)
{
	std::string frame_id = msg->header.frame_id;
	int idx = msg->header.seq;

	// Read camera intrinsic parameters:
	Eigen::Matrix3f intrinsic_matrix;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			intrinsic_matrix(i, j) = msg->intrinsic_matrix[i * 3 + j];


	opt_msgs::BoundingBox2D max_box;
	max_box.x = 0;
	max_box.y = 0;
	max_box.width = 0;
	max_box.height = 0;


	for (int i = 0; i < msg->detections.size(); i++)
	{
		Eigen::Vector3f centroid3d(msg->detections[i].centroid.x, msg->detections[i].centroid.y, msg->detections[i].centroid.z);
        Eigen::Vector3f centroid2d = converter.world2cam(centroid3d, intrinsic_matrix);

        // theoretical person top point:
        Eigen::Vector3f top3d(msg->detections[i].top.x, msg->detections[i].top.y, msg->detections[i].top.z);
        Eigen::Vector3f top2d = converter.world2cam(top3d, intrinsic_matrix);

        // Define Rect and make sure it is not out of the image:
        int h = centroid2d(1) - top2d(1);
        int w = h * 2 / 3.0;
        int x = std::max(0, int(centroid2d(0) - w / 2.0));
        int y = std::max(0, int(top2d(1)));
        h = std::min(int(480 - y), int(h));
        w = std::min(int(640 - x), int(w));

		opt_msgs::BoundingBox2D bbox = msg->detections[i].box_2D;
		if (h*w > max_box.width*max_box.height && h<480 && w<640)
		{
			max_box.x = x;
			max_box.y = y;
			max_box.width = w;
			max_box.height = h;
		}
	}

	if (max_box.width*max_box.height > 0)
	{
		outputFile << frame_id << " " << idx << " " << max_box.x << " " << max_box.y << " " << max_box.x+max_box.width << " " << max_box.y+max_box.height << endl;
	}


}



int main(int argc, char **argv)
{
	outputFile.open("out1.txt");
	cout << "Archivo abierto" << endl;

	ros::init(argc, argv, "featurePublisher");

	ros::NodeHandle n;

	ros::Subscriber image_sub = n.subscribe("detector/detections", 1, detectionsCb);

	ros::spin();

	outputFile.close();
	cout << "Archivo cerrado" << endl;

	return 0;
}