#include "ros/ros.h"
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/Header.h"
#include "sensor_msgs/CameraInfo.h"
#include "stereo_msgs/DisparityImage.h"
//#include <image_get/MyImage.h>
//#include <image_get/Blobs.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>


#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
//#include <opencv2/dpm.hpp>
//#include <dirent.h>

//#include "dpm.hpp"

#include <boost/array.hpp>
#include <vector>

using namespace cv;
using namespace std;

ros::Publisher pub_camera_info;
//ros::Publisher pub_disparity;
//const std::vector<double> D2 = {0.0, 0.0, 0.0, 0.0, 0.0};

boost::array<double, 9> K = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
boost::array<double, 9> R = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
boost::array<double, 12> P = {525.0, 0.0, 319.5, -39.375, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0};
//boost::array<double, 12> P = {525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0};

void trackerCallback(const sensor_msgs::Image::ConstPtr& msg)
{

//boost::array<double, 5> D2= {0.0, 0.0, 0.0, 0.0, 0.0};
//std::vector<double> D2;// = {0.0, 0.0, 0.0, 0.0, 0.0};
//D2.insert(0.0);D2.insert(0.0);D2.insert(0.0);D2.insert(0.0);D2.insert(0.0);



	ROS_INFO("Hola");
  //ROS_INFO("I receive: [%s]", msg->header->seq.c_str());

	//try
	//{
	
		sensor_msgs::CameraInfo camera_info;
		camera_info.header.seq = msg->header.seq;
		camera_info.header.stamp = msg->header.stamp;
		camera_info.header.frame_id = msg->header.frame_id;

		camera_info.height = 480;
		camera_info.width = 640;
		camera_info.distortion_model = "plumb_bob";
		//camera_info.D = D2;
		camera_info.K = K;
		camera_info.R = R;
		camera_info.P = P	;
		camera_info.binning_x = 0;
		camera_info.binning_y = 0;
		camera_info.roi.x_offset = 0;
		camera_info.roi.y_offset = 0;
		camera_info.roi.height = 0;
		camera_info.roi.width = 0;
		camera_info.roi.do_rectify = false;

		pub_camera_info.publish(camera_info);

		// DISPARIDAD
		/*stereo_msgs::DisparityImage disparity;
		disparity.header = camera_info.header;
		disparity.f = 0.0075;
		disparity.T = 580;
		disparity.min_disparity = 100;
		disparity.max_disparity = 50000;
		disparity.image = *msg;*/
		//disparity.image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
		//disparity.image.data.resize(disparity.image.height * disparity.image.width * sizeof(float));
		//float* ptr = reinterpret_cast<float*>(&disparity.image.data[0]);

		//MODIFICAR VALORES DE LA MATRIZ z=fb/d
		/*for (unsigned int i = 0; i < disparity.image.height; i++){
			unsigned int row_index = i * disparity.image.width;
			for (unsigned int j = 0; j < disparity.image.width; j++){
				float index = row_index + j;
				disparity.image.data[index] = index/10;
				
			}
		}*/

		//pub_disparity.publish(disparity);

		waitKey(30);
	//}
	//catch (cv_bridge::Exception& e)
	/*{
		ROS_ERROR("error cv_bridge");
	}*/

}



int main(int argc, char **argv)
{

  ros::init(argc, argv, "addCameraInfo");

  ros::NodeHandle n;

	ROS_INFO("INIT ADD_CAMERA_INFO");

  ros::Subscriber sub = n.subscribe("camera/depth/image", 10, trackerCallback);
	pub_camera_info = n.advertise<sensor_msgs::CameraInfo>("camera/depth/camera_info",30);
	//pub_disparity = n.advertise<stereo_msgs::DisparityImage>("camera/depth/disparity",30);

  ros::spin();

  return 0;
}

