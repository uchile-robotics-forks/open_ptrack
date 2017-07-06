#include "ros/ros.h"//Used for launch file parameter parsing
#include <string>//Used for rois message vector
#include <vector>
#include <sstream>

//Included for files
#include <iostream>
#include <fstream>
#include "stdio.h"
#include "dirent.h"

//Publish Messages
#include "std_msgs/String.h"
#include "opt_msgs/Track.h"


//Subscribe Messages
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

// Image Transport
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

// Used to display OPENCV images
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



//using namespace sensor_msgs;
using namespace message_filters::sync_policies;
using namespace sensor_msgs::image_encodings;
using namespace opt_msgs;
using namespace cv;

//NOTE: Where is the best place to put these
//typedef ApproximateTime<Image, opt_msgs::Track> ApproximatePolicy;
//typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;

cv::Scalar darkBlue(130,0,0);
cv::Scalar white(255,255,255);

class targetViewerNode
{
  private:
    // Define Node
    ros::NodeHandle node_;

    // Subscribe to Messages
    ros::Subscriber sub_image_;
		ros::Subscriber sub_track_;


    // Launch file Parameters
    bool color_image;
		cv::Mat im_;

  public:

    explicit targetViewerNode(const ros::NodeHandle& nh):
    node_(nh)
    {

      //Read mode from launch file
      std::string mode="";
      node_.param(ros::this_node::getName() + "/mode", mode, std::string("none"));
      ROS_INFO("Selected mode: %s",mode.c_str());

      if(mode.compare("roi_display")==0){
        ROS_INFO("MODE: %s",mode.c_str());

        //Read parameter stating if the image is grayscale or with colors
        node_.param(ros::this_node::getName()+"/color_image", color_image, true);

        // Subscribe to Messages
        sub_image_ = node_.subscribe("input_image",1,imageCb);
        sub_track_ = node_.subscribe("input_track",5,trackCb);

 
      }else{

        ROS_INFO("Unknown mode:%s  Please set to {roi_display} in targetViewer.launch",mode.c_str());
      }
      // Visualization
      cv::namedWindow("Target", 0 ); // non-autosized
      cv::startWindowThread();

    }

    void imageCb(const sensor_msgs::ImageConstPtr& image_msg){

      std::string filename = image_msg->header.frame_id.c_str();
      std::string imgName = filename.substr(filename.find_last_of("/")+1);

      //ROS_INFO("targetViewer Callback called for image: %s", imgName.c_str());

      //Use CV Bridge to convert images
      cv_bridge::CvImagePtr cv_ptr;
      if (color_image)
      {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
      }
      else
      {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
      }

			im_ = cv::Mat(cv_ptr->image);

    }

		void trackCb(const opt_msgs::Track::ConstPtr& track_msg){

			int x = track_msg->box_2D.x + 0.125*track_msg->box_2D.width;
			int y = track_msg->box_2D.y - 0.02*track_msg->box_2D.height;
			int w = 0.75*track_msg->box_2D.width;
			int h = 0.5*track_msg->box_2D.height;

			Rect rect(x,y,w,h);

			if (rect.area() > 0)
				rectangle(im_, rect, Scalar(255,255,255));

			cv::imshow("Target", im_);
			cv::waitKey(10);

		}

    ~targetViewerNode()
    {
    }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "targetViewer");
  ros::NodeHandle n;
  targetViewerNode targetViewerNode(n);

  ros::spin();

	destroyWindow("Target");

  return 0;
}

