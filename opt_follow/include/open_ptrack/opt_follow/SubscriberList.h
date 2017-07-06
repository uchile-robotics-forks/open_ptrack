/*#include <ros/ros.h>
#include <ros/package.h>

#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include "opt_msgs/DetectionArray.h"
#include "opt_msgs/Detection.h"
#include "opt_msgs/TrackArray.h"
#include "opt_msgs/Track.h"

//C++
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <boost/filesystem.hpp>
#include <valarray>
#include <numeric>

//typedef message_filters::Subscriber mfs;

typedef message_filters::Subscriber<sensor_msgs::Image>  SImage;
typedef message_filters::Subscriber<opt_msgs::TrackArray>  STrack_array;
typedef message_filters::Subscriber<opt_msgs::DetectionArray>  SDetection_array;

class SubscriberList
{
	public:
		SImage sub_image_;
		STrack_array sub_tracks_;
		SDetection_array sub_detect_;
		SImage sub_disparity_;

		SubscriberList(SImage sub_image,
				STrack_array sub_tracks,
				SDetection_array sub_detect,
				SImage sub_disparity)
		{
			sub_image_ = sub_image;
			sub_tracks_ = sub_tracks;
			sub_detect_ = sub_detect;
			sub_disparity_ = sub_disparity;
		}

		void setSubs(SImage sub_image,
				STrack_array sub_tracks,
				SDetection_array sub_detect,
				SImage sub_disparity)
		{
			sub_image_ = sub_image;
			sub_tracks_ = sub_tracks;
			sub_detect_ = sub_detect;
			sub_disparity_ = sub_disparity;
		}

};*/