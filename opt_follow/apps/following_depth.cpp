#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>
#include "std_msgs/String.h"
#include "std_msgs/UInt32.h"
#include <cv_bridge/cv_bridge.h>
//#include <bender_srvs/ImageDetection.h>
#include "opt_msgs/DetectionArray.h"
#include "opt_msgs/Detection.h"
#include "opt_msgs/TrackArray.h"
#include "opt_msgs/Track.h"
#include "open_ptrack/opt_utils/conversions.h"

//Opencv
//#include "cv.h"
//#include "highgui.h"
//#include <ml.h>	
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>

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
//#include <cmath>

//Time Synchronizer
// NOTE: Time Synchronizer conflicts with QT includes may need to investigate
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "open_ptrack/detection/detection.h"
#include "open_ptrack/opt_follow/FeatureExtractor.h"

#include "opt_msgs/Onoff.h"

using namespace cv;
using namespace std;

class FollowerNode
{

private:

	ros::NodeHandle node_;

	message_filters::Subscriber<sensor_msgs::Image> sub_image_;
	message_filters::Subscriber<opt_msgs::TrackArray> sub_tracks_;
	message_filters::Subscriber<opt_msgs::DetectionArray> sub_detect_;
	message_filters::Subscriber<sensor_msgs::Image> sub_disparity_;

	bool activo;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, opt_msgs::TrackArray, opt_msgs::DetectionArray, sensor_msgs::Image> MySyncPolicy;
	typedef message_filters::Synchronizer<MySyncPolicy> MySync;
	boost::shared_ptr<MySync> sync3;

	ros::Publisher pub;

	ros::ServiceServer service;
	ros::ServiceClient client; // trackArray client

	bool init_done_flag_;
	int target_id_;
	int curr_max_id_;

	FeatureExtractor feat_ext;
	vector<Histogram> prev_hist_features_target_;
	vector<Histogram> hist_features_target_;
	vector< vector<Histogram> > hist_distractors_;
	//const int max_distractors_ = 10;
	int max_distractors_;
	int idx_pop_distractor_;

	bool set_camera_parameters_;
	open_ptrack::opt_utils::Conversions converter_;
	Eigen::Matrix3f intrinsic_matrix_;
	int height_;
	int width_;

	bool flag_perdido_recien;
	double target_dist_ref_; // node param = 0.15
	double target_dist_thr_; // se ajusta a target_dist_ref_, pero se va adaptando
	double target_max_dist_thr_;

	bool feat_only_start; // node param = false
	int nro_only_start;	// node param = 5; numero de frames para extraer caract.
	int nro_start;

	int target_frames; // numero de frames diferentes que recuerda

	vector< vector<Histogram> > old_features; // size -> node param = 3

	vector<double> target_dists_;
	int target_dists_idx_;
	vector<double> distractor_dists_;
	int distractor_dists_idx_;

	double _min_height;	//Minima altura de target
	double _max_height;	//Maxima altura de target
	double _target_distance; //Distancia del target en metros
	bool _close_range;

	double time_;

public:

	explicit FollowerNode(const ros::NodeHandle& nh):
	node_(nh)
	{
		activo = false;
		init_done_flag_ = false;
		target_id_ = -1;
		curr_max_id_ = -1;
		max_distractors_ = 10;
		idx_pop_distractor_ = 0;
		set_camera_parameters_ = false;
		flag_perdido_recien = false;
		target_max_dist_thr_ = 0.15;
		nro_start = 0;
		target_dists_idx_ = 0;
		distractor_dists_idx_ = 0;
		_target_distance = -1;
		_close_range = false;
		time_ = 0;

		//feat_ext.init();

      	node_.param("target_dist_thr", target_dist_ref_, 0.15);
		target_dist_thr_ = target_dist_ref_;

		node_.param("get_feature_only_start", feat_only_start, false);
		
		node_.param("nro_only_start", nro_only_start, 5);
		
		node_.param("target_frames", target_frames, 3);
		old_features.resize(target_frames);

		double training_factor;
		node_.param("training_factor", training_factor, 0.125); // 1/8
		feat_ext.setTrainingFactor(training_factor);

		pub = node_.advertise<opt_msgs::Track>("tracker/target",30);
	
		sub_image_.subscribe(node_, "/HaarDispAdaColorImage", 3);
		sub_tracks_.subscribe(node_, "/tracker/tracks_smoothed", 3);
		sub_detect_.subscribe(node_, "/detector/detections", 3);
		sub_disparity_.subscribe(node_, "/depth_image", 3);

		sync3.reset(new MySync(MySyncPolicy(10), 
			sub_image_, 
			sub_tracks_, 
			sub_detect_, 
			sub_disparity_));
	    
	    
	    /*sync3->registerCallback(boost::bind(&FollowerNode::callbackNoFeatCarac,
	    	  this,
	          _1,
	          _2,
	          _3, 
	          _4));
	    */
	    service = node_.advertiseService("Active", &FollowerNode::Active, this);

	    client = node_.serviceClient<opt_msgs::Onoff>("/moving_average_filter_node/Active",1);

	}


void clearNode()
{
	init_done_flag_ = false;
	target_id_ = -1;
	curr_max_id_ = -1;
	idx_pop_distractor_ = 0;
	flag_perdido_recien = false;
	target_max_dist_thr_ = 0.15;
	nro_start = 0;
	target_dists_idx_ = 0;
	distractor_dists_idx_ = 0;
	target_dist_thr_ = target_dist_ref_;
	_target_distance = -1;

	prev_hist_features_target_.clear();
	hist_features_target_.clear();
	hist_distractors_.clear();

	old_features.clear();
	old_features.resize(target_frames);

	target_dists_.clear();
	distractor_dists_.clear();
}


void printFeatureWeights(void)
{
	feat_ext.printWeights();
}

cv::Mat convertROSImageToCV(const sensor_msgs::ImageConstPtr &image)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{	
		cv_ptr = cv_bridge::toCvCopy(image, "bgr8");
		//cv_ptr = cv_bridge::toCvCopy(image, image->encoding);
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}

	return cv::Mat(cv_ptr->image);
}

cv::Mat convertROSDisparityImageToCV(const stereo_msgs::DisparityImageConstPtr &disparity)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{	
		cv_ptr = cv_bridge::toCvCopy(disparity->image, disparity->image.encoding);
		//cv_ptr = cv_bridge::toCvCopy(image, image->encoding);
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}

	return cv::Mat(cv_ptr->image);
}

cv::Mat convertROSDepthImageToCV(const sensor_msgs::ImageConstPtr &depth)
{
	cv_bridge::CvImagePtr cv_ptr;
	try
	{	
		cv_ptr = cv_bridge::toCvCopy(depth, depth->encoding);
		//cv_ptr = cv_bridge::toCvCopy(image, image->encoding);
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}

	return cv::Mat(cv_ptr->image);
}


void convertCVToROSImage(cv::Mat input, sensor_msgs::Image &output, std::string encoding)
{
	cv_bridge::CvImage cv_i;
	cv_i.header.stamp = ros::Time::now();
	cv_i.header.frame_id = "image";
	cv_i.encoding = encoding; //sensor_msgs::image_encodings::MONO8; BGR8;
	cv_i.image = input;

	sensor_msgs::Image im;
	cv_i.toImageMsg(im);
	output = im;
}


int isIndexInTracks(opt_msgs::TrackArray &tracks_msg, int id)
{

	for (int i = 0; i < tracks_msg.tracks.size(); i++)
	{
		if (tracks_msg.tracks[i].id == id)
			return i;
	}
	
	return -1;
}


void printHistograms(vector<Histogram> &hists)
{
	cout << "Histogramas: " << endl;

	for (int i = 0; i < hists.size(); i++)
	{
		Histogram hist = hists.at(i);

		cout << "Histograma nro " << i << ": " << endl;
		cout << "Size: " << hist.size() << endl;
		for (int j = 0; j < hist.size(); j++)
		{
			cout << hist[j] << ", ";
		}
		cout << endl;
	}
}


void getBbox(opt_msgs::Track &msg_track, Rect &rect)
{
	//Rect rect;
	//cout << "antes  ";
	/*rect.x = (int) (msg_track.box_2D.x + 0.2*msg_track.box_2D.width);
	rect.y = (int) (msg_track.box_2D.y + 0.01*msg_track.box_2D.height);
	rect.width = (int) (0.6*msg_track.box_2D.width);
	rect.height = (int) (0.4*msg_track.box_2D.height);*/

	//cout << "rect " << rect << endl;
	/*rect.x = msg_track.box_2D.x + 0.125*msg_track.box_2D.width;
	rect.y = msg_track.box_2D.y - 0.02*msg_track.box_2D.height;
	rect.width = 0.75*msg_track.box_2D.width;
	rect.height = 0.55*msg_track.box_2D.height;*/

	/*rect.x = msg_track.box_2D.x;
	rect.y = msg_track.box_2D.y;
	rect.width = msg_track.box_2D.width;
	rect.height = msg_track.box_2D.height;*/

	rect.x = msg_track.box_2D.x + 0.125*msg_track.box_2D.width;
	rect.y = msg_track.box_2D.y - 0.02*msg_track.box_2D.height;
	rect.width = 0.75*msg_track.box_2D.width;
	rect.height = 0.5*msg_track.box_2D.height;

	//cout << "rect gb: " << rect << endl;
	//return rect;
}


// Recorta imagen
cv::Mat cropImage(cv::Mat &image, Rect &rect)
{
	// Achicar rect
	rect.x = rect.x + 0.1*rect.width;
	rect.y = rect.y + 0.1*rect.height;
	rect.width = 0.8*rect.width;
	rect.height = 0.8*rect.height;

	int cols = image.cols;
	int rows = image.rows;
	int x2 = min(rect.x+rect.width, cols);
	int y2 = min(rect.y+rect.height, rows);

	rect.x = max(rect.x, 0);
	rect.y = max(rect.y, 0);
	rect.width = x2 - rect.x;
	rect.height = y2 - rect.y;

	if (rect.y+rect.height >= rows)
		rect.height = rows - rect.y - 1;
	if (rect.x+rect.width >= cols)
		rect.width = cols - rect.x -1;

	cv::Mat rtn = image(rect).clone();

	return rtn;
}


cv::Mat cropMask(cv::Mat &depth, Rect rect_target, double distance)
{
	cv::Mat cropped_depth = depth(rect_target).clone();
	cv::Mat mask(cropped_depth.rows, cropped_depth.cols, CV_8UC1);
	for (int r = 0; r < mask.rows; r++)
	{
		for (int c = 0; c < mask.cols; c++)
		{
			double curr_dist = (double) cropped_depth.at<float>(r,c);
			if ((curr_dist < distance + 0.3) && (curr_dist > distance - 0.3))
			{
				mask.at<uchar>(r,c) = 1;
			}
			else
			{
				mask.at<uchar>(r,c) = 0;
			}
		}
	}

	return mask;
}

void viewMaskedImage(cv::Mat &image, cv::Mat &mask, String nameWindow)
{
	// Si servicio no esta activo, no hacer nada
	if (!activo)
	{
		return;
	}

	cv::Mat im = image.clone();
	for (int r = 0; r < image.rows; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			if (mask.at<uchar>(r,c) == 0)
			{
				im.at<cv::Vec3b>(r,c)[0] = 0;
				im.at<cv::Vec3b>(r,c)[1] = 0;
				im.at<cv::Vec3b>(r,c)[2] = 0;
			}
		}
	}

	imshow(nameWindow, im);

}


bool overlaps(cv::Rect &r1, cv::Rect &r2)
{
	Rect r3 = r1 & r2; //interseccion
	return ((r3.area()) > 0);
}

double interOverUnion(cv::Rect &r1, cv::Rect &r2)
{
	Rect r_union = r1 | r2;
	Rect r_inter = r1 & r2;

	if (r_union.area() == 0)
		return 0.0;

	double rtn = (1.0 * r_inter.area() ) / r_union.area();
	return rtn;
}

void addDist(vector<double> &distances, double dist, int max_size, int *idx)
{
	int vector_size = distances.size();
	if (vector_size < max_size)
	{
		distances.push_back(dist);
	}
	else
	{
		distances.at(*idx) = dist;
		*idx = (*idx + 1) % vector_size;
	}
}



//Distractor list
void updateDistractorList(cv::Mat &img, vector<Rect> &r_detections, Rect &r_target,  vector<double> &dists, double target_dist, cv::Mat &depth)
{	
	cv::Mat img_target = cropImage(img, r_target);
	cv::Mat mask = cropMask(depth, r_target, target_dist);

	viewMaskedImage(img_target, mask, "Positivo");

	
	vector<Histogram> hist_features_target;
	
	feat_ext.getFeaturesMasked(img_target, hist_features_target, mask);
	
	prev_hist_features_target_ = hist_features_target_;
	hist_features_target_ = hist_features_target;
	
	double dist_between_target = feat_ext.compareArrayHist(prev_hist_features_target_, hist_features_target);
	addDist(target_dists_, dist_between_target, 40, &target_dists_idx_);

	double dist_old = feat_ext.compareArrayHist(old_features.at(0), hist_features_target);
	//addDist(target_dists_, dist_old, 40, &target_dists_idx_);

	cout << "Distancia entre target previo: " << dist_between_target << endl;

	// Si una muestra del target es lo suficientemente diferente a la anterior, almacenarla en old_features
	if (old_features.size() > 0)
	{
		if (feat_ext.compareArrayHist(hist_features_target_, old_features.at(0)) > target_max_dist_thr_)
		{
			for (int i = old_features.size(); i > 1; i--)
			{
				old_features.at(i-1) = old_features.at(i-2);
			}
			old_features.at(0) = hist_features_target_;
		}
	}

	for (int i = 0; i < r_detections.size(); i++)
	{
		Rect rect_detection = r_detections.at(i);

		if (overlaps(r_target, rect_detection))
			continue;

		vector<Histogram> hist_distractor;
		cv::Mat blob_im = cropImage(img,rect_detection);
		cv::Mat mask = cropMask(depth, rect_detection, dists.at(i));
		
		if (blob_im.cols < 70 || blob_im.rows < 100)
			continue;

		feat_ext.getFeaturesMasked(blob_im, hist_distractor, mask);
		
		if (hist_distractors_.size() < max_distractors_)
		{
			hist_distractors_.push_back(hist_distractor);
		}
		else
		{
			hist_distractors_.at(idx_pop_distractor_) = hist_distractor;
			idx_pop_distractor_ = (idx_pop_distractor_ + 1) % max_distractors_;
		}

		

		double distractor_dist = feat_ext.compareArrayHist(hist_features_target, hist_distractor);
		addDist(distractor_dists_, distractor_dist, 80, &distractor_dists_idx_);
		cout << "Distancia entre diferentes   : " << distractor_dist << endl;

		//cv::imshow("Muestra negativa", blob_im);
		viewMaskedImage(blob_im, mask, "Negativo");
		//cv::waitKey(30);
	}
}



void updateFeatureWeights()
{
	for (int i = 0; i < feat_ext.getNroFeatures(); i++)
	{
		Histogram prev_hist, last_hist;
		prev_hist = prev_hist_features_target_.at(i);
		last_hist = hist_features_target_.at(i);

		double last_target_dist = feat_ext.d_bhattacharyya(prev_hist, last_hist);

		double average_dist = 0;
		for (int j = 0; j < hist_distractors_.size(); j++)
		{
			Histogram distractor_hist = hist_distractors_.at(j).at(i);
			//double distractor_dist = feat_ext.d_bhattacharyya(last_hist, distractor_hist);
			average_dist += feat_ext.d_bhattacharyya(last_hist, distractor_hist);;
		}

		if (hist_distractors_.size() > 0)
			average_dist = average_dist / hist_distractors_.size();
		else
		{
			if (i == 0 || i == 7 || i == 8) //HS full, HS head, CSLBP head
				average_dist = 1;
			else
				average_dist = 0.5;//0.5
		}

		double score_i = average_dist - last_target_dist;
		//cout << "score " << i << ": " << score_i << endl;
		feat_ext.updateWeight(i, score_i);
	}

	feat_ext.normalizeWeights();
}

void updateDistanceThreshold()
{
	double sum_target = accumulate(target_dists_.begin(), target_dists_.end(), 0.0);
	double mean_target =  sum_target / target_dists_.size();

	double sum_distractor = accumulate(distractor_dists_.begin(), distractor_dists_.end(), 0.0);
	double mean_distractor =  sum_distractor / distractor_dists_.size();

	double accum;

	accum = 0.0;
	for(vector<double>::const_iterator it = target_dists_.begin(); it != target_dists_.end(); it++) 
	{
    	accum += (*it - mean_target) * (*it - mean_target);
	};
	double stdev_target = sqrt(accum/(target_dists_.size()-1));

	accum = 0.0;
	for(vector<double>::const_iterator it = distractor_dists_.begin(); it != distractor_dists_.end(); it++) 
	{
    	accum += (*it - mean_distractor) * (*it - mean_distractor);
	};
	double stdev_distractor = sqrt(accum/(distractor_dists_.size()-1));

	cout << "MEAN TARGET    : " << mean_target << " \t";
	cout << "STD TARGET     : " << stdev_target << endl;
	cout << "MEAN DISTRACTOR: " << mean_distractor << "\t";
	cout << "STD DISTRACTOR : " << stdev_distractor << endl;
	
	if (isnan(stdev_target))
	{
		target_dist_thr_ = target_dist_ref_;
		cout << "Threshold0 changed to: " << target_dist_thr_ << endl;
		return;
	}

	if (isnan(mean_distractor) || isnan(stdev_distractor))
	{
		target_dist_thr_ = 1.5*mean_target + 1.5*stdev_target;
		cout << "Threshold1 changed to:" << target_dist_thr_ << endl;
		return;
	}

	if (mean_target+2*stdev_target < mean_distractor-4*stdev_distractor)
	{
		target_dist_thr_ = (mean_target+2*stdev_target + mean_distractor-4*stdev_distractor)/2.0;
		cout << "Threshold2 changed to: " << target_dist_thr_ << endl;
		return;
	}

	if (mean_target+2*stdev_target < mean_distractor-2*stdev_distractor)
	{
		target_dist_thr_ = (mean_target+2*stdev_target + mean_distractor-2*stdev_distractor)/2.0;
		cout << "Threshold3 changed to: " << target_dist_thr_ << endl;
		return;
	}
	if (mean_target < mean_distractor && (stdev_target + stdev_distractor) != 0)
	{
		double x = (mean_distractor - mean_target)/(stdev_target + stdev_distractor);
		target_dist_thr_ = (mean_target+ x*stdev_target + mean_distractor- x*stdev_distractor)/2.0;
		cout << "Threshold4 changed to: " << target_dist_thr_ << endl;
		return;
	}
	else
	{
		target_dist_thr_ = mean_target + 1*stdev_target;
		cout << "Threshold5 changed to: " << target_dist_thr_ << endl;
		return;
	}

	/*if (mean_target+4*stdev_target < mean_distractor-4*stdev_distractor)
	{
		target_dist_thr_ = (mean_target+3.5*stdev_target + mean_distractor-2.5*stdev_distractor)/2.0;
		cout << "Threshold1 changed to: " << target_dist_thr_ << endl;
		return;
	}
	else
	{
		target_dist_thr_ = mean_target + 5*stdev_target;
		cout << "Threshold3 changed to: " << target_dist_thr_ << endl;
		return;
	}*/
}



void getRect(opt_msgs::Detection &detection, cv::Rect &rect)
{

	Eigen::Vector3f centroid3d(detection.centroid.x, detection.centroid.y, detection.centroid.z);
    Eigen::Vector3f centroid2d = converter_.world2cam(centroid3d, intrinsic_matrix_);

    // theoretical person top point:
    Eigen::Vector3f top3d(detection.top.x, detection.top.y, detection.top.z);
    Eigen::Vector3f top2d = converter_.world2cam(top3d, intrinsic_matrix_);

    // Define Rect and make sure it is not out of the image:
    int h = centroid2d(1) - top2d(1);
    int w = h * 2 / 3.0;
    int x = std::max(0, int(centroid2d(0) - w / 2.0));
    int y = std::max(0, int(top2d(1)));
    h = std::min(int(height_ - y), int(h));
    w = std::min(int(width_ - x), int(w));

    rect.x = x;
    rect.y = y;
    rect.width = w;
    rect.height = h;
}


bool findTargetRect(cv::Mat &image, opt_msgs::TrackArray &tracks_msg, opt_msgs::DetectionArray &detections_msg, Rect &r_target, vector<Rect> &rects, double *target_dist, vector<double> &dists)
{
	int idx = isIndexInTracks(tracks_msg, target_id_);
	if (idx < 0)
		return false;

	if (image.cols < (70*1.25) || image.rows < (100*1.25))
		return false;

	// Buscar solamente si el track es valido y confiable
	opt_msgs::Track track = tracks_msg.tracks[idx];
	int matchs;
	int detection_idx;
	if (track.confidence > 1 && track.visibility == opt_msgs::Track::VISIBLE)
	{
		Rect r_track;
		getBbox(track, r_track);
		//cv::rectangle(image, r_track, Scalar(0,255,255));

		matchs = 0;
		for (int i = 0; i < detections_msg.detections.size(); i++)
		{
			opt_msgs::Detection detection = detections_msg.detections[i];
			Rect r_detection;
			getRect(detection, r_detection);
			//cv::rectangle(image, r_detection, Scalar(0,255,0));

			if (interOverUnion(r_track, r_detection) > 0.65 && r_detection.width > 70*1.25 && r_detection.height > 100*1.25)
			{
				detection_idx = i;
				matchs++;
			}
		}
	}
	else
	{
		return false;	
	}

	if (matchs == 1)
	{
		rects.clear();
		for (int i = 0; i < detections_msg.detections.size(); i++)
		{
			opt_msgs::Detection detection = detections_msg.detections[i];
			Rect r_detection;
			getRect(detection, r_detection);

			if (detection_idx == i)
			{
				r_target = r_detection;
				*target_dist = detection.distance;
				//cv::rectangle(image, r_target, Scalar(0,0,255));
			}
			else
			{
				rects.push_back(r_detection);
				dists.push_back(detection.distance);
				//cv::rectangle(image, r_detection, Scalar(255,0,0));
			}
		}

		return true;
	}

	return false;
}


bool findTrackRect(opt_msgs::Track &track, opt_msgs::DetectionArray &detections_msg, Rect &rect_track)
{
	Rect r_track;
	getBbox(track, r_track);

	if (r_track.width < 70 || r_track.height < 100)
		return false;

	int track_idx;
	int matchs = 0;
	for (int i = 0; i < detections_msg.detections.size(); i++)
	{
		opt_msgs::Detection detection = detections_msg.detections[i];
		Rect r_detection;
		getRect(detection, r_detection);

		if (interOverUnion(r_track, r_detection) > 0.5)
		{
			rect_track = r_detection;
			track_idx = i;
			matchs++;
		}

	}

	if (rect_track.width < 70 || rect_track.height < 100)
		return false;

	if (matchs == 1)
	{
		return true;
	}

	return false;
}


void setTargetHeight(double height)
{
	_min_height = height;
	_max_height = height;
}

void updateTargetHeight(double height)
{
	if (height > _max_height)
		_max_height = height;
	if (height < _min_height)
		_min_height = height;
}

bool compareTargetHeight(double height)
{
	if (height < _max_height*1.2 && height > _min_height*0.8)
		return true;

	return false;
}


double meanDepth(cv::Mat &depth)
{
	double suma = 0;
	double n = 0;
	for (int r = 0; r < depth.rows; r++)
	{
		for (int c = 0; c < depth.cols; c++)
		{
			double auxi = (double) depth.at<float>(r,c);
			if (!isnan(auxi))
			{
				suma = suma + auxi;
				n++;
			}
		}
	}

	if (n == 0)
		return 1000;

	return suma/n;
}


bool targetIsClose(cv::Mat &depth)
{
	if (_target_distance > 0.4)
		return false;

	if (meanDepth(depth) > 0.4)
		return false;

	return true;
}


//void callbackNoFeatCarac(const sensor_msgs::ImageConstPtr &image_msg, const opt_msgs::TrackArray::ConstPtr &tracks_msg, const opt_msgs::DetectionArray::ConstPtr &detections_msg, const stereo_msgs::DisparityImageConstPtr &disparity_msg)
void callbackNoFeatCarac(const sensor_msgs::ImageConstPtr &image_msg, const opt_msgs::TrackArray::ConstPtr &tracks_msg, const opt_msgs::DetectionArray::ConstPtr &detections_msg, const sensor_msgs::ImageConstPtr &depth_msg)
{
	Mat image = convertROSImageToCV(image_msg);
	//Mat disp = convertROSDisparityImageToCV(disparity_msg);
	Mat depth = convertROSDepthImageToCV(depth_msg);

	opt_msgs::Track track_target_msg;
	track_target_msg.id = 0;
	track_target_msg.x = 0;
	track_target_msg.y = 0;
	track_target_msg.height = 0;
	track_target_msg.confidence = -1;

	if (!set_camera_parameters_)
	{
		intrinsic_matrix_;
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				intrinsic_matrix_(i, j) = detections_msg->intrinsic_matrix[i * 3 + j];
		
		height_ = image.rows;
		width_ = image.cols;
		set_camera_parameters_ = true;
	}

	double distance_thr = 2.5;
	if (!init_done_flag_)
	{
		ROS_INFO("Buscando target");
		for (int i = 0; i < tracks_msg->tracks.size(); i++)
		{
			opt_msgs::Track msg_track = tracks_msg->tracks[i];
			cout << "distancia: " << msg_track.distance << " confidencia: " << msg_track.confidence << endl;
			if (msg_track.distance < distance_thr && msg_track.confidence > 0.8) //se define por una distancia minima
			{
				//TODO: almacenar informacion (crear clase Target con caracteristicas, cara, etc.)

				Rect rect_target;
				opt_msgs::DetectionArray det_arr = *detections_msg;

				if (findTrackRect(msg_track, det_arr, rect_target))
				{
					target_id_ = msg_track.id;

					track_target_msg = msg_track;

					_target_distance = msg_track.distance;

					cout << "Target encontrado: " << target_id_ << endl;

					setTargetHeight(msg_track.height);	//altura del target

					cv::Mat blob_im = cropImage(image,rect_target);
					cv::Mat mask = cropMask(depth, rect_target, msg_track.distance);

					feat_ext.getFeaturesMasked(blob_im, hist_features_target_, mask);
					prev_hist_features_target_ = hist_features_target_;
					
					//cv::Mat blob_im = cropImage(image,rect_target);
					//feat_ext.getFeatures(blob_im, hist_features_target_, mask);

					init_done_flag_ = true;

					for (int i = 0; i < old_features.size(); i++)
					{
						old_features.at(i) = hist_features_target_;
					}

					//cv::imshow("Reencontrado", blob_im);
					viewMaskedImage(blob_im, mask, "Positivo");

					Rect r_aux;
					getBbox(msg_track, r_aux);
					cv::rectangle(image, r_aux, Scalar(255,255,0));

					break;
				}
			}
		}
	}
	else // !init_done_flag
	{
		opt_msgs::TrackArray track_array = *tracks_msg;
		opt_msgs::DetectionArray detection_array = *detections_msg;

		int idx = isIndexInTracks(track_array, target_id_);
		if (idx >= 0) //Si no se ha perdido el track
		{
			track_target_msg = track_array.tracks[idx];

			_target_distance = track_target_msg.distance;

			Rect r_target;
			double target_dist;
			vector<Rect> rects;
			vector<double> dists;

			if (findTargetRect(image, track_array, detection_array, r_target, rects, &target_dist, dists) && (!feat_only_start || nro_start<nro_only_start))
			{   
				// Actualizar distractores
				updateDistractorList(image, rects, r_target, dists, target_dist, depth);
				updateFeatureWeights();
				updateDistanceThreshold();

				//Actualizar altura de target
				updateTargetHeight(track_target_msg.height);

				// Ignorar a los tracks que esten vivos hace mas de 1 segundo
				for (int i = track_array.tracks.size(); i > 0; i--)
				{
					double curr_age = track_array.tracks[i-1].age;
					if (curr_age > 1)
					{
						curr_max_id_ = track_array.tracks[i-1].id;
						break;
					}
				}
				//curr_max_id_ = track_array.tracks[track_array.tracks.size()-1].id;

				if (feat_only_start)
				{
					nro_start++;
					cout << "Target's frame found to save" << endl;
				}


				// Dibujar
				cv::rectangle(image, r_target, Scalar(0,0,255));//Dibujar target rojo
				for (int i = 0; i < rects.size(); i++)//Dibujar los tracks que no son target en azul
				{
					cv::rectangle(image, rects.at(i), Scalar(255,0,0));
				}

			}
			else //Dibujar el track del target
			{
				opt_msgs::Track track = track_array.tracks[idx];
				Rect r_track;
				getBbox(track, r_track);
				cv::rectangle(image, r_track, Scalar(255,255,255));
			}
		}
		else //Se perdio el track, hay que encontrar al target dentro de las demas detecciones
		{
			if (!flag_perdido_recien)
			{
				cout << "Target perdido" << endl;
				flag_perdido_recien = true;

				_close_range = targetIsClose(depth);
			}

			if (_close_range && targetIsClose(depth))
			{
				track_target_msg.id = curr_max_id_+1;
				track_target_msg.x = 0.1;
				track_target_msg.y = 0;
				track_target_msg.height = (_min_height+_max_height)/2;
				track_target_msg.confidence = 1;
			}
			else
			{
				_close_range = false;
			}

			for (int i = 0; i < tracks_msg->tracks.size(); i++)
			{
				opt_msgs::Track msg_track = tracks_msg->tracks[i];

				// No hay que revisar ids de personas que fueron detectadas cuando el target se vio.
				if (msg_track.id <= curr_max_id_)
				{
					continue;
				}

				// No analiza si la altura no es similar a la del target
				if (compareTargetHeight(msg_track.height) == false)
				{
					cout << "Alturas distintas: " << msg_track.height << " no pertenece a [" << _min_height << ", " << _max_height << "]" << endl;
					continue;
				}

				opt_msgs::DetectionArray det_arr = *detections_msg;
				Rect rect_target;
				if (findTrackRect(msg_track, det_arr, rect_target))
				{
					cv::Mat blob_im = cropImage(image,rect_target);
					cv::Mat mask = cropMask(depth, rect_target, msg_track.distance);
					vector<Histogram> hist_track;
					feat_ext.getFeaturesMasked(blob_im, hist_track, mask);

					double dist_target_track1 = feat_ext.compareArrayHist(hist_features_target_, hist_track);
					double dist_target_track2 = feat_ext.compareArrayHist(prev_hist_features_target_, hist_track);
					cout << "Dist_hist: " << dist_target_track1 << ",\t";
					cout << "Dist_prev: " << dist_target_track2 << endl;

					//if (abs(dist_target_track1) < target_dist_thr_ || abs(dist_target_track2) < target_dist_thr_)
					if (abs(dist_target_track2) < target_dist_thr_ && abs(dist_target_track1) < target_dist_thr_)
					{
						track_target_msg = msg_track;

						target_id_ = msg_track.id;
						prev_hist_features_target_ = hist_features_target_;
						hist_features_target_ = hist_track;

						flag_perdido_recien = false;

						feat_ext.printWeights();

						//cv::imshow("Reencontrado", blob_im);
						viewMaskedImage(blob_im, mask, "Positivo");

						cout << "\nRe-encontrado en ID: " << target_id_;
						cout << ", \tConfidence: " << min(abs(dist_target_track1), abs(dist_target_track2)) << endl;
						break;
					}


					// Busca en los old_features si existe una muestra antigua similar a un track
					bool detected = false;
					for (int i = 0; i < old_features.size(); i++)
					{
						double d = feat_ext.compareArrayHist(old_features.at(i), hist_track);
						if (abs(d) < target_dist_thr_)
						{
							track_target_msg = msg_track;

							target_id_ = msg_track.id;
							prev_hist_features_target_ = hist_features_target_;
							hist_features_target_ = hist_track;

							flag_perdido_recien = false;

							feat_ext.printWeights();

							//cv::imshow("Reencontrado", blob_im);
							viewMaskedImage(blob_im, mask, "Positivo");

							cout << "\nRe-encontrado en ID: " << target_id_ << " \t Muestra: " << i << endl;
							cout << "Confidencia: " << abs(d) << endl;
							detected = true;
							break;
						}
					}
					if (detected)
					{
						break;
					}

					// Si no hay muchos distractores, agregar
					if (hist_distractors_.size() < max_distractors_ || true)
					{
						if (abs(dist_target_track2) > 3*target_dist_thr_)
						{
							cout << "Added new distractor" << endl;
							//imshow("agrega", blob_im);
							hist_distractors_.push_back(hist_track);
							addDist(distractor_dists_, dist_target_track2, 80, &distractor_dists_idx_);
							updateFeatureWeights();
							updateDistanceThreshold();
						}
					}


				}
			}
		}

		// Indicar si esta muy cerca.
		if (_close_range)
		{
			char str[14];
			sprintf(str,"Close Target");
			putText(image, str, Point2f(100,100), FONT_HERSHEY_PLAIN, 2,  Scalar(0,0,255,255), 4);
		}

	}

	double aux_time = ros::Time::now().toSec();

	if (aux_time - time_ > 0.05) // 0.1 [s] => 10 [Hz], 0.05 [s] => 20 [Hz]
	{
		pub.publish(track_target_msg);
		time_ = aux_time;
	}

	//pub.publish(track_target_msg);

	if (activo)
	{		
		cv::imshow("Follower", image);
		cv::waitKey(30);	
	}
}


//bool Active(bender_srvs::Onoff::Request  &req, bender_srvs::Onoff::Response &res)
bool Active(opt_msgs::Onoff::Request  &req, opt_msgs::Onoff::Response &res)
{

    if(req.select == true){
		if (activo) {
			ROS_INFO_STREAM("Already turned on");
		} else {

			opt_msgs::Onoff srv;
			srv.request.select = true;
			
			if(client.call(srv))
			{
				activo = true;

				sync3.reset(new MySync(MySyncPolicy(10), 
						sub_image_, 
						sub_tracks_, 
						sub_detect_, 
						sub_disparity_));
		    
		    	sync3->registerCallback(boost::bind(&FollowerNode::callbackNoFeatCarac,
						this,
						_1,
						_2,
						_3, 
						_4));

		    	namedWindow("Follower"); // non-autosized
		    	namedWindow("Positivo");
				namedWindow("Negativo");
				startWindowThread();
				
				ROS_INFO_STREAM("Starting Follower. . . OK");
			}
			else {
				ROS_ERROR("Follower-Track service failed");
			}
		}
    }
    else{
	 if (activo) {

	 	opt_msgs::Onoff srv;
		srv.request.select = false;

		if(client.call(srv))
		{
			activo = false;

			// Set defaults values
			clearNode();

			sync3.reset(new MySync(MySyncPolicy(10), 
					sub_image_, 
					sub_tracks_, 
					sub_detect_, 
					sub_disparity_));

			/*destroyWindow("Follower");
			destroyWindow("Positivo");
			destroyWindow("Negativo");*/
			destroyAllWindows();
			//waitKey(1);

			ROS_INFO_STREAM(" Turning off . . . OK");
		}
		else {
			ROS_ERROR("Follower-Track service failed");
		}
	  } else {
		      ROS_INFO_STREAM("Already turned off");
	  }
    }
    return true;
}

~FollowerNode()
    {
    }

};



int main(int argc, char **argv)
{
	ros::init(argc, argv, "follower");

	ros::NodeHandle n("~");

	FollowerNode FN(n);

	

	/*
	n.param("target_dist_thr", target_dist_ref_, 0.15);
	target_dist_thr_ = target_dist_ref_;

	n.param("get_feature_only_start", feat_only_start, false);

	n.param("nro_only_start", nro_only_start, 5);

	int target_frames;
	n.param("target_frames", target_frames, 3);
	old_features.resize(target_frames);

	double training_factor;
	n.param("training_factor", training_factor, 0.125); // 1/8
	feat_ext.setTrainingFactor(training_factor);

	pub = n.advertise<opt_msgs::Track>("tracker/target",30);
	*/

	/*
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, opt_msgs::TrackArray, opt_msgs::DetectionArray, sensor_msgs::Image> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> sync2(MySyncPolicy(10), sub_image_, sub_tracks_, sub_detect_, sub_disparity_);
	sync2.registerCallback(boost::bind(&callbackNoFeatCarac, _1, _2, _3, _4));
	*/

	/*
	message_filters::Subscriber<sensor_msgs::Image> sub_image_(n, "/HaarDispAdaColorImage", 3);
	message_filters::Subscriber<opt_msgs::TrackArray> sub_tracks_(n, "/tracker/tracks_smoothed", 3);
	message_filters::Subscriber<opt_msgs::DetectionArray> sub_detect_(n, "/detector/detections", 3);
	message_filters::Subscriber<sensor_msgs::Image> sub_disparity_(n, "/depth_image", 3);

	
	sync3.reset(new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), 
		sub_image_, 
		sub_tracks_, 
		sub_detect_, 
		sub_disparity_));

    sync3->registerCallback(boost::bind(&callbackNoFeatCarac,
          _1,
          _2,
          _3, 
          _4));
    */
    


	ROS_INFO("Follower listo");

	/*namedWindow("Follower");
	namedWindow("Positivo");
	namedWindow("Negativo");
	moveWindow("Positivo", 700,20);
	moveWindow("Negativo", 900,20);
	moveWindow("Follower", 20, 20);*/



	ros::spin();

	FN.printFeatureWeights();


	return 0;
}
