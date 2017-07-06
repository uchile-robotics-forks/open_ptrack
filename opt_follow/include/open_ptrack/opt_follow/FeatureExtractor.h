#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "opencv2/opencv.hpp"
#include <vector>
//#include "image_get/funciones_comp.h"
//#include "image_get/RectFrame.h"

#ifndef FUNCIONES_COMP_H_
typedef std::vector<float> Histogram;
#endif

using namespace cv;
using namespace std;

class FeatureExtractor
{
private:

	double _w[];

	

public:

	Histogram _hist_h;
	Histogram _hist_s;
	Histogram _hist_v;
	Histogram _hist_edgeDensity;
	Histogram _hist_edgeDescriptor;
	Histogram _hist_cslbp;
	Histogram _hist_ddx;
	Histogram _hist_ddy;
	cv::Mat _curr_mask;

	FeatureExtractor(void);

	void edgeDensityImage(cv::Mat &input_gray, cv::Mat &output);

	bool hola(void);
	bool getHistLBP(Mat &im);

	void setTrainingFactor(double factor);

	void convertBGRtoHSV(cv::Mat &input, cv::Mat &output);
	void equalizeColorImage(const cv::Mat &input, cv::Mat &output);
	Histogram edgeDensity(Mat &image);
	void histHS(cv::Mat &input_bgr, Histogram &hist_hs);
	void histHSV(cv::Mat &input_bgr, Histogram &hist_h, Histogram &hist_s, Histogram &hist_v);
	Histogram edgeDescriptor(cv::Mat &input);
	void secondDerivative(cv::Mat &image, Histogram &hist_x, Histogram &hist_y);
	Histogram CSLBP(cv::Mat &input);

	void getFeatures(cv::Mat &image);
	void getFeatures(cv::Mat &image, vector<Histogram> &histograms);
	void getFeaturesMasked(cv::Mat &image, vector<Histogram> &histograms, cv::Mat &mask);

	double d_bhattacharyya(const Histogram &h1, const Histogram &h2);

	void printWeights(void);

	double compareArrayHist(vector<Histogram> &hs1, vector<Histogram> &h2);

	int getNroFeatures(void);

	void updateWeight(int idx, double offset);
	void normalizeWeights(void);

/*
	vector<float> histLBP;
	vector<float> histRGB;
	vector<float> histHS;
	Rect bb;

	static vector<float> getLBP(const cv::Mat &img,const cv::Rect &rect);

	bool computeFrom(Mat &im, Rect bbx,const char *nomImg);

	bool drawHist(vector<float> &hist, const char *nomHist);

	vector<float> getFeatures(const FeatureGenerator &other);
	static int getSize() {return 7;} // 8+1

	vector<float> getFeaturesRed(const FeatureGenerator &other);
	static int getSizeRed() {return 2;} //6// 8+1
*/
};

/*
bool is_valid(Rect bbx);
Rect repairBB(Rect bbx);
Point center_of_mass(Rect bbx);
*/

#endif // FEATURE_EXTRACTOR_H_
