#include "open_ptrack/opt_follow/FeatureExtractor.h"

unsigned int lowThreshold_ = 50; // threshold canny para edgeDensity
const int _nFeat = 9;
double factor_;

FeatureExtractor::FeatureExtractor ()
{
	_w[_nFeat];
	for (int i = 0; i < _nFeat; i++)
	{
		_w[i] = 1.0/_nFeat;
	}

	factor_ = 1/8;
}

void FeatureExtractor::init(void)
{
	for (int i = 0; i < _nFeat; i++)
	{
		_w[i] = 1.0/_nFeat;
	}

	factor_ = 1/8;
}

bool FeatureExtractor::hola(void)
{
	cout << "hola Feat Ext" << endl;
	return true;
}


int absMaxIndex(float *array, int size)
{
	float maxValue = 0;
	int index = -1;

	for (int i = 0; i < size; i++)
	{
		if (maxValue < abs(array[i]))
		{
			maxValue = abs(array[i]);
			index = i;
		}

	}

	return index;
}


void FeatureExtractor::setTrainingFactor(double factor)
{
	factor_ = factor;
}


// Equalize BGR image
void FeatureExtractor::equalizeColorImage(const cv::Mat &input, cv::Mat &output)
{
	output = input;
  cv::cvtColor(output, output, CV_BGR2GRAY);
	cv::equalizeHist(output, output);
}


// BGR8 to HSV
void FeatureExtractor::convertBGRtoHSV(cv::Mat &input, cv::Mat &output)
{
	cv::cvtColor(input, output, CV_BGR2HSV);
}


bool rectHead(cv::Mat &mask, cv::Rect &head_rect)
{
	if (mask.empty())
		return false;
	
	int width = mask.cols;
	int height = mask.rows;

	int head_height = height/3;
	head_rect.x = 0;
	head_rect.y = 0;
	head_rect.width = width;
	head_rect.height = head_height;

	return true;
}

void FeatureExtractor::histHS(cv::Mat &input_bgr, Histogram &hist_hs)
{
	cv::Mat hsv;
	cv::cvtColor(input_bgr, hsv, CV_BGR2HSV);

	int hbins = 18, sbins = 16;
	int histSize[] = {hbins, sbins};
	float hrange[] = {0, 180};
	float srange[] = {0, 256};
	const float* ranges[] = {hrange, srange};
	int channels[] = {0, 1};

	cv::Mat hist;
	cv::Mat mask;
	if (_curr_mask.empty())
		mask = cv::Mat();
	else
		mask = _curr_mask;

	cv::calcHist(&hsv, 1, channels, mask, hist, 2, histSize, ranges, true, false);

	hist_hs.resize(hist.rows*hist.cols);
	float sum = 0;
	for (int i = 0; i < hist_hs.size(); i++)
	{
		hist_hs[i] = hist.at<float>(i);
		sum += hist_hs[i];
	}

	float factor = 1.0;
	if (sum != 0)
		float factor = 1.0/sum;
	for (int i = 0; i < hist_hs.size(); i++)
	{
		hist_hs[i] = hist_hs[i] * factor;
	}
}

// Histogramas H,S,V
void FeatureExtractor::histHSV(cv::Mat &input_bgr, Histogram &hist_h, Histogram &hist_s, Histogram &hist_v)
{
	cv::Mat hsv;
	cv::cvtColor(input_bgr, hsv, CV_BGR2HSV);
	cv::Mat hsv_split[3];
	cv::split(hsv, hsv_split);

	int hbins = 8, sbins = 8, vbins = 8;
	//int hbins = 18, sbins = 60/3, vbins = 60/3;
	//int hbins = 50/3, sbins = 60/3, vbins = 60/3;
	float hrange[] = { 0, 180 };
  	float srange[] = { 0, 256 };
	float vrange[] = { 0, 256 };

	const float* hranges[] = {hrange};
	const float* sranges[] = {srange};
	const float* vranges[] = {vrange};

	cv::Mat hist0, hist1, hist2;

	cv::Mat mask;
	if (_curr_mask.empty())
		mask = cv::Mat();
	else
		mask = _curr_mask;

	cv::calcHist(&hsv_split[0], 1, 0, mask, hist0, 1, &hbins, hranges, true, false);
	cv::calcHist(&hsv_split[1], 1, 0, mask, hist1, 1, &sbins, sranges, true, false);
	cv::calcHist(&hsv_split[2], 1, 0, mask, hist2, 1, &vbins, vranges, true, false);

	cv::normalize(hist0, hist0, 1, 0, NORM_L1, -1, Mat());
	cv::normalize(hist1, hist1, 1, 0, NORM_L1, -1, Mat());
	cv::normalize(hist2, hist2, 1, 0, NORM_L1, -1, Mat());

	hist_h.resize(hist0.cols*hist0.rows);
	hist_s.resize(hist1.cols*hist1.rows);
	hist_v.resize(hist2.cols*hist2.rows);

	for (int i=0; i < hist0.cols*hist0.rows; i++)
		hist_h[i] = hist0.at<float>(i);
	for (int i=0; i < hist1.cols*hist1.rows; i++)
		hist_s[i] = hist1.at<float>(i);
	for (int i=0; i < hist2.cols*hist2.rows; i++)
		hist_v[i] = hist2.at<float>(i);
}


// Edge Density descriptor
void FeatureExtractor::edgeDensityImage(cv::Mat &input_gray, cv::Mat &output)
{
	unsigned int lowThreshold = lowThreshold_;
	int kernel_size = 3;
	int ratio = 3;	

	// Canny edge detector
	cv::Mat cannyImg;
	//Mat input = input_gray; equalizeColorImage(input, cannyImg);
	cv::blur(input_gray, cannyImg, Size(3,3) );
	cv::Canny(cannyImg, cannyImg, lowThreshold, lowThreshold*ratio, kernel_size );

	// Binarize image in {0,1} values
	for (int i = 0; i < cannyImg.rows; i++)
	{
		for (int j = 0; j < cannyImg.cols; j++)
		{
			if (cannyImg.at<uchar>(i,j) > 0)
			{
				cannyImg.at<uchar>(i,j) = 1;
			}
		}
	}

	// Suma los bordes en una ventana de 9x9
	int rows = cannyImg.rows-8;
	int cols = cannyImg.cols-8;
	output = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (int k1 = 0; k1 < 9; k1++)
	{
		for (int k2 = 0; k2 < 9; k2++)
		{
			cv::add(output, cannyImg(Rect(k1,k2,cols,rows)), output);

		}
	}

	// Umbral para valores altos (heuristica): 31
	for (int i = 0; i < output.rows; i++)
	{
		for (int j = 0; j < output.cols; j++)
		{
			if (output.at<uchar>(i,j) > 31)
			{
				output.at<uchar>(i,j) = 31;
			}
		}
	}
}


// Histograma Edge density descriptor
Histogram FeatureExtractor::edgeDensity(Mat &image_gray)
{
	cv::Mat edge_density;
	edgeDensityImage(image_gray, edge_density);
	
	cv::Mat hist;
	int bins = 16; // bins of edgeDensity {0,1,2,...,31} //int bins = 32;
	float range[] = {0, 32};
	const float* ranges[] = {range};

	cv::Mat mask;
	if (_curr_mask.empty())
		mask = cv::Mat();
	else
	{
		int x_off = (_curr_mask.cols - edge_density.cols)/2;
		int y_off = (_curr_mask.rows - edge_density.rows)/2;
		mask = _curr_mask(Rect(x_off, y_off, edge_density.cols, edge_density.rows)).clone();//= _curr_mask(Rect());
	}

	cv::calcHist(&edge_density, 1, 0, mask, hist, 1, &bins, ranges, true, false);
	cv::normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());

	std::vector<float> histogram(hist.rows, 0); // hist.rows = 32
	for (int i = 0; i < histogram.size(); i++)
	{
		histogram[i] = hist.at<float>(i);
	}

	return histogram;
}


// Histograma Edge Descriptor basado en MPEG-7
Histogram FeatureExtractor::edgeDescriptor(cv::Mat &img)
{
	float x = sqrt(img.rows * img.cols / 1100);
	int block_size = floor(x/2) * 2;
	int block_width = floor(img.cols / block_size);
	int block_height = floor(img.rows / block_size);

	std::vector<float> hist(5, 0);
	float num_edges = 0;

	for (int i = 0; i < img.rows - block_height + 1; i=i+block_height)
	{
		for (int j = 0; j < img.cols - block_width + 1; j=j+block_width)
		{
			int w = floor(block_width/2);
			int h = floor(block_height/2);
			cv::Rect rect0(j,i,w,h);
			cv::Rect rect1(j,i+h,w,h);
			cv::Rect rect2(j+w,i,w,h);
			cv::Rect rect3(j+w,i+h,w,h);
			
			cv::Scalar tempVal;
			tempVal = cv::mean(img(rect0));
			float val0 = tempVal.val[0];
			tempVal = cv::mean(img(rect1));
			float val1 = tempVal.val[0];
			tempVal = cv::mean(img(rect2));
			float val2 = tempVal.val[0];
			tempVal = cv::mean(img(rect3));
			float val3 = tempVal.val[0];

			float ver_edge = val0 - val1 + val2 - val3;
			float hor_edge = val0 + val1 - val2 - val3;
			float dia45_edge = sqrt(2) * (val0 - val2);
			float dia135_edge = sqrt(2) * (-val3 + val1);
			float nond_edge = 2*val0 - 2*val1 - 2*val2 + 2*val3;

			float edges[] = {ver_edge, hor_edge, dia45_edge, dia135_edge, nond_edge};
			int max_idx = absMaxIndex(edges, 5);

			if (max_idx > -1 && edges[max_idx] > 11) // 11 = threshold_edge
			{
				hist[max_idx]++;
				num_edges++;
			}
		}
	}
	//cout << "histograma:" << hist[0] << " " << hist[1] << " " << hist[4] << endl;
	for (int i = 0; i < 5; i++)
	{
		if (num_edges == 0)
			hist[i] = 1.0/5;
		else
			hist[i] = hist[i]/num_edges;
	}

	return hist;
}


// Histograma CSLBP
Histogram FeatureExtractor::CSLBP(cv::Mat &input_gray)
{
	cv::Mat im_lbp = cv::Mat(input_gray.rows-2, input_gray.cols-2,CV_8UC1,cv::Scalar(0));

	// we lose a border of 1 pixel
	for (int i = 1; i < input_gray.rows-1; ++i) 
	{
		uchar *lbp_row = im_lbp.ptr<uchar>(i-1);    // current lbp row
		const uchar *pre__ = input_gray.ptr<uchar>(i-1); // row i-1
		const uchar *curr_ = input_gray.ptr<uchar>(i);   // row i
		const uchar *post_ = input_gray.ptr<uchar>(i+1); // row i+1
		for (int j = 1; j < input_gray.cols-1; ++j)
		{
			// compute typical lbp as usual
			uchar lbp = 0;
			uchar mask = 1 << 7;
			mask >>= 3;
			mask >>= 1; lbp = (lbp & ~mask) | ( (post_[j-1] >= pre__[j+1]) << 3);
			mask >>= 1; lbp = (lbp & ~mask) | ( (post_[j+0] >= pre__[j+0]) << 2);
			mask >>= 1; lbp = (lbp & ~mask) | ( (post_[j+1] >= pre__[j-1]) << 1);
			mask >>= 1; lbp = (lbp & ~mask) | ( (curr_[j+1] >= curr_[j-1]) << 0);

			// save lbp byte
			lbp_row[j-1] = lbp;
		}
	}

	// Histogram
	int bins = 8; // bins of csLBP //int bins = 16;
	float range[] = {0, 16};
	const float* ranges[] = {range};

	int w = floor(im_lbp.cols/4);
	int h = floor(im_lbp.rows/4);

	cv::Mat hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8,
			hist9, hist10, hist11, hist12, hist13, hist14, hist15, hist16;
	cv::Mat tempLBP;
	tempLBP = im_lbp(Rect(0*w,0*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist1, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(0*w,1*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist2, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(0*w,2*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist3, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(0*w,3*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist4, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(1*w,0*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist5, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(1*w,1*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist6, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(1*w,2*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist7, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(1*w,3*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist8, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(2*w,0*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist9, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(2*w,1*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist10, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(2*w,2*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist11, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(2*w,3*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist12, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(3*w,0*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist13, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(3*w,1*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist14, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(3*w,2*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist15, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(3*w,3*h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist16, 1, &bins, ranges, true, false);

	std::vector<float> hist(8*16, 0); //std::vector<float> hist(256, 0);
	for (int i = 0; i < floor(hist.size()/16); i++)
	{
		hist[i+0*hist.size()/16] = hist1.at<float>(i);
		hist[i+1*hist.size()/16] = hist2.at<float>(i);
		hist[i+2*hist.size()/16] = hist3.at<float>(i);
		hist[i+3*hist.size()/16] = hist4.at<float>(i);
		hist[i+4*hist.size()/16] = hist5.at<float>(i);
		hist[i+5*hist.size()/16] = hist6.at<float>(i);
		hist[i+6*hist.size()/16] = hist7.at<float>(i);
		hist[i+7*hist.size()/16] = hist8.at<float>(i);
		hist[i+8*hist.size()/16] = hist9.at<float>(i);
		hist[i+9*hist.size()/16] = hist10.at<float>(i);
		hist[i+10*hist.size()/16] = hist11.at<float>(i);
		hist[i+11*hist.size()/16] = hist12.at<float>(i);
		hist[i+12*hist.size()/16] = hist13.at<float>(i);
		hist[i+13*hist.size()/16] = hist14.at<float>(i);
		hist[i+14*hist.size()/16] = hist15.at<float>(i);
		hist[i+15*hist.size()/16] = hist16.at<float>(i);
	}

	/*
	int w = floor(im_lbp.cols/2);
	int h = floor(im_lbp.rows/2);

	cv::Mat hist1, hist2, hist3, hist4;
	cv::Mat tempLBP;
	tempLBP = im_lbp(Rect(0,0,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist1, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(w,0,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist2, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(0,h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist3, 1, &bins, ranges, true, false);
	tempLBP = im_lbp(Rect(w,h,w,h));
	cv::calcHist(&tempLBP, 1, 0, Mat(), hist4, 1, &bins, ranges, true, false);

	std::vector<float> hist(256, 0);
	for (int i = 0; i < floor(hist.size()/4); i++)
	{
		hist[i+0*hist.size()/4] = hist1.at<uchar>(i);
		hist[i+1*hist.size()/4] = hist2.at<uchar>(i);
		hist[i+2*hist.size()/4] = hist3.at<uchar>(i);
		hist[i+3*hist.size()/4] = hist4.at<uchar>(i);
	}
	*/

	cv::normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());

	for (int i = 0; i < hist.size(); i++)
	{
		if (hist[i] > 0.2)
		{
			hist[i] = 0.2;
		}
	}

	cv::normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());

	return hist;
}


// Segundas derivadas en X y en Y
void secondDerivativeImage(cv::Mat &input_gray, cv::Mat &outx, cv::Mat &outy)
{
	cv::Mat blured_img;	
	cv::blur(input_gray, blured_img, Size(5,5) );
	
	//cv::Mat filter_ddx = (Mat_<double>(3,3) << 0, 0, 0, 4, -8, 4, 0, 0, 0);
	//cv::Mat filter_ddy = (Mat_<double>(3,3) << 0, 4, 0, 0, -8, 0, 0, 4, 0);

	cv::Mat filter_ddx = (Mat_<float>(3,3) << 0, 0, 0, 0.25, -0.5, 0.25, 0, 0, 0);
	cv::Mat filter_ddy = (Mat_<float>(3,3) << 0, 0.25, 0, 0, -0.5, 0, 0, 0.25, 0);

	filter2D(blured_img, outx, CV_32F , filter_ddx, Point(-1, -1), 0, BORDER_DEFAULT);
	filter2D(blured_img, outy, CV_32F , filter_ddy, Point(-1, -1), 0, BORDER_DEFAULT);

}


void FeatureExtractor::secondDerivative(cv::Mat &image_gray, Histogram &hist_x, Histogram &hist_y)
{	
	cv::Mat ddx, ddy;
	secondDerivativeImage(image_gray, ddx, ddy);
	
	//int bins = 64; // bins of secondDerivatives
	//float range[] = {-6*255, 6*255+1};
	int bins = 32; // bins of secondDerivatives
	float range[] = {-255*0.05, 255*0.05+1};
	const float* ranges[] = {range};

	cv::Mat ddx_hist, ddy_hist;	

	cv::Mat mask;
	if (_curr_mask.empty())
		mask = cv::Mat();
	else
		mask = _curr_mask;

	cv::calcHist(&ddx, 1, 0, mask, ddx_hist, 1, &bins, ranges, true, false);
	cv::calcHist(&ddy, 1, 0, mask, ddy_hist, 1, &bins, ranges, true, false);
	cv::normalize(ddx_hist, ddx_hist, 1, 0, NORM_L1, -1, Mat());
	cv::normalize(ddy_hist, ddy_hist, 1, 0, NORM_L1, -1, Mat());
	//cout << "ddx: " << ddx_hist << endl;	

	hist_x.resize(bins);
	for (int i = 0; i < hist_x.size(); i++)
	{
		hist_x[i] = ddx_hist.at<float>(i);
	}

	hist_y.resize(bins);
	for (int i = 0; i < hist_y.size(); i++)
	{
		hist_y[i] = ddy_hist.at<float>(i);
	}
}


void FeatureExtractor::getFeatures(cv::Mat &image)
{
	cv::Mat gray;
	equalizeColorImage(image, gray);

	histHSV(image, _hist_h, _hist_s, _hist_v);		// H, S, V
	secondDerivative(gray, _hist_ddx, _hist_ddy);	// ddX, ddY
	_hist_edgeDensity = edgeDensity(gray);				// Edge Density
	_hist_edgeDescriptor = edgeDescriptor(gray);	// Edge Descriptor
	_hist_cslbp = CSLBP(gray);										// CSLBP
}


void FeatureExtractor::getFeatures(cv::Mat &image, vector<Histogram> &histograms)
{
	histograms.clear();
	histograms.resize(_nFeat);
	//cv::imshow("hola2", image);

	//En teoria nunca entra acá, porque se verifica previamente.
	if (image.empty())
		return;

	cv::Mat gray;
	equalizeColorImage(image, gray);

	//Histogram trash_hist;
	//cout << "init histHSV:" << endl;

	Histogram hist1, hist2;
	histHS(image, histograms.at(0)); //histograma HS
	histHSV(image, hist1, hist2, histograms.at(1));
	histograms.at(2) = (CSLBP(gray));

	//histHSV(image, histograms.at(0), histograms.at(1), histograms.at(2));//cout << "init secondDerivative:" << endl;
	secondDerivative(gray, histograms.at(3), histograms.at(4));//cout << "init edgeDensity:" << endl;
	histograms.at(5) = (edgeDensity(gray));//cout << "init edgeDescriptor:" << endl;
	histograms.at(6) = (edgeDescriptor(gray));//cout << "init CSLBP:" << endl;
	//histograms.at(7) = (CSLBP(gray));

	Rect r_head;
	bool head_detected = rectHead(_curr_mask, r_head);
	if (head_detected)
	{
		cv::Mat aux_mask = _curr_mask.clone();
		_curr_mask = _curr_mask(r_head).clone();

		cv::Mat aux_image = image(r_head).clone();
		cv::Mat aux_gray = gray(r_head).clone();

		histHS(aux_image, histograms.at(7));
		histograms.at(8) = (CSLBP(aux_gray));
		
		_curr_mask = aux_mask;
	}

	
}

void FeatureExtractor::getFeaturesMasked(cv::Mat &image, vector<Histogram> &histograms, cv::Mat &mask)
{
	_curr_mask = mask.clone();
	FeatureExtractor::getFeatures(image, histograms);
	_curr_mask.release();
}


bool areComparable(const Histogram &h1, const Histogram &h2)
{
	if (h1.size() == 0 || h2.size() == 0 || h1.size() != h2.size())
	{
		std::cout << "Histogramas incompatibles" << std::endl;
		return false;
	}
	return true;
}


double d(const Histogram& h1, const Histogram& h2) {

	double dist = 0;
    if (!areComparable(h1, h2))
    {	
    	return -1;
    }

    float sum21=0, sum22=0;

    Histogram::const_iterator it1, it2;

    for (it1 = h1.begin(),it2 = h2.begin(); it1 != h1.end(); it1++, it2++) {
    	sum21 += (*it1)*(*it1);
    	sum22 += (*it2)*(*it2);
    }

    float norm1, norm2;
    norm1 = sqrt(sum21);
    norm2 = sqrt(sum22);

    for (it1 = h1.begin(),it2 = h2.begin(); it1 != h1.end(); it1++, it2++) {
    	double w1 = (*it1) / norm1;
    	double w2 = (*it2) / norm2;
        dist += (w1 - w2)*(w1 - w2);
    }
    dist = sqrt(dist);

    return dist;
}


double FeatureExtractor::d_bhattacharyya(const Histogram &h1, const Histogram &h2)
{
	if (!areComparable(h1, h2))
    {
    	return -1;
    }

	return compareHist(h1, h2, CV_COMP_BHATTACHARYYA);
}


void FeatureExtractor::printWeights(void)
{
	cout << "Pesos: ";
	for (int i = 0; i < _nFeat; i++)
	{
		cout << _w[i] << "; ";
	}
	cout << endl;
}


double FeatureExtractor::compareArrayHist(vector<Histogram> &hs1, vector<Histogram> &hs2)
{
	if (hs1.size() != hs2.size())
	{
		std::cout << "Arreglos de Histogramas incompatibles" << std::endl;
		return -1;
	}

	double dist = 0;

	for (int i = 0; i < _nFeat; i++)
	{
		Histogram h1 = hs1.at(i);
		Histogram h2 = hs2.at(i);

		double d = d_bhattacharyya(h1, h2);
		if (d == -1)
		{
			return -1;
		}

		dist += _w[i] * d;
	}

	return dist;

}


int FeatureExtractor::getNroFeatures()
{
	return _nFeat;
}


void FeatureExtractor::updateWeight(int idx, double offset)
{
	_w[idx] = _w[idx] + offset*factor_;
	
	/*if (offset < 0.25)
	{
		_w[idx] = 0;
	}
	else
	{
		_w[idx] = 0.5*_w[idx] + offset;
	}*/
	
}


void FeatureExtractor::normalizeWeights()
{
	double sum = 0;

	for (int i = 0; i < _nFeat; i++)
	{
		if (_w[i] < 0)
		{
			_w[i] = 0;
		}

		sum += _w[i];
	}

	for (int i = 0; i < _nFeat; i++)
	{
		if (sum == 0)
		{
			_w[i] = 1.0/_nFeat;
		}
		else
		{
			_w[i] = _w[i] / sum;
		}
		
	}
}
