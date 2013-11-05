#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::string;
using std::stringstream;
using std::vector;
using std::map;
using std::pair;
using cv::Mat;
using cv::Rect;
using cv::RotatedRect;
using cv::Scalar;
using cv::Vec3f;
using cv::Point;
using cv::VideoCapture;
using cv::imread;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;
using cv::circle;
using cv::ellipse;
using cv::rectangle;

#define TEXT_MIN_SIZE 12

#define PARAM_FIT_ELLIPSIS_SCALE_FOR_HOUGH_CIRCLE 1.50
#define PARAM_TEXT_MIN_PERCENTAGE_IN_SIGN 0.01

#define COLOR_HOUGH_CIRCLES_BGR Scalar(255,0,0)
#define COLOR_ELLIPSIS_BGR Scalar(0,255,0)
#define COLOR_IMAGE_ROI_BACKGROUND_HSV Scalar(0,0,255)
#define COLOR_TEXT_HSV Scalar(30, 255, 255)
#define COLOR_LABEL_BOX_HSV Scalar(45,255,255)
#define COLOR_LABEL_TEXT_HSV Scalar(45,255,255)

#define WINDOW_NAME_MAIN "0. Original image"
#define WINDOW_NAME_OPTIONS "Parameterization"
#define WINDOW_NAME_BILATERAL_FILTER "1. Bilateral filter"
#define WINDOW_NAME_BILATERAL_FILTER_OPTIONS "1.1. Bilateral filter options"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION "2. Global histogram equalization (not used)"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE "2. Histogram equalization CLAHE"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS "2.1. Histogram equalization CLAHE options"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS "3. Contrast and brightness and bilateral filtering"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS "3.1. Contrast and brightness options"
#define WINDOW_NAME_COLOR_SEGMENTATION "4. Color segmentation"
#define WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS "4.1. Color segmentation configuration"
#define WINDOW_NAME_SIGNAL_CANNY "5. Canny edge detector"
#define WINDOW_NAME_SIGNAL_CANNY_OPTIONS "5.1. Canny edge detector options"
#define WINDOW_NAME_SIGNAL_RECOGNITION "6. Signal circle and ellipse recognition"
#define WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS "6.1. Signal recognition options"
#define WINDOW_NAME_SIGNAL_TEXTS "7. Traffic signals texts"
#define WINDOW_NAME_SIGNAL_TEXTS_OPTIONS "7.1. Traffic signals texts options"
#define WINDOW_NAME_SIGNAL_ROI "8. Traffic signals ROIs"

#define TRACK_BAR_NAME_BI_FILTER_DIST "1Dist"
#define TRACK_BAR_NAME_BI_FILTER_COLOR_SIG "1Color Sig"
#define TRACK_BAR_NAME_BI_FILTER_SPACE_SIG "1Space Sig"
#define TRACK_BAR_NAME_CLAHE_CLIP "2Clip"
#define TRACK_BAR_NAME_CLAHE_TILE_X "2Tile X"
#define TRACK_BAR_NAME_CLAHE_TILE_Y "2Tile Y"
#define TRACK_BAR_NAME_CONTRAST "3Contr*10"
#define TRACK_BAR_NAME_BRIGHTNESS "3Brigh*10"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_HUE "4Min Hue"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_HUE "4Max Hue"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_SAT "4Min Sat"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_SAT "4Max Sat"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_VAL "4Min Val"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_VAL "4Max Val"
//#define TRACK_BAR_NAME_CANNY_LOWER_HYSTERESIS_THRESHOLD "5CLowHyst"
//#define TRACK_BAR_NAME_CANNY_HIGHER_HYSTERESIS_THRESHOLD "5CHighHyst"
//#define TRACK_BAR_NAME_CANNY_KERNEL_SIZE_SOBEL "5CApperSbl"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_DP "6Hough DP"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_DIST_CENTERS "6MinDCntr%"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_CANNY_HIGHER_THRESHOLD "6CnyHiThrs"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD "6AccumThrs"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_RADIUS "6MinRadiu%"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MAX_RADIUS "6MaxRadiu%"

#define	WINDOW_HEADER_HEIGHT 32
#define WINDOW_FRAME_THICKNESS 8
#define WINDOW_OPTIONS_WIDTH 350
#define WINDOW_OPTIONS_HIGHT 935
#define WINDOW_OPTIONS_TRACKBAR_HEIGHT 44
#define ESC_KEYCODE 27


class ImageAnalysis {
	public:
		ImageAnalysis();
		virtual ~ImageAnalysis();
				
		bool processImage(string path, bool useCVHighGUI = true);
		bool processImage(Mat& image, bool useCVHighGUI = true);

		void preprocessImage(Mat& image, bool useCVHighGUI = true);
		void histogramEqualization(Mat& image, bool use_CLAHE = true, bool useCVHighGUI = true);		

		Mat segmentImageByTrafficSignColor(Mat& preprocessedImage, bool useCVHighGUI = true);
		void recognizeTrafficSignsEllipsis(Mat& colorSegmentedImage, Mat& preprocessedImage, vector< pair<Rect, RotatedRect> >& outputRecognizedEllipsis, bool useCVHighGUI = true);
		void filterRecognizedTrafficSignCircles(const vector<Vec3f>& houghCircles, vector<Vec3f>& outputHoughCirclesFiltered);
		void flatClustersByMeanCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);
		void flatClustersByMedianCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);
		void flatClustersByMaxRadius(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);
		bool aggregateCircleIntoClusters(vector< vector<Vec3f> >& houghCirclesClusters, const Vec3f& centerToAdd);
		void retrieveEllipsisFromHoughCircles(const Mat& colorSegmentedImage, const vector<Vec3f>& houghCirclesFiltered, vector<pair<Rect, RotatedRect> >& outputTrafficSignEllipsis, bool useCVHighGUI = true);
		
		vector<int> segmentImageByTrafficSignText(Mat& preprocessedImage, vector< pair<Rect, RotatedRect> >& trafficSignEllipsis, bool useCVHighGUI = true);
		int recognizeTrafficSignText(Mat& preprocessedImage, Mat& textColorSegmentation, const Rect& ellipseBoundingRect, bool useCVHighGUI = true);

		bool updateImage();
		
		bool processVideo(string path, bool useCVHighGUI = true);
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);
		
		void drawTrafficSignLabel(string text, Mat& image, const Rect& signBoundingRect);

		void setupMainWindow();
		void setupResultsWindows(bool optionsOneWindow = false);
		bool outputResults();		

		pair<int, int> addHighGUIWindow(int column, int row, string windowName, int numberColumns = 4, int numberRows = 2, int xOffset = 0, int yOffset = 0);
		pair<int, int> addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber, int xOffset = 0, int yOffset = 0);

	private:
		vector<int> detectedSigns;
		Mat originalImage;
		Mat preprocessedImage;
		Mat processedImage;
		bool useCVHiGUI;
		bool windowsInitialized;

		int frameRate;
		int screenWidth;
		int screenHeight;

		int claehClipLimit;
		int claehTileXSize;
		int claehTileYSize;

		int bilateralFilterDistance;
		int bilateralFilterSigmaColor;
		int bilateralFilterSigmaSpace;

		int contrast;
		int brightness;

		int colorSegmentationLowerHue;
		int colorSegmentationUpperHue;
		int colorSegmentationLowerSaturation;
		int colorSegmentationUpperSaturation;
		int colorSegmentationLowerValue;				
		int colorSegmentationUpperValue;

		int textColorSegmentationLowerHue;
		int textColorSegmentationUpperHue;
		int textColorSegmentationLowerSaturation;
		int textColorSegmentationUpperSaturation;
		int textColorSegmentationLowerValue;				
		int textColorSegmentationUpperValue;

		int cannyLowerHysteresisThreshold;
		int cannyHigherHysteresisThreshold;
		int cannySobelOperatorKernelSize;

		int houghCirclesDP;
		int houghCirclesMinDistanceCenters;
		int houghCirclesCannyHigherThreshold;
		int houghCirclesAccumulatorThreshold;
		int houghCirclesMinRadius;
		int houghCirclesMaxRadius;
};
