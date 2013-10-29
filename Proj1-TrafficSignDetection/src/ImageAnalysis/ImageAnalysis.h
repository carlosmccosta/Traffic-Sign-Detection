#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::Scalar;
using cv::imread;
using cv::VideoCapture;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;
using cv::resizeWindow;


#define WINDOW_NAME_MAIN "0. Original image"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION "1. Histogram equalization"
#define WINDOW_NAME_BILATERAL_FILTER "2. Bilateral filter"
#define WINDOW_NAME_BILATERAL_FILTER_OPTIONS "2.1. Bilateral filter options"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS "3. Contrast and brightness"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS "3.1. Contrast and brightness options"
#define WINDOW_NAME_COLOR_SEGMENTATION "4. Color segmentation"
#define WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS "4.1. Color segmentation configuration"


#define	WINDOW_HEADER_HEIGHT 32
#define WINDOW_FRAME_THICKNESS 8
#define ESC_KEYCODE 27

class ImageAnalysis {
	private:
		vector<string> detectedSigns;
		Mat originalImage;
		bool useCVHiGUI;
		bool windowsInitialized;

		int frameRate;
		int screenWidth;
		int screenHeight;

		int bilateralFilterDistance;
		int bilateralFilterSigmaColor;
		int bilateralFilterSigmaSpace;

		int contrast;
		int breightness;

		int colorSegmentationLowerHue;
		int colorSegmentationUpperHue;
		int colorSegmentationLowerSaturation;
		int colorSegmentationUpperSaturation;
		int colorSegmentationLowerValue;				
		int colorSegmentationUpperValue;


	public:
		ImageAnalysis();
		virtual ~ImageAnalysis();
		
		bool updateImage();
		bool processImage(string path, bool useCVHighGUI = true);
		bool processImage(Mat& image, bool useCVHighGUI = true);	
		
		bool processVideo(string path, bool useCVHighGUI = true);
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);
		
		void setupMainWindow();
		void setupResultsWindows();
		bool outputResults();


		void preprocessImage(Mat& image, bool useCVHighGUI = true);
		void segmentImage(Mat& image, bool useCVHighGUI = true);
		void recognizeTrafficSigns(Mat& image, bool useCVHighGUI = true);

		void addWindow(int column, int row, string name, int numberColumns = 4, int numberRows = 2);

};

