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
#define WINDOW_NAME_OPTIONS "Parameterization"
#define WINDOW_NAME_BILATERAL_FILTER "1. Bilateral filter"
#define WINDOW_NAME_BILATERAL_FILTER_OPTIONS "1.1. Bilateral filter options"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION "2a. Global histogram equalization (not used)"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE "2b. Histogram equalization CLAHE"
#define WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS "2.1. Histogram equalization CLAHE options"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS "3. Contrast and brightness"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS "3.1. Contrast and brightness options"
#define WINDOW_NAME_COLOR_SEGMENTATION "4. Color segmentation"
#define WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS "4.1. Color segmentation configuration"
#define WINDOW_NAME_SIGNAL_RECOGNITION "5. Signal recognition"

#define TRACK_BAR_NAME_CLAHE_CLIP "Clip"
#define TRACK_BAR_NAME_CLAHE_TILE_X "Tile X"
#define TRACK_BAR_NAME_CLAHE_TILE_Y "Tile Y"
#define TRACK_BAR_NAME_BI_FILTER_DIST "Dist"
#define TRACK_BAR_NAME_BI_FILTER_COLOR_SIG "Color Sig"
#define TRACK_BAR_NAME_BI_FILTER_SPACE_SIG "Space Sig"
#define TRACK_BAR_NAME_CONTRAST "Contr * 10"
#define TRACK_BAR_NAME_BRIGHTNESS "Brigh * 10"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_HUE "Min Hue"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_HUE "Max Hue"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_SAT "Min Sat"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_SAT "Max Sat"
#define TRACK_BAR_NAME_COLOR_SEG_MIN_VAL "Min Val"
#define TRACK_BAR_NAME_COLOR_SEG_MAX_VAL "Max Val"

#define	WINDOW_HEADER_HEIGHT 32
#define WINDOW_FRAME_THICKNESS 8
#define WINDOW_OPTIONS_WIDTH 350
#define WINDOW_OPTIONS_HIGHT 666
#define WINDOW_OPTIONS_TRACKBAR_HEIGHT 45
#define ESC_KEYCODE 27


class ImageAnalysis {
	public:
		ImageAnalysis();
		virtual ~ImageAnalysis();
				
		bool processImage(string path, bool useCVHighGUI = true);
		bool processImage(Mat& image, bool useCVHighGUI = true);

		void preprocessImage(Mat& image, bool useCVHighGUI = true);
		void histogramEqualization(Mat& image, bool use_CLAHE = true, bool useCVHighGUI = true);		

		void segmentImage(Mat& image, bool useCVHighGUI = true);
		void recognizeTrafficSigns(Mat& image, Mat colorSegmentedImage, bool useCVHighGUI = true);

		bool updateImage();
		
		bool processVideo(string path, bool useCVHighGUI = true);
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);
		
		void setupMainWindow();
		void setupResultsWindows(bool optionsOneWindow = false);
		bool outputResults();		

		void addHighGUIWindow(int column, int row, string windowName, int numberColumns = 3, int numberRows = 2);
		void addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber);


	private:
		vector<string> detectedSigns;
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
};
