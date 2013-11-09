#pragma once

//#define USE_TESSERACT 1

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef USE_TESSERACT
	#define _CRT_SECURE_NO_WARNINGS 1
	#include <tesseract/baseapi.h>
	#include <leptonica/allheaders.h>
#endif 

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
using cv::Point2f;
using cv::Size;
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


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <defines>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
#define PATH_IMAGES_DIGITS_TEMPLATES "./imgs/digits/"

#define TEXT_MIN_SIZE 12

#define PARAM_FIT_ELLIPSIS_SCALE_FOR_HOUGH_CIRCLE 1.50
#define PARAM_TEXT_MIN_AREA_PERCENTAGE_IN_SIGN 0.01
#define PARAM_TEXT_MIN_HEIGHT_PERCENTAGE_IN_SIGN 0.27

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
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS "3. Contrast, brightness and bilateral filtering (2nd pass)"
#define WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS "3.1. Contrast and brightness options"
#define WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION "4. Signal color segmentation with morphology operators"
#define WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS "4.1. Signal color segmentation options"
#define WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS "4.2. Signal morphology operators options"
//#define WINDOW_NAME_SIGNAL_CANNY "5. Canny edge detector"
//#define WINDOW_NAME_SIGNAL_CANNY_OPTIONS "5.1. Canny edge detector options"
#define WINDOW_NAME_SIGNAL_RECOGNITION "5. Signal circle and ellipse recognition"
#define WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS "5.1. Signal circle and ellipse recognition options"
#define WINDOW_NAME_TEXT_COLOR_SEGMENTATION "6. Text color segmentation with morphology operators"
#define WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS "6.1. Text color segmentation options"
#define WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS "6.2. Text morphology operators options"
#define WINDOW_NAME_SIGNAL_ROI "7. Traffic signals ROIs"
#define WINDOW_NAME_FEATURE_MATCHES "8. Digits features matches"
#define WINDOW_NAME_FEATURE_GOOD_MATCHES "8. Digits features good matches"
#define WINDOW_NAME_DIGITS_SKELETON(number) "Digit " << number << " skeleton"
#define WINDOW_NAME_DIGITS_FEATURE_POINTS(signNumber, digitPosition) "Sign " << signNumber << ", digit position " << digitPosition << ", feature points good matches"

#define TRACK_BAR_NAME_BI_FILTER_DIST "1Dist"
#define TRACK_BAR_NAME_BI_FILTER_COLOR_SIG "1Color Sig"
#define TRACK_BAR_NAME_BI_FILTER_SPACE_SIG "1Space Sig"
#define TRACK_BAR_NAME_CLAHE_CLIP "2Clip"
#define TRACK_BAR_NAME_CLAHE_TILE_X "2Tile X"
#define TRACK_BAR_NAME_CLAHE_TILE_Y "2Tile Y"
#define TRACK_BAR_NAME_CONTRAST "3Contr*10"
#define TRACK_BAR_NAME_BRIGHTNESS "3Brigh*10"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_HUE "4Min Hue"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_HUE "4Max Hue"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_SAT "4Min Sat"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_SAT "4Max Sat"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_VAL "4Min Val"
#define TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_VAL "4Max Val"
#define TRACK_BAR_NAME_SIGNAL_MORPH_OPERATOR "4MorphOper"
#define TRACK_BAR_NAME_SIGNAL_MORPH_KERNEL_SIZE_X "4MorKrnRdX"
#define TRACK_BAR_NAME_SIGNAL_MORPH_KERNEL_SIZE_Y "4MorKrnRdY"
#define TRACK_BAR_NAME_SIGNAL_MORPH_ITERATIONS "4MorphIter"
//#define TRACK_BAR_NAME_CANNY_LOWER_HYSTERESIS_THRESHOLD "5CLowHyst"
//#define TRACK_BAR_NAME_CANNY_HIGHER_HYSTERESIS_THRESHOLD "5CHighHyst"
//#define TRACK_BAR_NAME_CANNY_KERNEL_SIZE_SOBEL "5CApperSbl"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_DP "5Hough DP"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_DIST_CENTERS "5MinDCntr%"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_CANNY_HIGHER_THRESHOLD "5CnyHiThrs"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD "5AccumThrs"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_RADIUS "5MinRadiu%"
#define TRACK_BAR_NAME_HOUGH_CIRCLES_MAX_RADIUS "5MaxRadiu%"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_HUE "6Min Hue"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_HUE "6Max Hue"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_SAT "6Min Sat"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_SAT "6Max Sat"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_VAL "6Min Val"
#define TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_VAL "6Max Val"
#define TRACK_BAR_NAME_TEXT_MORPH_OPERATOR "6MorphOper"
#define TRACK_BAR_NAME_TEXT_MORPH_KERNEL_SIZE_X "6MorKrnRdX"
#define TRACK_BAR_NAME_TEXT_MORPH_KERNEL_SIZE_Y "6MorKrnRdY"
#define TRACK_BAR_NAME_TEXT_MORPH_ITERATIONS "6MorphIter"
#define TRACK_BAR_NAME_TEXT_SKELETONIZATION_KERNEL_SIZE_X "6SklKrnRdX"
#define TRACK_BAR_NAME_TEXT_SKELETONIZATION_KERNEL_SIZE_Y "6SklKrnRdY"
#define TRACK_BAR_NAME_TEXT_SKELETONIZATION_ITERATIONS "6SkelIter"
#define TRACK_BAR_NAME_TEXT_MIN_MATCH_PERCENTAGE "6MinMatchP"
#define TRACK_BAR_NAME_TEXT_DIGIT_RECOGNITION_METHOD "6DigRecMtd"
#define TRACK_BAR_NAME_TEXT_TEMPLATE_MATCH_METHOD "6TMatchMtd"

#define	WINDOW_HEADER_HEIGHT 32
#define WINDOW_FRAME_THICKNESS 8
#define WINDOW_OPTIONS_WIDTH 350
#define WINDOW_OPTIONS_HIGHT 935
#define WINDOW_DIGITS_HEIGHT 200
#define WINDOW_OPTIONS_TRACKBAR_HEIGHT 44

#define ESC_KEYCODE 27
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </defines>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
enum DigitRecognitonMethod {
	DIGIT_RECOGNITION_TEMPLATE_MATHING,
	DIGIT_RECOGNITION_FEATURE_DETECTION,
	DIGIT_RECOGNITION_TESSERACT
};

/// Image analysis class that detects speed limits signs and recognizes the speed limit number
class ImageAnalysis {
	public:
		
		/// Constructor with initialization of parameters with default value		 		 
		ImageAnalysis();
		
		/// ImageAnalysis destructor that performs cleanup of OpenCV HighGUI windows (in case they are used)		 
		virtual ~ImageAnalysis();
				

		/// Loads the digits image templates, applying a thresholding algorithm in order to have binary images for each number (0 to 9)		 		
		void loadDigitsTemplateImages();


		/*!
		 * \brief Processes the image from the specified path
		 * \param path Full path to image
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(string path, bool useCVHighGUI = true);


		/*!
		 * \brief Processes the image already loaded
		 * \param image Image loaded and ready to be processed
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return true if image was successfully processed
		 */
		bool processImage(Mat& image, bool useCVHighGUI = true);


		/*!
		 * \brief Preprocesses the image by applying bilateral filtering, histogram equalization, contrast and brightness correction and bilateral filtering again
		 * \param image Image to be preprocessed
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 */
		void preprocessImage(Mat& image, bool useCVHighGUI = true);


		/*!
		 * \brief Applies histogram equalization to the specified image
		 * \param image Image to equalize
		 * \param useCLAHE If true, uses the contrast limited adaptive histogram equalization (CLAHE)
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return 
		 */
		void histogramEqualization(Mat& image, bool useCLAHE = true, bool useCVHighGUI = true);		


		/*!
		 * \brief Identifies traffic sign position by using color segmentation, and reduces noise by using morphological operators (opening or closing)
		 * \param image Image to segment
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return Binary image segmented
		 */
		Mat segmentImageByTrafficSignColor(Mat& preprocessedImage, bool useCVHighGUI = true);


		/*!
		 * \brief Detects speed limit signs by using Hough Circles transform and then extracts the speed limit ellipse for each circle detected
		 * \param colorSegmentedImage Binary image processed with color segmentation to identify the position of traffic signs
		 * \param preprocessedImage Preprocessed image where the Hough circles and traffic sign ellipsis (along with its bounding box), will be drawn, in case useCVHighGUI is true
		 * \param outputRecognizedEllipsis Detected traffic sign ellipsis with their bounding box
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 */
		void recognizeTrafficSignsEllipsis(Mat& colorSegmentedImage, Mat& preprocessedImage, vector< pair<Rect, RotatedRect> >& outputRecognizedEllipsis, bool useCVHighGUI = true);


		/*!
		 * \brief Filters the detected houghCircles by grouping then in clusters
		 * 
		 * Each cluster is formed by circles which have their center inside of another circle.
		 * After creating the clusters, one circle for each cluster is computed based on the mean of their radius and their positions
		 * \param houghCircles Detected hough circles
		 * \param outputHoughCirclesFiltered Hough circles filtered
		 */
		void filterRecognizedTrafficSignCircles(const vector<Vec3f>& houghCircles, vector<Vec3f>& outputHoughCirclesFiltered);


		/*!
		 * \brief Flattens each circle cluster using the mean for its elements position and radius
		 * \param houghCirclesClusters Clusters to be flatten
		 * \param outputHoughCirclesFiltered Clusters flattened
		 */
		void flatClustersByMeanCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);
		
		
		/*!
		 * \brief Flattens each circle cluster using the circle with the median y position
		 * \param houghCirclesClusters Clusters to be flatten
		 * \param outputHoughCirclesFiltered Clusters flattened
		 */
		void flatClustersByMedianCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);


		/*!
		 * \brief Flattens each circle cluster using the circle with the maximum radius
		 * \param houghCirclesClusters Clusters to be flatten
		 * \param outputHoughCirclesFiltered Clusters flattened
		 */
		void flatClustersByMaxRadius(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered);


		/*!
		 * \brief Tries to aggregate the new circle with the provided cluster
		 * \param circle to be aggregated
		 * \return true if circle center is inside any of the cluster circles (and as such, was aggregated with the cluster)
		 */
		bool aggregateCircleIntoClusters(vector< vector<Vec3f> >& houghCirclesClusters, const Vec3f& centerToAdd);


		/*!
		 * \brief Retrieves the Hough Circles corresponding ellipsis, along with their bounding rectangles
		 * \param colorSegmentedImage Color segmented binary image with speed limit signs
		 * \param preprocessedImage Image where ellipsis and their bounding rectangles will be drawn if useCVHighGUI is true
		 * \param houghCirclesFiltered Hough circles used to find the speed limit sign ellipsis
		 * \param outputTrafficSignEllipsis Detected ellipsis for the speed limit signs
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 */
		void retrieveEllipsisFromHoughCircles(const Mat& colorSegmentedImage, Mat& preprocessedImage, const vector<Vec3f>& houghCirclesFiltered, vector<pair<Rect, RotatedRect> >& outputTrafficSignEllipsis, bool useCVHighGUI = true);
		

		/*!
		 * \brief Detects digits inside each ellipse roi in the preprocessed image
		 * \param preprocessedImage Image were the digits are going to be searched
		 * \param trafficSignEllipsis Ellipsis stating the regions of interest in the preprocessedImage were the digits might be
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return All speed limit numbers detected in the trafficSignEllipsis regions of interest
		 */
		vector<int> segmentImageByTrafficSignText(Mat& preprocessedImage, vector< pair<Rect, RotatedRect> >& trafficSignEllipsis, bool useCVHighGUI = true);


		/*!
		 * \brief Detects the speed limit sign number in the textColorSegmentation delimited by the ellipseBoundingRect roi 
		 * \param preprocessedImage Image were the digits contours and bounding rectangles are going to be drawn in case useCVHighGUI is true
		 * \param textColorSegmentation Color segmented binary image, with the regions that might be numbers
		 * \param ellipseBoundingRect Ellipse stating the region of interest were the speed limit sign is 
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \param currentSignBeingProcessed Auxiliary parameter to allow the creation of different window names for each digit (in case feature detection is used)
		 * \return Speed limit number detected or -1 in case of no valid number
		 */
		int recognizeTrafficSignText(Mat& preprocessedImage, Mat& textColorSegmentation, const Rect& ellipseBoundingRect, bool useCVHighGUI = true, size_t currentSignBeingProcessed = 0);


		/*!
		 * \brief Recognizes a digit in the textColorSegmentationDigitROI using one of the possible methods (template matching, feature detection or using the tesseract library)
		 * \param textColorSegmentationDigitROI Binary region of interest were there might be a digit
		 * \param feacturePointsGoodMatches If useCVHighGUI is true and the recognition method is DIGIT_RECOGNITION_FEATURE_DETECTION, then this image will contain the good matches between the image digit and the template digit
		 * \param useSkeletonization Flag to indicate to use skeletonization of the digits (if false, it will use erode instead)
		 * \param numberRecognitionMethod Method to use in the digit recognition process (can be DIGIT_RECOGNITION_TEMPLATE_MATHING, DIGIT_RECOGNITION_FEATURE_DETECTION, DIGIT_RECOGNITION_TESSERACT)
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return Digit recognized or -1 if none was detected
		 */
		int recognizeDigit(Mat& textColorSegmentationDigitROI, Mat& feacturePointsGoodMatches, bool useSkeletonization = false, int numberRecognitionMethod = DIGIT_RECOGNITION_FEATURE_DETECTION, bool useCVHighGUI = true);


		/*!
		 * \brief Computes the probability of the digitImageTemplate is present in the textColorSegmentationDigitROI
		 * \param textColorSegmentationDigitROI Image were the digit might be
		 * \param digitImageTemplate Template of the digit that is going to be searched
		 * \param feacturePointsGoodMatches Image were the good matches between the textColorSegmentationDigitROI and digitImageTemplate will be drawn if useCVHighGUI is true
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return Probability of the digitImageTemplate is present in the textColorSegmentationDigitROI
		 */
		float recognizeDigitWithFeatureMatching(Mat& textColorSegmentationDigitROI, Mat& digitImageTemplate, Mat& feacturePointsGoodMatches, bool useCVHighGUI = true);		
		
		
		/*!
		 * \brief Performs the skeletonization of the image using the specified kernel
		 * \param image Image to be processed
		 * \param kernel Kernel to use
		 * \param numberIterations Number of iterations to run the skeletonization
		 * \return 
		 */
		Mat textSkeletonization(Mat& image, Mat& kernel, int numberIterations);

		
		 /// Thins the templates to match the degree of skeletonization that will be applied to the image digits		 		 		 
		void skeletonizeTemplates();


#ifdef USE_TESSERACT
		/*!
		 * \brief Uses Tesseract library to recognize digits
		 * \param textColorSegmentationDigitROI Image with numbers to recognize
		 * \return Number recognized or -1 if none was found
		 */
		int recognizeDigitWithTesseract(Mat& textColorSegmentationDigitROI);
#endif

		/*!
		 * \brief Processes the image to reflect any internal parameter change		 
		 * \return True if processing finished successfully
		 */
		bool updateImage();
		

		/*!
		 * \brief Processes a video from a file, analyzing the presence of speed limit signs
		 * \param path Full path to video
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(string path, bool useCVHighGUI = true);

		/*!
		 * \brief Processes a video from a camera, analyzing the presence of speed limit signs
		 * \param cameraDeviceNumber Camera device number
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);


		/*!
		 * \brief Processes a video from a VideoCapture source, analyzing the presence of speed limit signs
		 * \param useCVHighGUI Optional parameter specifying if the results and the intermediate processing should be displayed using OpenCV HighGUI
		 * \return True if processing finished successfully
		 */
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);
		

		/*!
		 * \brief Draws a label in image in the top part of the signBoundingRect
		 * \param text Text to draw
		 * \param image Image where the text is going to be drawn
		 * \param signBoundingRect Rectangle with the region of interest were the text is going to be positioned inside the image
		 */
		void drawTrafficSignLabel(string text, Mat& image, const Rect& signBoundingRect);

		
		/// brief Setups the HighGUI window were the original image is going to be drawn		 		 
		void setupMainWindow();


		/*!
		 * \brief Setups the windows were the results will be presented
		 * \param optionsOneWindow Flag to indicate to group the trackbars in one window		 
		 */
		void setupResultsWindows(bool optionsOneWindow = false);			


		/*!
		 * \brief Adds a OpenCV HighGUI window, resized, aligned and positioned with the specified parameters
		 * \param column Window column where the window will be moved
		 * \param row Window row where the window will be moved
		 * \param windowName Window name
		 * \param numberColumns Number of window columns in which the screen is going to be divided
		 * \param numberRows Number of window rows in which the screen is going to be divided
		 * \param xOffset X offset to add to the original position
		 * \param yOffset Y offset to add to the original position
		 * \param windowWidth Override the computed window width
		 * \param windowHeight Override the computed window height
		 * \param imageWidth Override the computed image width
		 * \param imageHeight Override the computed image height
		 * \return Pair with he (x, y) position and the (width, height) of the window added 
		 */
		pair< pair<int, int>, pair<int, int> > addHighGUIWindow(int column, int row, string windowName, int numberColumns = 4, int numberRows = 2, int xOffset = 0, int yOffset = 0, int windowWidth = -1, int windowHeight = -1, int imageWidth = -1, int imageHeight = -1);


		/*!
		 * \brief Adds an OpenCV HighGUI trackbar, resized, aligned and positioned with the specified parameters
		 * \param windowName Window name
		 * \param numberTrackBars Number of trackbars that are going to be added to the window (to properly move the window that is going to be added)
		 * \param cumulativeTrackBarPosition Number of trackbars that are staked vertically in the windows above (to properly move the window that is going to be added)
		 * \param trackBarWindowNumber Number of trackbar windows that are going to be above (to properly move the window that is going to be added)
		 * \param xOffset Offset to adjust the x position
		 * \param yOffset Offset to adjust the y position
		 * \return Pair with he (x, y) position and the (width, height) of the window added 
		 */
		pair< pair<int, int>, pair<int, int> > addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber, int xOffset = 0, int yOffset = 0);


		// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Gets / sets>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<		
		int ScreenWidth() const;
		void ScreenWidth(int val);

		int ScreenHeight() const;
		void ScreenHeight(int val);	

		bool OptionsOneWindow() const;
		void OptionsOneWindow(bool val);
		// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Gets / sets>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	private:
		vector<int> detectedSigns;
		vector<Mat> digitsImagesTemplates;
		vector<Mat> digitsImagesTemplatesSkeletons;
		Mat originalImage;
		Mat preprocessedImage;
		Mat processedImage;
		bool useCVHiGUI;
		bool windowsInitialized;
		bool optionsOneWindow;
		
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

		int signalColorSegmentationLowerHue;
		int signalColorSegmentationUpperHue;
		int signalColorSegmentationLowerSaturation;
		int signalColorSegmentationUpperSaturation;
		int signalColorSegmentationLowerValue;				
		int signalColorSegmentationUpperValue;
		int signalColorSegmentationMorphType;
		int signalColorSegmentationMorphKernelSizeX;
		int signalColorSegmentationMorphKernelSizeY;
		int signalColorSegmentationMorphIterations;

		int textColorSegmentationLowerHue;
		int textColorSegmentationUpperHue;
		int textColorSegmentationLowerSaturation;
		int textColorSegmentationUpperSaturation;
		int textColorSegmentationLowerValue;				
		int textColorSegmentationUpperValue;
		int textColorSegmentationMorphType;
		int textColorSegmentationMorphKernelSizeX;
		int textColorSegmentationMorphKernelSizeY;
		int textColorSegmentationMorphIterations;

		int textMinMatchPercentage;		
		int digitRecognitionMethod;
		int textFeatureDetectionMaxDistancePercentageKeypoint;
		int textTemplateMatchMethod;

		int textSkeletonKernelPercentageX;
		int textSkeletonKernelPercentageY;
		int textSkeletonIterations;
		bool useSkeletonizationOnDigits;

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
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
