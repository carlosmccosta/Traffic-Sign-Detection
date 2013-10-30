#include "ImageAnalysis.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageAnalysis::ImageAnalysis() :
	useCVHiGUI(true), windowsInitialized(false),
	frameRate(30), screenWidth(1920), screenHeight(1080),
	claehClipLimit(2), claehTileXSize(2), claehTileYSize(2),
	bilateralFilterDistance(9), bilateralFilterSigmaColor(50), bilateralFilterSigmaSpace(10),
	contrast(12), brightness(50),
	colorSegmentationLowerHue(145), colorSegmentationUpperHue(10),
	colorSegmentationLowerSaturation(56), colorSegmentationUpperSaturation(255),
	colorSegmentationLowerValue(64), colorSegmentationUpperValue(255) {};


ImageAnalysis::~ImageAnalysis() {
	if (useCVHiGUI) {
		cv::destroyAllWindows();
	}
}


bool ImageAnalysis::processImage(string path, bool useCVHighGUI) {		
	Mat imageToProcess;
	if (path != "") {
		try {
			imageToProcess = imread(path, CV_LOAD_IMAGE_COLOR);	
		} catch (...) {
			return false;
		}			

		if (!imageToProcess.data) {
			return false;
		}
	} else {		
		return false;
	}

	useCVHiGUI = useCVHighGUI;
	windowsInitialized = false;
	bool status = processImage(imageToProcess, useCVHighGUI);	

	while(waitKey(0) != ESC_KEYCODE) {}
	if (useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return status;
}


bool ImageAnalysis::processImage(Mat& image, bool useCVHighGUI) {
	detectedSigns.clear();
	originalImage = image.clone();
	processedImage = image;
	useCVHiGUI = useCVHighGUI;
	
	if (useCVHighGUI) {		
		if (!windowsInitialized) {
			setupMainWindow();
			setupResultsWindows();
			windowsInitialized = true;
		}		

		imshow(WINDOW_NAME_MAIN, originalImage);		
	}

	preprocessedImage = image.clone();
	preprocessImage(preprocessedImage, useCVHighGUI);

	Mat imageColorSegmented = preprocessedImage.clone();
	segmentImage(preprocessedImage, useCVHighGUI);
	

	if (useCVHighGUI) {
		//imshow(WINDOW_NAME_SIGNAL_RECOGNITION, processedImage);
		outputResults();		
	}

	return true;
}


void ImageAnalysis::preprocessImage(Mat& image, bool useCVHighGUI ) {	
	// remove noise with bilateral filter
	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}

	// histogram equalization to improve color segmentation
	histogramEqualization(image.clone(), false, useCVHighGUI);
	histogramEqualization(image, true, useCVHighGUI);

	// increase contrast and brightness to improve detection of numbers inside traffic sign
	image.convertTo(image, -1, (double)contrast / 10.0, (double)brightness / 10.0);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);	
	}
}


void ImageAnalysis::histogramEqualization(Mat& image, bool use_CLAHE, bool useCVHighGUI) {	
	cvtColor(image, image, CV_BGR2YCrCb);
	vector<Mat> channels;
	cv::split(image, channels);

	if (use_CLAHE) {
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((claehClipLimit < 1 ? 1 : claehClipLimit), cv::Size((claehTileXSize < 1 ? 1 : claehTileXSize) , (claehTileYSize < 1? 1 : claehTileYSize)));
		clahe->apply(channels[0], channels[0]);
	} else {
		cv::equalizeHist(channels[0], channels[0]);
	}

	cv::merge(channels, image);
	cvtColor(image, image, CV_YCrCb2BGR);	
	if (useCVHighGUI) {
		if (use_CLAHE) {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE, image);
		} else {
			imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);
		}
	}
}


void ImageAnalysis::segmentImage(Mat& image, bool useCVHighGUI) {
	// color segmentation	
	cvtColor(image, image, CV_BGR2HSV);
	Mat colorSegmentation;

	if (colorSegmentationLowerHue < colorSegmentationUpperHue) {
		cv::inRange(image,
			Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			colorSegmentation);
	} else {
		// when colors wrap around from near 180 to 0+				
		Mat lowerRange;
		cv::inRange(image,
			Scalar(0, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			lowerRange);
	
		Mat higherRange;
		cv::inRange(image,
			Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(180, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			higherRange);

		cv::bitwise_or(lowerRange, higherRange, colorSegmentation);
	}

	cvtColor(image, image, CV_HSV2BGR);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_COLOR_SEGMENTATION, colorSegmentation);
	}
}


void ImageAnalysis::recognizeTrafficSigns(Mat& image, Mat colorSegmentedImage, bool useCVHighGUI) {

}


bool ImageAnalysis::updateImage() {
	return processImage(originalImage.clone(), useCVHiGUI);
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Video processing>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
bool ImageAnalysis::processVideo(string path, bool useCVHighGUI) {	
	VideoCapture videoCapture;
	
	try {
		videoCapture = VideoCapture(path);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(int cameraDeviceNumber, bool useCVHighGUI) {	
	VideoCapture videoCapture;

	try {
		videoCapture = VideoCapture(cameraDeviceNumber);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(VideoCapture videoCapture, bool useCVHighGUI) {		
	if (!videoCapture.isOpened()) {
		return false;
	}

	useCVHiGUI = useCVHighGUI;
	windowsInitialized = false;

	int millisecPollingInterval = 1000 / frameRate;
	if (millisecPollingInterval < 10)
		millisecPollingInterval = 10;
	
	Mat currentFrame;	
	while (videoCapture.read(currentFrame)) {
		processImage(currentFrame, useCVHighGUI);
		
		if (waitKey(millisecPollingInterval) == ESC_KEYCODE) {
			break;
		}
	}

	if (useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return true;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </Video processing>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <OpenCV HighGUI>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void updateImageAnalysis(int position, void* userData) {		
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->updateImage();
}


void ImageAnalysis::setupMainWindow() {
	addHighGUIWindow(0, 0, WINDOW_NAME_MAIN);	
}


void ImageAnalysis::setupResultsWindows(bool optionsOneWindow) {
	addHighGUIWindow(1, 0, WINDOW_NAME_BILATERAL_FILTER);
	addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION);
	addHighGUIWindow(0, 1, WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE);
	addHighGUIWindow(1, 1, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS);
	addHighGUIWindow(2, 1, WINDOW_NAME_COLOR_SEGMENTATION);
	//addHighGUIWindow(2, 1, WINDOW_NAME_SIGNAL_RECOGNITION);
	
	if (optionsOneWindow) {		
		namedWindow(WINDOW_NAME_OPTIONS, CV_WINDOW_NORMAL);
		resizeWindow(WINDOW_NAME_OPTIONS, WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2, WINDOW_OPTIONS_HIGHT);
		moveWindow(WINDOW_NAME_OPTIONS, screenWidth - WINDOW_OPTIONS_WIDTH, 0);
	} else {						
		addHighGUITrackBarWindow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS, 3, 0, 0);
		addHighGUITrackBarWindow(WINDOW_NAME_BILATERAL_FILTER_OPTIONS, 3, 3, 1);
		addHighGUITrackBarWindow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, 2, 6, 2);
		addHighGUITrackBarWindow(WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, 6, 8, 3);
	}
		
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_CLIP, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehClipLimit, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileXSize, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileYSize, 20, updateImageAnalysis, (void*)this);
	
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_DIST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterDistance, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_COLOR_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaColor, 200, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_SPACE_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaSpace, 200, updateImageAnalysis, (void*)this);
		
	cv::createTrackbar(TRACK_BAR_NAME_CONTRAST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &contrast, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BRIGHTNESS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &brightness, 1000, updateImageAnalysis, (void*)this);
		
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperValue, 255, updateImageAnalysis, (void*)this);
}




void ImageAnalysis::addHighGUIWindow(int column, int row, string windowName, int numberColumns, int numberRows) {
	if (numberColumns < 1 || numberRows < 1)
		return;
	
	int imageWidth = originalImage.size().width;
	if (imageWidth < 10)
		imageWidth = screenWidth / 2;

	int imageHeight = originalImage.size().height;
	if (imageHeight < 10)
		imageHeight = screenHeight / 2;	


	int windowHeight = screenHeight / numberRows;
	int windowWidth = imageWidth * windowHeight / imageHeight;

	if ((windowWidth * numberColumns + WINDOW_OPTIONS_WIDTH) > screenWidth) {		
		windowWidth = ((screenWidth - WINDOW_OPTIONS_WIDTH) / numberColumns);
		windowHeight = imageHeight * windowWidth / imageWidth;
	}

	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	resizeWindow(windowName, windowWidth - 2 * WINDOW_FRAME_THICKNESS, windowHeight - WINDOW_FRAME_THICKNESS - WINDOW_HEADER_HEIGHT);
	
	int x = 0;
	if (column != 0) {
		x = windowWidth * column;
	}

	int y = 0;
	if (row != 0) {
		y = windowHeight * row;
	}

	moveWindow(windowName, x, y);
}

void ImageAnalysis::addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber) {
	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	int width = WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2;
	int height = numberTrackBars * WINDOW_OPTIONS_TRACKBAR_HEIGHT;
	resizeWindow(windowName, width, height);

	int heightPos = (WINDOW_FRAME_THICKNESS + WINDOW_HEADER_HEIGHT) * trackBarWindowNumber + WINDOW_OPTIONS_TRACKBAR_HEIGHT * cumulativeTrackBarPosition;
	moveWindow(windowName, screenWidth - WINDOW_OPTIONS_WIDTH, heightPos);
}


bool ImageAnalysis::outputResults() {	
	return true;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </OpenCV HighGUI>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
