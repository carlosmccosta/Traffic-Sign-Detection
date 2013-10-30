#include "ImageAnalysis.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageAnalysis::ImageAnalysis() :
	useCVHiGUI(true), windowsInitialized(false),
	frameRate(30), screenWidth(1920), screenHeight(1080),
	bilateralFilterDistance(9), bilateralFilterSigmaColor(50), bilateralFilterSigmaSpace(10),
	contrast(12), brightness(50),
	colorSegmentationLowerHue(140), colorSegmentationUpperHue(210),
	colorSegmentationLowerSaturation(32), colorSegmentationUpperSaturation(255),
	colorSegmentationLowerValue(32), colorSegmentationUpperValue(255) {};


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

	if (waitKey(0) == ESC_KEYCODE && useCVHighGUI) {
		cv::destroyAllWindows();
	}

	return status;
}


bool ImageAnalysis::processImage(Mat& image, bool useCVHighGUI) {
	detectedSigns.clear();
	originalImage = image.clone();
	useCVHiGUI = useCVHighGUI;		
	
	if (useCVHighGUI) {		
		if (!windowsInitialized) {
			setupMainWindow();
			setupResultsWindows();
			windowsInitialized = true;
		}		

		imshow(WINDOW_NAME_MAIN, originalImage);		
	}

	preprocessImage(image, useCVHighGUI);
	segmentImage(image, useCVHighGUI);


	if (useCVHighGUI) {
		outputResults();		
	}

	return true;
}


void ImageAnalysis::preprocessImage(Mat& image, bool useCVHighGUI ) {	
	// histogram equalization to improve color segmentation
	cvtColor(image, image, CV_BGR2YCrCb);
	vector<Mat> channels;
	cv::split(image, channels);
	cv::equalizeHist(channels[0], channels[0]);	
	cv::merge(channels, image);
	cvtColor(image, image, CV_YCrCb2BGR);	
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);
	}

	// remove noise with bilateral filter
	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}

	// increase contrast and brightness to improve detection of numbers inside traffic sign
	image.convertTo(image, -1, (double)contrast / 10.0, (double)brightness / 10.0);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);	
	}
}


void ImageAnalysis::segmentImage(Mat& image, bool useCVHighGUI) {
	// color segmentation	
	cvtColor(image, image, CV_BGR2HSV);
	Mat colorSegmentation;	
	cv::inRange(image,
		Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
		Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
		colorSegmentation);
	cvtColor(image, image, CV_HSV2BGR);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_COLOR_SEGMENTATION, colorSegmentation);
	}
}


void ImageAnalysis::recognizeTrafficSigns(Mat& image, bool useCVHighGUI) {

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
void udpateImageAnalysis(int position, void* userData) {
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->updateImage();
}


void ImageAnalysis::setupMainWindow() {
	addHighGUIWindow(0, 0, WINDOW_NAME_MAIN);
}


void ImageAnalysis::setupResultsWindows() {
	addHighGUIWindow(1, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION);
	
	addHighGUIWindow(2, 0, WINDOW_NAME_BILATERAL_FILTER);
	addHighGUIWindow(3, 0, WINDOW_NAME_BILATERAL_FILTER_OPTIONS);
	cv::createTrackbar("Dist", WINDOW_NAME_BILATERAL_FILTER_OPTIONS, &bilateralFilterDistance, 100, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Color Sig", WINDOW_NAME_BILATERAL_FILTER_OPTIONS, &bilateralFilterSigmaColor, 200, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Space Sig", WINDOW_NAME_BILATERAL_FILTER_OPTIONS, &bilateralFilterSigmaSpace, 200, udpateImageAnalysis, (void*)this);
	
	addHighGUIWindow(0, 1, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS);
	addHighGUIWindow(1, 1, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS);
	cv::createTrackbar("Contr * 10", WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, &contrast, 100, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Brigh * 10", WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, &brightness, 1000, udpateImageAnalysis, (void*)this);

	addHighGUIWindow(2, 1, WINDOW_NAME_COLOR_SEGMENTATION);	
	addHighGUIWindow(3, 1, WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS);
	cv::createTrackbar("Min Hue", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerHue, 360, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Max Hue", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperHue, 360, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Min Sat", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerSaturation, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Max Sat", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperSaturation, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Min Val", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerValue, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("Max Val", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperValue, 255, udpateImageAnalysis, (void*)this);	
}


void ImageAnalysis::addHighGUIWindow(int column, int row, string windowName, int numberColumns, int numberRows) {
	if (numberColumns < 1 || numberRows < 1)
		return;
	
	int imageHeight = originalImage.size().height;
	if (imageHeight < 10)
		imageHeight = screenHeight / 2;

	int imageWidth = originalImage.size().width;
	if (imageWidth < 10)
		imageWidth = screenWidth / 2;

	int height = screenHeight / numberRows - WINDOW_FRAME_THICKNESS - WINDOW_HEADER_HEIGHT;
	int width = originalImage.size().width * height / imageHeight;

	if (width * numberColumns > screenWidth) {		
		width = (screenWidth / numberColumns) - 2 * WINDOW_FRAME_THICKNESS;
		height = originalImage.size().height * width / imageWidth;
	}

	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	resizeWindow(windowName, width, height);
	
	int x = 0;
	if (column != 0) {
		x = (width + WINDOW_FRAME_THICKNESS * 2) * column;
	}

	int y = 0;
	if (row != 0) {
		y = (height + WINDOW_HEADER_HEIGHT + WINDOW_FRAME_THICKNESS) * row;
	}

	moveWindow(windowName, x, y);
}


bool ImageAnalysis::outputResults() {	
	return true;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </OpenCV HighGUI>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
