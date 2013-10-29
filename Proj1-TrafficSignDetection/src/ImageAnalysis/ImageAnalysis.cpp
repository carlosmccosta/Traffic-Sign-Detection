#include "ImageAnalysis.h"


ImageAnalysis::ImageAnalysis() :
	frameRate(30), screenWidth(1920), screenHeight(1080),
	colorSegmentationLowerHue(140), colorSegmentationUpperHue(210),
	colorSegmentationLowerSaturation(32), colorSegmentationUpperSaturation(200),
	colorSegmentationLowerValue(32), colorSegmentationUpperValue(255) {};


void udpateImageAnalysis(int position, void* userData) {
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->updateImage();
}

bool ImageAnalysis::processImage(Mat& image, bool useCVHighGUI) {
	detectedSigns.clear();
	originalImage = image.clone();
	useCVHiGUI = useCVHighGUI;

	if (useCVHighGUI) {
		setupMainWindow();
		imshow(WINDOW_NAME_MAIN, originalImage);
		setupResultsWindows();
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
	cv::bilateralFilter(originalImage, image, 9, 50, 10);		
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_BILATERAL_FILTER, image);
	}

	// increase contrast and brightness to improve detection of numbers inside traffic sign
	image.convertTo(image, -1, 1.2, 5);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);	
	}
}

void ImageAnalysis::segmentImage(Mat& image, bool useCVHighGUI) {
	// color segmentation	
	cvtColor(image, image, CV_BGR2HSV);
	Mat colorSegmentation;	
	cv::inRange(image, Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue), Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue), colorSegmentation);		
	cvtColor(image, image, CV_HSV2BGR);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_COLOR_SEGMENTATION, colorSegmentation);
	}
}

void ImageAnalysis::recognizeTrafficSigns(Mat& image, bool useCVHighGUI) {

}




bool ImageAnalysis::updateImage() {
	return processImage(originalImage, useCVHiGUI);
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
	
	bool status = processImage(imageToProcess, useCVHighGUI);	

	if (waitKey(0) == ESC_KEYCODE) {
		cv::destroyAllWindows();
	}

	return status;
}


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


void ImageAnalysis::setupMainWindow() {	
	addWindow(0, 0, WINDOW_NAME_MAIN);
}


void ImageAnalysis::setupResultsWindows() {
	addWindow(1, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION);
	addWindow(2, 0, WINDOW_NAME_BILATERAL_FILTER);
	addWindow(3, 0, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS);
	
	addWindow(0, 1, WINDOW_NAME_COLOR_SEGMENTATION);

	/*namedWindow(WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	moveWindow(WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, screenWidth / 2, screenHeight / 2);*/
	/*addWindow(1, 1, WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS);*/
	cv::createTrackbar("mHue", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerHue, 360, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("MHue", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperHue, 360, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("mSat", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerSaturation, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("MSat", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperSaturation, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("mVal", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationLowerValue, 255, udpateImageAnalysis, (void*)this);
	cv::createTrackbar("MVal", WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, &colorSegmentationUpperValue, 255, udpateImageAnalysis, (void*)this);
	addWindow(1, 1, WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS);
}


void ImageAnalysis::addWindow(int column, int row, string name, int numberColumns, int numberRows) {
	namedWindow(name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);	
	
	int height = screenHeight / numberRows;
	int width = originalImage.size().width * height / originalImage.size().height;

	if (width * numberColumns > screenWidth) {		
		width = (screenWidth / numberColumns) - 2 * WINDOW_FRAME_THICKNESS;
		height = originalImage.size().height * width / originalImage.size().width;
	}

	resizeWindow(name, width, height);
	
	int x = 0;
	if (column != 0) {
		x = (width + WINDOW_FRAME_THICKNESS * 2) * column;
	}

	int y = 0;
	if (row != 0) {
		y = (height + WINDOW_HEADER_HEIGHT + WINDOW_FRAME_THICKNESS) * row;
	}

	moveWindow(name, x, y);
}


bool ImageAnalysis::outputResults() {
	setupResultsWindows();

	return true;
}

