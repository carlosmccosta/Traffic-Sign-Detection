#include "ImageAnalysis.h"



bool ImageAnalysis::processImage(Mat& image, bool useCVHighGUI) {
	detectedSigns.clear();
	originalImage = image.clone();

	if (useCVHighGUI) {
		setupMainWindow();
		imshow(WINDOW_NAME_MAIN, originalImage);
		setupResultsWindows();
	}

	preprocessImage(image, useCVHighGUI);


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
	imshow(WINDOW_NAME_HISTOGRAM_EQUALIZATION, image);

	// remove noise with bilateral filter
	cv::bilateralFilter(originalImage, image, 9, 50, 10);		
	imshow(WINDOW_NAME_BILATERAL_FILTER, image);

	// increase contrast and brightness to improve detection of numbers inside traffic sign
	image.convertTo(image, -1, 1.5, 20);
	imshow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS, image);	
}

void ImageAnalysis::segmentImage(Mat& image, bool useCVHighGUI) {

}

void ImageAnalysis::recognizeTrafficSigns(Mat& image, bool useCVHighGUI) {

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

