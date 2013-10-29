#include "ImageAnalysis.h"

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
	
	return processImage(imageToProcess, useCVHighGUI);	
}


bool ImageAnalysis::processImage(Mat image, bool useCVHighGUI) {
	originalImage = image;
	
	if (useCVHighGUI) {
		setupMainWindow();
		imshow(NAME_MAIN_WINDOW, originalImage);		
	}

	detectedSigns.clear();


	if (useCVHighGUI) {
		outputResults();

		if (waitKey(0) == ESC_KEYCODE) {
			cv::destroyAllWindows();
		}
	}

	return true;
}


bool ImageAnalysis::processVideo(string path, bool useCVHighGUI) {	
	VideoCapture videoCapture;
	
	try {
		videoCapture.open(path);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(int cameraDeviceNumber, bool useCVHighGUI) {	
	VideoCapture videoCapture;

	try {
		videoCapture.open(cameraDeviceNumber);
	} catch (...) {
		return false;
	}

	return processVideo(videoCapture, useCVHighGUI);
}


bool ImageAnalysis::processVideo(VideoCapture videoCapture, bool useCVHighGUI) {		
	if (!videoCapture.isOpened()) {
		return false;
	}

	if (useCVHighGUI) {
		setupMainWindow();
	}

	int millisecPollingInterval = 1000 / frameRate;
	
	while (videoCapture.read(originalImage)) {
		processImage(originalImage, false);

		if (useCVHighGUI) {
			imshow(NAME_MAIN_WINDOW, originalImage);
			outputResults();
		}

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
	namedWindow(NAME_MAIN_WINDOW, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);	
	moveWindow(NAME_MAIN_WINDOW, 0, 0);
}


void ImageAnalysis::setupResultsWindows() {
	
}

bool ImageAnalysis::outputResults() {
	setupResultsWindows();

	return true;
}
