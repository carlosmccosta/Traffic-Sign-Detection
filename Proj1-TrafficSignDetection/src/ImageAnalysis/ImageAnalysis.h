#pragma once

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::imread;
using cv::VideoCapture;
using cv::waitKey;
using cv::imshow;
using cv::namedWindow;
using cv::moveWindow;


#define NAME_MAIN_WINDOW "Image source"


class ImageAnalysis {
	private:
		vector<string> detectedSigns;
		Mat originalImage;

		int frameRate;


	public:
		ImageAnalysis() : frameRate(30) {};
		virtual ~ImageAnalysis();
		
		bool processImage(string path, bool useCVHighGUI = true);
		bool processImage(Mat image, bool useCVHighGUI = true);	
		
		bool processVideo(string path, bool useCVHighGUI = true);
		bool processVideo(int cameraDeviceNumber, bool useCVHighGUI = true);
		bool processVideo(VideoCapture videoCapture, bool useCVHighGUI = true);
		
		void setupMainWindow();
		void setupResultsWindows();
		bool outputResults();
};

