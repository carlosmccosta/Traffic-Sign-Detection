#include "ImageAnalysis.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageAnalysis::ImageAnalysis() :
	useCVHiGUI(true), windowsInitialized(false),
	frameRate(30), screenWidth(1920), screenHeight(1080),
	claehClipLimit(2), claehTileXSize(2), claehTileYSize(2),
	bilateralFilterDistance(9), bilateralFilterSigmaColor(50), bilateralFilterSigmaSpace(10),
	contrast(11), brightness(25),
	colorSegmentationLowerHue(147), colorSegmentationUpperHue(7),
	colorSegmentationLowerSaturation(112), colorSegmentationUpperSaturation(255),
	colorSegmentationLowerValue(32), colorSegmentationUpperValue(255),
	textColorSegmentationLowerHue(20), textColorSegmentationUpperHue(140),
	textColorSegmentationLowerSaturation(0), textColorSegmentationUpperSaturation(255),
	textColorSegmentationLowerValue(0), textColorSegmentationUpperValue(147),
	cannyLowerHysteresisThreshold(100), cannyHigherHysteresisThreshold(200), cannySobelOperatorKernelSize(3),
	houghCirclesDP(1), houghCirclesMinDistanceCenters(2),
	houghCirclesCannyHigherThreshold(200), houghCirclesAccumulatorThreshold(25),
	houghCirclesMinRadius(1), houghCirclesMaxRadius(100) {};


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
	}

	preprocessedImage = image.clone();	
	preprocessImage(preprocessedImage, useCVHighGUI);

	Mat preprocessedImageClone = preprocessedImage.clone();

	Mat imageColorSegmented = segmentImageByTrafficSignColor(preprocessedImage, useCVHighGUI);	
	vector< pair<Rect, RotatedRect> > recognizedEllipsis;
	recognizeTrafficSignsEllipsis(imageColorSegmented, preprocessedImage, recognizedEllipsis, useCVHighGUI);		
	segmentImageByTrafficSignText(preprocessedImageClone, recognizedEllipsis, useCVHighGUI);


	processedImage = preprocessedImage;
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_MAIN, originalImage);
		imshow(WINDOW_NAME_SIGNAL_RECOGNITION, processedImage);
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
	//histogramEqualization(image.clone(), false, useCVHighGUI);
	histogramEqualization(image, true, useCVHighGUI);

	// increase contrast and brightness to improve detection of numbers inside traffic sign
	image.convertTo(image, -1, (double)contrast / 10.0, (double)brightness / 10.0);

	cv::bilateralFilter(image.clone(), image, bilateralFilterDistance, bilateralFilterSigmaColor, bilateralFilterSigmaSpace);
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


Mat ImageAnalysis::segmentImageByTrafficSignColor(Mat& preprocessedImage, bool useCVHighGUI) {
	// color segmentation	
	cvtColor(preprocessedImage, preprocessedImage, CV_BGR2HSV);
	Mat colorSegmentation;

	if (colorSegmentationLowerHue < colorSegmentationUpperHue) {
		cv::inRange(preprocessedImage,
			Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			colorSegmentation);
	} else {
		// when colors wrap around from near 180 to 0+				
		Mat lowerRange;
		cv::inRange(preprocessedImage,
			Scalar(0, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(colorSegmentationUpperHue, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			lowerRange);
	
		Mat higherRange;
		cv::inRange(preprocessedImage,
			Scalar(colorSegmentationLowerHue, colorSegmentationLowerSaturation, colorSegmentationLowerValue),
			Scalar(180, colorSegmentationUpperSaturation, colorSegmentationUpperValue),
			higherRange);

		cv::bitwise_or(lowerRange, higherRange, colorSegmentation);
	}

	cvtColor(preprocessedImage, preprocessedImage, CV_HSV2BGR);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_COLOR_SEGMENTATION, colorSegmentation);
	}

	return colorSegmentation;
}


void ImageAnalysis::recognizeTrafficSignsEllipsis(Mat& colorSegmentedImage, Mat& preprocessedImage, vector< pair<Rect,RotatedRect> >& outputRecognizedEllipsis, bool useCVHighGUI) {	
	/*if (cannySobelOperatorKernelSize == 4 || cannySobelOperatorKernelSize == 6)
		--cannySobelOperatorKernelSize;

	if (cannySobelOperatorKernelSize < 3)
		cannySobelOperatorKernelSize = 3;

	if (cannySobelOperatorKernelSize > 7)
		cannySobelOperatorKernelSize = 7;

	Canny(colorSegmentedImage, colorSegmentedImage,
		(cannyLowerHysteresisThreshold < 1 ? 1 : cannyLowerHysteresisThreshold),
		(cannyHigherHysteresisThreshold < 1 ? 1 : cannyHigherHysteresisThreshold),
		cannySobelOperatorKernelSize, true);
	*/
	/*if (useCVHighGUI) {
		imshow(WINDOW_NAME_SIGNAL_CANNY, colorSegmentedImage);
	}*/

	vector<Vec3f> houghCircles;
	int imageHeight = colorSegmentedImage.size().height;
	int circlesMinDistanceCenters = imageHeight * houghCirclesMinDistanceCenters / 100;
	int circlesMinRadius = imageHeight * houghCirclesMinRadius / 100;
	int circlesMaxRadius = imageHeight * houghCirclesMaxRadius / 100;

	cv::HoughCircles(colorSegmentedImage, houghCircles, CV_HOUGH_GRADIENT,
		(houghCirclesDP < 1 ? 1 : houghCirclesDP),
		(circlesMinDistanceCenters < 1 ? 1 : circlesMinDistanceCenters),
		(houghCirclesCannyHigherThreshold < 1 ? 1 : houghCirclesCannyHigherThreshold),
		(houghCirclesAccumulatorThreshold < 1 ? 1 : houghCirclesAccumulatorThreshold),
		circlesMinRadius, circlesMaxRadius);
	
	vector<Vec3f> houghCirclesFiltered;
	filterRecognizedTrafficSignCircles(houghCircles, houghCirclesFiltered);

	if (useCVHighGUI) {
		for (size_t i = 0; i < houghCirclesFiltered.size(); ++i) {
			Point center(cvRound(houghCirclesFiltered[i][0]), cvRound(houghCirclesFiltered[i][1]));
			int radius = cvRound(houghCirclesFiltered[i][2]);

			circle(preprocessedImage, center, 1, COLOR_HOUGH_CIRCLES_BGR, 2);
			circle(preprocessedImage, center, radius, COLOR_HOUGH_CIRCLES_BGR, 2);
		}
	}
	
	retrieveEllipsisFromHoughCircles(colorSegmentedImage, houghCirclesFiltered, outputRecognizedEllipsis, useCVHighGUI);
}


bool sortCircleClusterByMedianY(const Vec3f& left, const Vec3f& right) {
	return left[1] < right[1];
}

bool sortCircleClusterByRadius(const Vec3f& left, const Vec3f& right) {
	return left[2] < right[2];
}

void ImageAnalysis::filterRecognizedTrafficSignCircles(const vector<Vec3f>& houghCircles, vector<Vec3f>& outputHoughCirclesFiltered) {
	if (houghCircles.size() > 1) {				
		// aggregate circles closer to each other (with centers inside each other forming a cluster)
		vector< vector<Vec3f> > houghCirclesClusters;
		vector<Vec3f> firstCluster;
		firstCluster.push_back(houghCircles[0]);
		houghCirclesClusters.push_back(firstCluster);

		for (size_t allCirclesPos = 1; allCirclesPos < houghCircles.size(); ++allCirclesPos) {
			const Vec3f& centerToAdd = houghCircles[allCirclesPos];

			if (!aggregateCircleIntoClusters(houghCirclesClusters, centerToAdd)) {
				vector<Vec3f> newCluster;
				newCluster.push_back(centerToAdd);
				houghCirclesClusters.push_back(newCluster);
			}
		}


		// select the circle with the median y in each cluster (different traffic sign)	
		flatClustersByMeanCenter(houghCirclesClusters, outputHoughCirclesFiltered);
		//flatClustersByMedianCenter(houghCirclesClusters, outputHoughCirclesFiltered);
		//flatClustersByMaxRadius(houghCirclesClusters, outputHoughCirclesFiltered);	
	}
}

void ImageAnalysis::flatClustersByMeanCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered) {	
	for (size_t circleClusterPos = 0; circleClusterPos < houghCirclesClusters.size(); ++circleClusterPos) {
		vector<Vec3f> currentCluster = houghCirclesClusters[circleClusterPos];

		if (currentCluster.size() > 1) {
			Vec3f meanCircle = currentCluster[0];
			for (size_t circleInClusterPos = 1; circleInClusterPos < currentCluster.size(); ++circleInClusterPos) {
				meanCircle[0] += currentCluster[circleInClusterPos][0];
				meanCircle[1] += currentCluster[circleInClusterPos][1];
				meanCircle[2] += currentCluster[circleInClusterPos][2];
			}

			meanCircle[0] /= currentCluster.size();
			meanCircle[1] /= currentCluster.size();
			meanCircle[2] /= currentCluster.size();

			outputHoughCirclesFiltered.push_back(meanCircle);
		} else if (currentCluster.size() == 1) {
			outputHoughCirclesFiltered.push_back(currentCluster[0]);
		}
	}
}


void ImageAnalysis::flatClustersByMedianCenter(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered) {	
	for (size_t circleClusterPos = 0; circleClusterPos < houghCirclesClusters.size(); ++circleClusterPos) {
		vector<Vec3f> currentCluster = houghCirclesClusters[circleClusterPos];

		if (currentCluster.size() > 1) {
			sort(currentCluster.begin(), currentCluster.end(), sortCircleClusterByMedianY);
			Vec3f selectedCircle;
			int middlePos = currentCluster.size() / 2;
			int middlePosMinus1 = middlePos - 1;
			if (currentCluster.size() % 2) {	
				selectedCircle[0] = (currentCluster[middlePos][0] + currentCluster[middlePosMinus1][0]) / 2;
				selectedCircle[1] = (currentCluster[middlePos][1] + currentCluster[middlePosMinus1][1]) / 2;
				selectedCircle[2] = (currentCluster[middlePos][2] + currentCluster[middlePosMinus1][2]) / 2;
			} else {
				selectedCircle = currentCluster[middlePos];
			}
			outputHoughCirclesFiltered.push_back(selectedCircle);
		} else if (currentCluster.size() == 1) {
			outputHoughCirclesFiltered.push_back(currentCluster[0]);
		}
	}
}


void ImageAnalysis::flatClustersByMaxRadius(vector< vector<Vec3f> > &houghCirclesClusters, vector<Vec3f> &outputHoughCirclesFiltered) {	
	for (size_t circleClusterPos = 0; circleClusterPos < houghCirclesClusters.size(); ++circleClusterPos) {
		vector<Vec3f> currentCluster = houghCirclesClusters[circleClusterPos];

		if (currentCluster.size() > 1) {
			sort(currentCluster.begin(), currentCluster.end(), sortCircleClusterByRadius);			
			outputHoughCirclesFiltered.push_back(currentCluster.back());
		} else if (currentCluster.size() == 1) {
			outputHoughCirclesFiltered.push_back(currentCluster[0]);
		}
	}
}


bool ImageAnalysis::aggregateCircleIntoClusters(vector< vector<Vec3f> >& houghCirclesClusters, const Vec3f& centerToAdd) {
	for (size_t clusterPos = 0; clusterPos < houghCirclesClusters.size(); ++clusterPos) {
		size_t numberCirclesToTest = houghCirclesClusters[clusterPos].size();
		for (size_t centerToTestPos = 0; centerToTestPos < numberCirclesToTest; ++centerToTestPos) {
			const Vec3f& centerToTest = houghCirclesClusters[clusterPos][centerToTestPos];
			float maxRadius = std::max(centerToTest[2], centerToAdd[2]);
			float dx = centerToTest[0] - centerToAdd[0];
			float dy = centerToTest[1] - centerToAdd[1];
			float distance = std::sqrt(dx*dx + dy*dy);

			// aggregate the new circle if one of the then has the center inside of the other
			if (distance < maxRadius) {
				houghCirclesClusters[clusterPos].push_back(centerToAdd);				
				return true;
			}
		}
	}
	
	return false;
}


void ImageAnalysis::retrieveEllipsisFromHoughCircles(const Mat& colorSegmentedImage, const vector<Vec3f>& houghCirclesFiltered, vector<pair<Rect, RotatedRect> >& outputTrafficSignEllipsis, bool useCVHighGUI) {
	Mat colorSegmentedImageContours = colorSegmentedImage.clone();
	int imageWidth = colorSegmentedImage.size().width;
	int imageHeight = colorSegmentedImage.size().height;
	for (size_t currentCirclePos = 0; currentCirclePos < houghCirclesFiltered.size(); ++currentCirclePos) {
		const Vec3f& currentCircle = houghCirclesFiltered[currentCirclePos];
		int currentCircleCenterX = cvRound(currentCircle[0]);
		int currentCircleCenterY = cvRound(currentCircle[1]);
		int currentCircleRadius = cvRound(currentCircle[2]);

		// add offset to compensate errors from hough transform (radius with offset cannot surpass image boundaries)		
		int maxRadiusAllowed = std::min(std::min(cvRound(currentCircle[0]), cvRound(currentCircle[1])), std::min(imageWidth - currentCircleCenterX, imageHeight - currentCircleCenterY));
		int radiusRadiusWithOffset = std::min(cvRound(currentCircleRadius * PARAM_FIT_ELLIPSIS_SCALE_FOR_HOUGH_CIRCLE), maxRadiusAllowed);
		int roiWidth = std::min(radiusRadiusWithOffset * 2, imageWidth);
		int roiHeight = std::min(radiusRadiusWithOffset * 2, imageHeight);
		int roiX = std::max(currentCircleCenterX - radiusRadiusWithOffset, 0);
		int roiY = std::max(currentCircleCenterY - radiusRadiusWithOffset, 0);
		
		// make sure roi is inside image
		roiWidth = std::min(roiWidth, imageWidth - roiX);
		roiHeight = std::min(roiHeight, imageHeight - roiY);
		
		try {
			Mat circleROI = colorSegmentedImageContours.clone()(Rect(roiX, roiY, roiWidth, roiHeight));
			vector<vector<Point> > contours;
				
			cv::findContours(circleROI, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(roiX, roiY));

			if (!contours.empty()) {		
				vector<Point>& biggestContour = contours[0];
				double biggestContourArea = cv::contourArea(biggestContour);
				for (size_t contourPos = 1; contourPos < contours.size(); ++contourPos) {
					vector<Point>& currentContour = contours[contourPos];
					double currentContourArea = cv::contourArea(currentContour);
					if (currentContourArea > biggestContourArea && currentContour.size() > 4) {
						biggestContour = currentContour;
						biggestContourArea = currentContourArea;
					}
				}

				// fitEllipse requires at lest 4 points
				if (biggestContour.size() > 4) {
					RotatedRect ellipseFound = cv::fitEllipse(biggestContour);
					Rect ellipseBoundingRect = boundingRect(biggestContour);

					// ellipse bounding rect must be inside image					
					ellipseBoundingRect.x = std::max(ellipseBoundingRect.x, 0);
					ellipseBoundingRect.y = std::max(ellipseBoundingRect.y, 0);
					ellipseBoundingRect.width = std::min(ellipseBoundingRect.width, imageWidth - ellipseBoundingRect.x);
					ellipseBoundingRect.height = std::min(ellipseBoundingRect.height, imageHeight - ellipseBoundingRect.y);
					
					// ellipse center must be inside hough transform circle
					int dx = cvRound(ellipseFound.center.x - currentCircleCenterX);
					int dy = cvRound(ellipseFound.center.y - currentCircleCenterY);
					if (sqrt(dx*dx + dy*dy) < currentCircleRadius) {
						outputTrafficSignEllipsis.push_back(pair<Rect, RotatedRect>(ellipseBoundingRect, ellipseFound));

						if (useCVHighGUI) {						
							circle(preprocessedImage, ellipseFound.center, 1, COLOR_ELLIPSIS_BGR, 2);
							ellipse(preprocessedImage, ellipseFound, COLOR_ELLIPSIS_BGR, 2);
							rectangle(preprocessedImage, ellipseBoundingRect, COLOR_ELLIPSIS_BGR, 2);
						}
					}
				}
			}
		} catch (...) { }
	}	
}


vector<int> ImageAnalysis::segmentImageByTrafficSignText(Mat& preprocessedImage, vector< pair<Rect, RotatedRect> >& trafficSignEllipsis, bool useCVHighGUI) {			
	Mat imageHSV;
	cvtColor(preprocessedImage, imageHSV, CV_BGR2HSV);
	Mat imageROIs(imageHSV.rows, imageHSV.cols, imageHSV.type(), Scalar(0,0,255));
	Mat imageTexts(imageHSV.rows, imageHSV.cols, CV_8U, Scalar(0));

	for(size_t ellipsePos = 0; ellipsePos < trafficSignEllipsis.size(); ++ellipsePos) {
		RotatedRect& currentEllipse = trafficSignEllipsis[ellipsePos].second;
		Rect ellipseBoundingRect = trafficSignEllipsis[ellipsePos].first;
		
		Mat imageHSVClone = imageHSV.clone();
		Mat ellipseROIMaskBlack(imageHSVClone.rows, imageHSVClone.cols, imageHSVClone.type(), Scalar(0,0,0));
		ellipse(ellipseROIMaskBlack, currentEllipse, Scalar(255,255,255), -1);

		Mat ellipseROIMaskWhite(imageHSVClone.rows, imageHSVClone.cols, imageHSVClone.type(), Scalar(0,0,255));
		ellipse(ellipseROIMaskWhite, currentEllipse, Scalar(0,0,0), -1);

		// extract pixels only inside ellipse		
		bitwise_and(imageHSVClone, ellipseROIMaskBlack, imageHSVClone);

		// pixels outside image cant be black because traffic sign letters are black and they will be color segmented
		cv::add(imageHSVClone, ellipseROIMaskWhite, imageHSVClone);
		
		
		Mat ellipseROI = imageHSVClone(ellipseBoundingRect);
		Mat textColorSegmentation;

		cv::inRange(ellipseROI,
			Scalar(textColorSegmentationLowerHue, textColorSegmentationLowerSaturation, textColorSegmentationLowerValue),
			Scalar(textColorSegmentationUpperHue, textColorSegmentationUpperSaturation, textColorSegmentationUpperValue),
			textColorSegmentation);

		if (useCVHighGUI) {
			try {
				ellipseROI.copyTo(imageROIs(ellipseBoundingRect));
				textColorSegmentation.copyTo(imageTexts(ellipseBoundingRect));
			} catch(...) {}
		}

		int detectedSign = recognizeTrafficSignText(imageROIs, textColorSegmentation, ellipseBoundingRect, useCVHighGUI);
		if (detectedSign > 0 && detectedSign % 5 == 0) {
			detectedSigns.push_back(detectedSign);

			if (useCVHighGUI) {
				stringstream ss;
				ss << detectedSign;
				drawTrafficSignLabel(ss.str(), imageROIs, ellipseBoundingRect);
			}
		}		
	}

	if (useCVHighGUI) {
		try {			
			cvtColor(imageROIs, imageROIs, CV_HSV2BGR);
			imshow(WINDOW_NAME_SIGNAL_ROI, imageROIs);
			imshow(WINDOW_NAME_SIGNAL_TEXTS, imageTexts);
		} catch(...) {}
	}

	return detectedSigns;
}


bool sortContourAreas(const pair< pair< vector<Point>*, size_t>, double >& left, const pair< pair< vector<Point>*, size_t>, double >& right) {
	return left.second > right.second;
}


bool sortContourByHorizontalPosition(const pair< pair<vector<Point>*, size_t>, Rect>& left, const pair< pair<vector<Point>*, size_t>, Rect>& right) {
	return left.second.x < right.second.x;
}


int ImageAnalysis::recognizeTrafficSignText(Mat& preprocessedImage, Mat& textColorSegmentation, const Rect& ellipseBoundingRect, bool useCVHighGUI) {
	vector<vector<Point> > contours;
	int roiX = ellipseBoundingRect.x;
	int roiY = ellipseBoundingRect.y;

	cv::findContours(textColorSegmentation.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(roiX, roiY));

	if (contours.empty()) {
		return -1;
	}

	vector< pair< pair<vector<Point>*, size_t>, Rect> > biggestContours;	

	// extract 3 biggest contours if their area is bigger than ellipseBoundingRectArea * PARAM_TEXT_MIN_PERCENTAGE_IN_SIGN
	if (contours.size() > 1) {
		vector< pair< pair< vector<Point>*, size_t>, double > > contourAreas;
	
		// compute contour area
		for (size_t contourPos = 0; contourPos < contours.size(); ++contourPos) {
			vector<Point>* currentContour = &(contours[contourPos]);
			double currentContourArea = cv::contourArea(*currentContour);
			contourAreas.push_back(pair< pair< vector<Point>*, size_t>, double >(pair< vector<Point>*, size_t>(currentContour, contourPos), currentContourArea));
		}

		sort(contourAreas.begin(), contourAreas.end(), sortContourAreas);

		double ellipseBoundingRectArea = ellipseBoundingRect.width * ellipseBoundingRect.height;
		double minAreaForSignText = ellipseBoundingRectArea * PARAM_TEXT_MIN_PERCENTAGE_IN_SIGN;
		
		// extract 3 biggest contours
		for (size_t contourAreasPos = 0; contourAreasPos < 3 && contourAreasPos < contourAreas.size(); ++contourAreasPos) {
			pair< pair< vector<Point>*, size_t>, double >& currentBiggestContour = contourAreas[contourAreasPos];
			if (currentBiggestContour.second > minAreaForSignText) {								
				biggestContours.push_back(pair< pair<vector<Point>*, size_t>, Rect>(currentBiggestContour.first, boundingRect(*(currentBiggestContour.first.first))));
			}
		}
		
		// sort contours by their horizontal position
		sort(biggestContours.begin(), biggestContours.end(), sortContourByHorizontalPosition);
	}


	if (useCVHighGUI) {
		for (size_t biggestContoursPos = 0; biggestContoursPos < biggestContours.size(); ++biggestContoursPos) {
			rectangle(preprocessedImage, biggestContours[biggestContoursPos].second, COLOR_TEXT_HSV, 2);
			drawContours(preprocessedImage, contours, biggestContours[biggestContoursPos].first.second, COLOR_TEXT_HSV, 2);
		}
	}
	

	return 120;
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
		try {
			processImage(currentFrame, useCVHighGUI);
		} catch(...) {}
		
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


void ImageAnalysis::drawTrafficSignLabel(string text, Mat& image, const Rect& signBoundingRect) {
	int textBoxHeight = (int)(signBoundingRect.height * 0.15);
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = (double)textBoxHeight / 46.0;
	int thickness = std::max(1, (int)(textBoxHeight * 0.05));
	int baseline = 0;

	Rect textBoundingRect = signBoundingRect;
	textBoundingRect.height = std::max(textBoxHeight, TEXT_MIN_SIZE);
	//textBoundingRect.y -= textBoundingRect.height;

	cv::Size textSize = cv::getTextSize(text, fontface, scale, thickness, &baseline);
	cv::Point textBottomLeftPoint(textBoundingRect.x + (textBoundingRect.width - textSize.width) / 2, textBoundingRect.y + (textBoundingRect.height + textSize.height) / 2);

	cv::rectangle(image, signBoundingRect, COLOR_LABEL_BOX_HSV, 2);
	cv::rectangle(image, textBoundingRect, COLOR_LABEL_BOX_HSV, 2);
	cv::putText(image, text, textBottomLeftPoint, fontface, scale, COLOR_LABEL_TEXT_HSV, thickness);
}


void ImageAnalysis::setupMainWindow() {
	addHighGUIWindow(0, 0, WINDOW_NAME_MAIN);	
}


void ImageAnalysis::setupResultsWindows(bool optionsOneWindow) {
	addHighGUIWindow(1, 0, WINDOW_NAME_BILATERAL_FILTER);
	//addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION);
	addHighGUIWindow(2, 0, WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE);
	addHighGUIWindow(3, 0, WINDOW_NAME_CONTRAST_AND_BRIGHTNESS);
	addHighGUIWindow(0, 1, WINDOW_NAME_COLOR_SEGMENTATION);
	//addHighGUIWindow(1, 1, WINDOW_NAME_SIGNAL_CANNY);
	addHighGUIWindow(1, 1, WINDOW_NAME_SIGNAL_RECOGNITION);
	addHighGUIWindow(2, 1, WINDOW_NAME_SIGNAL_TEXTS);
	addHighGUIWindow(3, 1, WINDOW_NAME_SIGNAL_ROI);	
	
	if (optionsOneWindow) {		
		namedWindow(WINDOW_NAME_OPTIONS, CV_WINDOW_NORMAL);
		resizeWindow(WINDOW_NAME_OPTIONS, WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2, WINDOW_OPTIONS_HIGHT);
		moveWindow(WINDOW_NAME_OPTIONS, screenWidth - WINDOW_OPTIONS_WIDTH, 0);
	} else {						
		addHighGUITrackBarWindow(WINDOW_NAME_BILATERAL_FILTER_OPTIONS, 3, 0, 0);
		addHighGUITrackBarWindow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS, 3, 3, 1);		
		addHighGUITrackBarWindow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, 2, 6, 2);
		addHighGUITrackBarWindow(WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS, 6, 8, 3, 2 * WINDOW_FRAME_THICKNESS);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_TEXTS_OPTIONS, 6, 8, 3, 0, WINDOW_HEADER_HEIGHT);
		/*addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_CANNY_OPTIONS, 3, 14, 4);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS, 6, 17, 5);*/
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS, 6, 14, 4);
	}	
	
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_DIST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterDistance, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_COLOR_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaColor, 200, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_SPACE_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaSpace, 200, updateImageAnalysis, (void*)this);
	
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_CLIP, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehClipLimit, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileXSize, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileYSize, 20, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_CONTRAST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &contrast, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BRIGHTNESS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &brightness, 1000, updateImageAnalysis, (void*)this);
		
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationLowerValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_COLOR_SEGMENTATION_OPTIONS), &colorSegmentationUpperValue, 255, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationLowerHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationUpperHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationLowerSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationUpperSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MIN_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationLowerValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_COLOR_SEG_MAX_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_TEXTS_OPTIONS), &textColorSegmentationUpperValue, 255, updateImageAnalysis, (void*)this);

	/*cv::createTrackbar(TRACK_BAR_NAME_CANNY_LOWER_HYSTERESIS_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannyLowerHysteresisThreshold, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CANNY_HIGHER_HYSTERESIS_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannyHigherHysteresisThreshold, 300, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CANNY_KERNEL_SIZE_SOBEL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannySobelOperatorKernelSize, 7, updateImageAnalysis, (void*)this);*/

	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_DP, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesDP, 10, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_DIST_CENTERS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMinDistanceCenters, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_CANNY_HIGHER_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesCannyHigherThreshold, 500, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesAccumulatorThreshold, 250, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_RADIUS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMinRadius, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MAX_RADIUS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMaxRadius, 100, updateImageAnalysis, (void*)this);
}


pair<int, int> ImageAnalysis::addHighGUIWindow(int column, int row, string windowName, int numberColumns, int numberRows, int xOffset, int yOffset) {
	if (numberColumns < 1 || numberRows < 1)
		return pair<int, int>(0, 0);
	
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

	x += xOffset;
	y += yOffset;
	
	moveWindow(windowName, x, y);

	return pair<int, int>(x, y);
}


pair<int, int> ImageAnalysis::addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber, int xOffset, int yOffset) {
	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	int width = WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2;
	int height = numberTrackBars * WINDOW_OPTIONS_TRACKBAR_HEIGHT;
	resizeWindow(windowName, width, height);

	int x = (screenWidth - WINDOW_OPTIONS_WIDTH) + xOffset;
	int y = ((WINDOW_HEADER_HEIGHT + WINDOW_FRAME_THICKNESS) * trackBarWindowNumber + WINDOW_OPTIONS_TRACKBAR_HEIGHT * cumulativeTrackBarPosition) + yOffset;
	
	moveWindow(windowName, x, y);


	

	return pair<int, int>(x, y);
}


bool ImageAnalysis::outputResults() {	
	return true;
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </OpenCV HighGUI>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
