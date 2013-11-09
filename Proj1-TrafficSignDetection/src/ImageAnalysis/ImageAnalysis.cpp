#include "ImageAnalysis.h"


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  <Image analysis>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
ImageAnalysis::ImageAnalysis() :
	useCVHiGUI(true), windowsInitialized(false),
	frameRate(30), screenWidth(1920), screenHeight(1080),
	claehClipLimit(2), claehTileXSize(2), claehTileYSize(2),
	bilateralFilterDistance(9), bilateralFilterSigmaColor(50), bilateralFilterSigmaSpace(10),
	contrast(11), brightness(25),
	signalColorSegmentationLowerHue(147), signalColorSegmentationUpperHue(7),
	signalColorSegmentationLowerSaturation(112), signalColorSegmentationUpperSaturation(255),
	signalColorSegmentationLowerValue(32), signalColorSegmentationUpperValue(255),
	signalColorSegmentationMorphType(0), signalColorSegmentationMorphKernelSizeX(2), signalColorSegmentationMorphKernelSizeY(2), signalColorSegmentationMorphIterations(1),
	textColorSegmentationLowerHue(20), textColorSegmentationUpperHue(140),
	textColorSegmentationLowerSaturation(0), textColorSegmentationUpperSaturation(255),
	textColorSegmentationLowerValue(0), textColorSegmentationUpperValue(147),
	textColorSegmentationMorphType(1), textColorSegmentationMorphKernelSizeX(1), textColorSegmentationMorphKernelSizeY(1), textColorSegmentationMorphIterations(1),
	textMinMatchPercentage(40), digitRecognitionMethod(DIGIT_RECOGNITION_FEATURE_DETECTION),
	textFeatureDetectionMaxDistancePercentageKeypoint(10), textTemplateMatchMethod(CV_TM_CCORR_NORMED),
	textSkeletonKernelPercentageX(6), textSkeletonKernelPercentageY(6), textSkeletonIterations(1), useSkeletonizationOnDigits(false),
	cannyLowerHysteresisThreshold(100), cannyHigherHysteresisThreshold(200), cannySobelOperatorKernelSize(3),
	houghCirclesDP(1), houghCirclesMinDistanceCenters(2),
	houghCirclesCannyHigherThreshold(200), houghCirclesAccumulatorThreshold(25),
	houghCirclesMinRadius(1), houghCirclesMaxRadius(100) {};


void ImageAnalysis::loadDigitsTemplateImages() {
	for(size_t number = 0; number < 10; ++number) {
		stringstream pathToImage;
		pathToImage << PATH_IMAGES_DIGITS_TEMPLATES << number << ".png";
		Mat digitImageTemplate = imread(pathToImage.str(), CV_LOAD_IMAGE_GRAYSCALE);
		cv::threshold(digitImageTemplate, digitImageTemplate, 0, 255, CV_THRESH_BINARY);
		digitsImagesTemplates.push_back(digitImageTemplate);
	}

	skeletonizeTemplates();
}


ImageAnalysis::~ImageAnalysis() {
	if (useCVHiGUI) {
		cv::destroyAllWindows();
	}
}


bool ImageAnalysis::processImage(string path, bool useCVHighGUI) {		
	Mat imageToProcess;
	bool loadSuccessful = true;
	if (path != "") {
		try {
			imageToProcess = imread(path, CV_LOAD_IMAGE_COLOR);	
		} catch (...) {
			loadSuccessful = false;
		}			

		if (!imageToProcess.data) {
			loadSuccessful = false;
		}
	} else {		
		loadSuccessful = false;
	}

	if (!loadSuccessful) {
		if (useCVHighGUI) {
			cv::destroyAllWindows();
		}

		return false;
	}

	useCVHiGUI = useCVHighGUI;
	windowsInitialized = false;

	loadDigitsTemplateImages();

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


void ImageAnalysis::histogramEqualization(Mat& image, bool useCLAHE, bool useCVHighGUI) {	
	cvtColor(image, image, CV_BGR2YCrCb);
	vector<Mat> channels;
	cv::split(image, channels);

	if (useCLAHE) {
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE((claehClipLimit < 1 ? 1 : claehClipLimit), cv::Size((claehTileXSize < 1 ? 1 : claehTileXSize) , (claehTileYSize < 1? 1 : claehTileYSize)));
		clahe->apply(channels[0], channels[0]);
	} else {
		cv::equalizeHist(channels[0], channels[0]);
	}

	cv::merge(channels, image);
	cvtColor(image, image, CV_YCrCb2BGR);	
	if (useCVHighGUI) {
		if (useCLAHE) {
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

	if (signalColorSegmentationLowerHue < signalColorSegmentationUpperHue) {
		cv::inRange(preprocessedImage,
			Scalar(signalColorSegmentationLowerHue, signalColorSegmentationLowerSaturation, signalColorSegmentationLowerValue),
			Scalar(signalColorSegmentationUpperHue, signalColorSegmentationUpperSaturation, signalColorSegmentationUpperValue),
			colorSegmentation);
	} else {
		// when colors wrap around from near 180 to 0+				
		Mat lowerRange;
		cv::inRange(preprocessedImage,
			Scalar(0, signalColorSegmentationLowerSaturation, signalColorSegmentationLowerValue),
			Scalar(signalColorSegmentationUpperHue, signalColorSegmentationUpperSaturation, signalColorSegmentationUpperValue),
			lowerRange);
	
		Mat higherRange;
		cv::inRange(preprocessedImage,
			Scalar(signalColorSegmentationLowerHue, signalColorSegmentationLowerSaturation, signalColorSegmentationLowerValue),
			Scalar(180, signalColorSegmentationUpperSaturation, signalColorSegmentationUpperValue),
			higherRange);

		cv::bitwise_or(lowerRange, higherRange, colorSegmentation);
	}

	
	// apply morphology operations
	if (signalColorSegmentationMorphKernelSizeX > 0 && signalColorSegmentationMorphKernelSizeY > 0 && signalColorSegmentationMorphIterations > 0) {
		Point anchor(signalColorSegmentationMorphKernelSizeX, signalColorSegmentationMorphKernelSizeY);
		Mat structuringElement = getStructuringElement(cv::MORPH_ELLIPSE,
			cv::Size(2 * signalColorSegmentationMorphKernelSizeX + 1, 2 * signalColorSegmentationMorphKernelSizeY + 1),
			anchor);

		cv::morphologyEx(colorSegmentation, colorSegmentation,
			(signalColorSegmentationMorphType == 0? cv::MORPH_OPEN : cv::MORPH_CLOSE),
			structuringElement, anchor,
			signalColorSegmentationMorphIterations);
	}


	cvtColor(preprocessedImage, preprocessedImage, CV_HSV2BGR);
	if (useCVHighGUI) {
		imshow(WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION, colorSegmentation);
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
	
	retrieveEllipsisFromHoughCircles(colorSegmentedImage, preprocessedImage, houghCirclesFiltered, outputRecognizedEllipsis, useCVHighGUI);
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
			float maxRadius = (std::max)(centerToTest[2], centerToAdd[2]);
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


void ImageAnalysis::retrieveEllipsisFromHoughCircles(const Mat& colorSegmentedImage, Mat& preprocessedImage, const vector<Vec3f>& houghCirclesFiltered, vector<pair<Rect, RotatedRect> >& outputTrafficSignEllipsis, bool useCVHighGUI) {
	Mat colorSegmentedImageContours = colorSegmentedImage.clone();
	int imageWidth = colorSegmentedImage.size().width;
	int imageHeight = colorSegmentedImage.size().height;
	for (size_t currentCirclePos = 0; currentCirclePos < houghCirclesFiltered.size(); ++currentCirclePos) {
		const Vec3f& currentCircle = houghCirclesFiltered[currentCirclePos];
		int currentCircleCenterX = cvRound(currentCircle[0]);
		int currentCircleCenterY = cvRound(currentCircle[1]);
		int currentCircleRadius = cvRound(currentCircle[2]);

		// add offset to compensate errors from hough transform (radius with offset cannot surpass image boundaries)		
		int maxRadiusAllowed = (std::min)((std::min)(cvRound(currentCircle[0]), cvRound(currentCircle[1])), (std::min)(imageWidth - currentCircleCenterX, imageHeight - currentCircleCenterY));
		int radiusRadiusWithOffset = (std::min)(cvRound(currentCircleRadius * PARAM_FIT_ELLIPSIS_SCALE_FOR_HOUGH_CIRCLE), maxRadiusAllowed);
		int roiWidth = (std::min)(radiusRadiusWithOffset * 2, imageWidth);
		int roiHeight = (std::min)(radiusRadiusWithOffset * 2, imageHeight);
		int roiX = (std::max)(currentCircleCenterX - radiusRadiusWithOffset, 0);
		int roiY = (std::max)(currentCircleCenterY - radiusRadiusWithOffset, 0);
		
		// make sure roi is inside image
		roiWidth = (std::min)(roiWidth, imageWidth - roiX);
		roiHeight = (std::min)(roiHeight, imageHeight - roiY);
		
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
					ellipseBoundingRect.x = (std::max)(ellipseBoundingRect.x, 0);
					ellipseBoundingRect.y = (std::max)(ellipseBoundingRect.y, 0);
					ellipseBoundingRect.width = (std::min)(ellipseBoundingRect.width, imageWidth - ellipseBoundingRect.x);
					ellipseBoundingRect.height = (std::min)(ellipseBoundingRect.height, imageHeight - ellipseBoundingRect.y);
					
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


		// apply morphology operations
		if (textColorSegmentationMorphKernelSizeX > 0 && textColorSegmentationMorphKernelSizeY > 0 && textColorSegmentationMorphIterations > 0) {
			Point anchor(textColorSegmentationMorphKernelSizeX, textColorSegmentationMorphKernelSizeY);
			Mat structuringElement = getStructuringElement(cv::MORPH_ELLIPSE,
				cv::Size(2 * textColorSegmentationMorphKernelSizeX + 1, 2 * textColorSegmentationMorphKernelSizeY + 1),
				anchor);

			cv::morphologyEx(textColorSegmentation, textColorSegmentation,
				(textColorSegmentationMorphType == 0? cv::MORPH_OPEN : cv::MORPH_CLOSE),
				structuringElement, anchor,
				textColorSegmentationMorphIterations);
		}		

		if (useCVHighGUI) {
			try {
				ellipseROI.copyTo(imageROIs(ellipseBoundingRect));				
			} catch(...) {}
		}

		int detectedSign = recognizeTrafficSignText(imageROIs, textColorSegmentation, ellipseBoundingRect, useCVHighGUI, ellipsePos);
				
		if (useCVHighGUI) {
			try {				
				textColorSegmentation.copyTo(imageTexts(ellipseBoundingRect));
			} catch(...) {}
		}
		
		if (detectedSign > 0) {
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
			imshow(WINDOW_NAME_TEXT_COLOR_SEGMENTATION, imageTexts);
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


int ImageAnalysis::recognizeTrafficSignText(Mat& preprocessedImage, Mat& textColorSegmentation, const Rect& ellipseBoundingRect, bool useCVHighGUI, size_t currentSignBeingProcessed) {
	vector<vector<Point> > contours;
	int roiX = ellipseBoundingRect.x;
	int roiY = ellipseBoundingRect.y;

	cv::findContours(textColorSegmentation.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(roiX, roiY));

	if (contours.empty()) {
		return -1;
	}

	vector< pair< pair<vector<Point>*, size_t>, Rect> > biggestContours;	

	// extract 3 biggest contours if their area and height is acceptable
	if (contours.size() == 1) {
		biggestContours.push_back(pair< pair<vector<Point>*, size_t>, Rect>(pair<vector<Point>*, size_t>(&contours[0], 0), boundingRect(contours[0])));
	} else if (contours.size() > 1) {
		vector< pair< pair< vector<Point>*, size_t>, double > > contourAreas;
	
		// compute contour area
		for (size_t contourPos = 0; contourPos < contours.size(); ++contourPos) {
			vector<Point>* currentContour = &(contours[contourPos]);
			double currentContourArea = cv::contourArea(*currentContour);
			contourAreas.push_back(pair< pair< vector<Point>*, size_t>, double >(pair< vector<Point>*, size_t>(currentContour, contourPos), currentContourArea));
		}

		sort(contourAreas.begin(), contourAreas.end(), sortContourAreas);

		double ellipseBoundingRectArea = ellipseBoundingRect.width * ellipseBoundingRect.height;
		double minAreaForTrafficSignText = ellipseBoundingRectArea * PARAM_TEXT_MIN_AREA_PERCENTAGE_IN_SIGN;
		int minHeightForTrafficSignText = (int)(ellipseBoundingRect.height * PARAM_TEXT_MIN_HEIGHT_PERCENTAGE_IN_SIGN);
		
		// extract 3 biggest contours
		for (size_t contourAreasPos = 0; contourAreasPos < 3 && contourAreasPos < contourAreas.size(); ++contourAreasPos) {
			pair< pair< vector<Point>*, size_t>, double >& currentBiggestContour = contourAreas[contourAreasPos];
			if (currentBiggestContour.second > minAreaForTrafficSignText) {								
				Rect currentBiggestContourBoundingRect = boundingRect(*(currentBiggestContour.first.first));
				if (currentBiggestContourBoundingRect.height > minHeightForTrafficSignText) {
					biggestContours.push_back(pair< pair<vector<Point>*, size_t>, Rect>(currentBiggestContour.first, currentBiggestContourBoundingRect));
				}
			}
		}
		
		// sort contours by their horizontal position
		sort(biggestContours.begin(), biggestContours.end(), sortContourByHorizontalPosition);
	}
	
	stringstream trafficSignNumberSS;
	int numberOfDigits = 0;
	for (size_t biggestContoursPos = 0; biggestContoursPos < biggestContours.size(); ++biggestContoursPos) {				
		// adjust roi offsets from preprocessedImage to textColorSegmentation
		Rect digitRect = biggestContours[biggestContoursPos].second;
		digitRect.x = (std::max)(digitRect.x - roiX, 0);
		digitRect.y = (std::max)(digitRect.y - roiY, 0);
		Mat textColorSegmentationDigitROI = textColorSegmentation(digitRect);

		Mat feacturePointsGoodMatches;		
		int recognizedDigit = recognizeDigit(textColorSegmentationDigitROI, feacturePointsGoodMatches, useSkeletonizationOnDigits, digitRecognitionMethod, useCVHighGUI);		
		if (recognizedDigit >= 0 && recognizedDigit < 10) {
			trafficSignNumberSS << recognizedDigit;
			++numberOfDigits;
		}

		if (digitRecognitionMethod == DIGIT_RECOGNITION_FEATURE_DETECTION && useCVHighGUI) {
			stringstream windowNameSS;
			windowNameSS << WINDOW_NAME_DIGITS_FEATURE_POINTS(currentSignBeingProcessed, biggestContoursPos);
			string windowName = windowNameSS.str();
			namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			//moveWindow(windowName, 0, 0);
			if (feacturePointsGoodMatches.data && feacturePointsGoodMatches.size().width > 0 && feacturePointsGoodMatches.size().height > 0) {
				imshow(windowName, feacturePointsGoodMatches);
			}
		}


		if (useCVHighGUI) {
			try {
				rectangle(preprocessedImage, biggestContours[biggestContoursPos].second, COLOR_TEXT_HSV, 2);
				drawContours(preprocessedImage, contours, biggestContours[biggestContoursPos].first.second, COLOR_TEXT_HSV, 2);
			} catch(...) {}
		}
	}
	
	if (numberOfDigits > 0) {
		int trafficSignNumber = -1;
		trafficSignNumberSS >> trafficSignNumber;
		if (trafficSignNumber >= std::pow(10, (numberOfDigits - 1))) {
			return trafficSignNumber;
		}
	}

	return -1;
}


int ImageAnalysis::recognizeDigit(Mat& textColorSegmentationDigitROI, Mat& feacturePointsGoodMatches, bool useSkeletonization, int numberRecognitionMethod, bool useCVHighGUI) {
	size_t bestMatchDigit = 0;
	float bestMatch = 0;
	Mat bestMatchFeacturePointsGoodMatches;

	int roiWidth = textColorSegmentationDigitROI.size().width;
	int roiHeight = textColorSegmentationDigitROI.size().height;
	int roiKernelSizeX = (std::max)(roiWidth * textSkeletonKernelPercentageX / 100, 3);
	int roiKernelSizeY = (std::max)(roiHeight * textSkeletonKernelPercentageY / 100, 3);
	Mat roiSkeletonizationStructuringElement = getStructuringElement(cv::MORPH_ELLIPSE, Size(roiKernelSizeX, roiKernelSizeY));

	if (useSkeletonization) {
		textColorSegmentationDigitROI = textSkeletonization(textColorSegmentationDigitROI, roiSkeletonizationStructuringElement, textSkeletonIterations);
	} else {
		for(int iterNumber = 0; iterNumber < textSkeletonIterations; ++iterNumber) {
			erode(textColorSegmentationDigitROI, textColorSegmentationDigitROI, roiSkeletonizationStructuringElement);
		}
	}

#ifdef USE_TESSERACT
	if (numberRecognitionMethod == DIGIT_RECOGNITION_TESSERACT) {
		return recognizeDigitWithTesseract(textColorSegmentationDigitROI);
	}
#endif

	for(size_t imageNumber = 0; imageNumber < digitsImagesTemplates.size(); ++imageNumber) {								
		float matchResult;

		Mat image;
		Mat digitTemplate;
		int digitTemplateWidth = digitsImagesTemplatesSkeletons[imageNumber].size().width;
		int digitTemplateHeight = digitsImagesTemplatesSkeletons[imageNumber].size().height;		

		if (textColorSegmentationDigitROI.size().width > digitTemplateWidth) {
			cv::resize(textColorSegmentationDigitROI, image, Size(digitTemplateWidth, digitTemplateHeight));
			digitTemplate = digitsImagesTemplatesSkeletons[imageNumber];
		} else {
			image = textColorSegmentationDigitROI;
			cv::resize(digitsImagesTemplatesSkeletons[imageNumber].clone(), digitTemplate, Size(textColorSegmentationDigitROI.size().width, textColorSegmentationDigitROI.size().height));
		}		

		Mat feacturePointsGoodMatchesTemp;
		if (numberRecognitionMethod == DIGIT_RECOGNITION_FEATURE_DETECTION) {		
			matchResult = recognizeDigitWithFeatureMatching(image, digitTemplate, feacturePointsGoodMatchesTemp, useCVHighGUI);
		} else {
			Mat result(1, 1, CV_32FC1);
			cv::matchTemplate(image, digitTemplate, result, textTemplateMatchMethod);
			matchResult = result.at<float>(0,0);
		}

		if (matchResult > bestMatch) {
			bestMatch = matchResult;
			bestMatchDigit = imageNumber;
			feacturePointsGoodMatches = feacturePointsGoodMatchesTemp;
		}
	}

	if (bestMatch * 100 > (float)textMinMatchPercentage) {
		return (int)bestMatchDigit;
	} else {
		return -1;
	}
}


float ImageAnalysis::recognizeDigitWithFeatureMatching(Mat& textColorSegmentationDigitROI, Mat& digitImageTemplate, Mat& feacturePointsGoodMatches, bool useCVHighGUI) {		
	//-- Step 1: Detect the keypoints
	//cv::SurfFeatureDetector detector(300);
	//cv::SiftFeatureDetector detector;
	//cv::FastFeatureDetector detector(15);
	cv::GoodFeaturesToTrackDetector detector;
	//cv::OrbFeatureDetector detector;
	//cv::MserFeatureDetector detector;
	//cv::StarFeatureDetector detector;	

	vector<cv::KeyPoint> keypointsDigitROI, keypointsDigitTemplate;
	detector.detect(textColorSegmentationDigitROI, keypointsDigitROI);
	detector.detect(digitImageTemplate, keypointsDigitTemplate);


	//-- Step 2: Calculate descriptors (feature vectors)
	//cv::SurfDescriptorExtractor extractor;
	cv::SiftDescriptorExtractor extractor;
	//cv::BriefDescriptorExtractor extractor;
	//cv::FREAK extractor;
	//cv::BRISK extractor;

	Mat descriptorsDigitROI, descriptorsDigitTemplate;
	extractor.compute(textColorSegmentationDigitROI, keypointsDigitROI, descriptorsDigitROI);
	extractor.compute(digitImageTemplate, keypointsDigitTemplate, descriptorsDigitTemplate);
	
	if (descriptorsDigitROI.empty() || descriptorsDigitTemplate.empty()) {
		return 0;
	}


	//-- Step 3: Matching descriptor vectors using FLANN matcher		
	cv::FlannBasedMatcher matcher;
	//cv::BFMatcher matcher(cv::NORM_L2);

	vector<cv::DMatch> matches;
	vector<cv::DMatch> goodMatches;	


	// 3.1
	matcher.match(descriptorsDigitROI, descriptorsDigitTemplate, matches);
	if (matches.empty()) {
		return 0;
	}
	
	double maxDistanceBetweenFeatureMatches = (double)textFeatureDetectionMaxDistancePercentageKeypoint * (double)digitImageTemplate.size().width / (double)100.0;	
	for(size_t matchPos = 0; matchPos < matches.size(); ++matchPos) {
		Point imagePoint = keypointsDigitROI[matches[matchPos].queryIdx].pt;
		Point templatePoint = keypointsDigitTemplate[matches[matchPos].trainIdx].pt;
		int dx = imagePoint.x - templatePoint.x;
		int dy = imagePoint.y - templatePoint.y;
		double distanceBetweenPoints = (double)std::sqrt(dx*dx + dy*dy);

		if(distanceBetweenPoints <= maxDistanceBetweenFeatureMatches) {
			goodMatches.push_back(matches[matchPos]);
		}
	}


	// 3.2
	//matcher.match(descriptorsDigitROI, descriptorsDigitTemplate, matches);
	//if (matches.empty()) {
	//	return 0;
	//}

	//float minDist = matches[0].distance;
	//float maxDist = matches[0].distance;
	//for(size_t matchPos = 1; matchPos < matches.size(); ++matchPos) {
	//	float dist = matches[matchPos].distance;
	//	if (dist < minDist) minDist = dist;
	//	if (dist > maxDist) maxDist = dist;
	//}

	//textFeatureDetectionMaxDistancePercentageKeypoint = 20;
	////float maxDistanceBetweenFeatureMatches = (float)textFeatureDetectionMaxDistancePercentageKeypoint * (float)digitImageTemplate.size().width / (float)100.0;
	//float maxDistanceBetweenFeatureMatches = minDist + ((float)(textFeatureDetectionMaxDistancePercentageKeypoint * (maxDist - minDist)) / (float)100.0);
	////float maxDistanceBetweenFeatureMatches = 2 * minDist;
	//for(size_t matchPos = 0; matchPos < matches.size(); ++matchPos) {
	//	if(matches[matchPos].distance <= maxDistanceBetweenFeatureMatches) {
	//		goodMatches.push_back(matches[matchPos]);
	//	}
	//}


	// 3.3
	//vector< vector<cv::DMatch> > matchesKNN;
	//float nnDistanceRatio = 0.8f;
	//matcher.knnMatch(descriptorsDigitROI, descriptorsDigitTemplate, matchesKNN, 2);
	////matcher.radiusMatch(descriptorsDigitROI, descriptorsDigitTemplate, matchesKNN, 2);
	//for(size_t matchPos = 0; matchPos < matchesKNN.size(); ++matchPos) {
	//	matches.push_back(matchesKNN[matchPos][0]);		

	//	if(matchesKNN[matchPos][0].distance <= nnDistanceRatio * matchesKNN[matchPos][1].distance) {
	//		goodMatches.push_back(matchesKNN[matchPos][0]);
	//	}
	//}
		


	if (useCVHighGUI) {
		/*Mat imgMatches;		
		cv::drawKeypoints(textColorSegmentationDigitROI, keypointsDigitROI, textColorSegmentationDigitROI);
		cv::drawMatches(
			textColorSegmentationDigitROI, keypointsDigitROI, digitImageTemplate, keypointsDigitTemplate,
			matches, imgMatches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		imshow(WINDOW_NAME_FEATURE_MATCHES, imgMatches);*/

		cv::drawMatches(
			textColorSegmentationDigitROI, keypointsDigitROI, digitImageTemplate, keypointsDigitTemplate,
			goodMatches, feacturePointsGoodMatches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		
		//imshow(WINDOW_NAME_FEATURE_GOOD_MATCHES, feacturePointsGoodMatches);
	}

	return (float)goodMatches.size() / (float)matches.size();
}

#ifdef USE_TESSERACT
int ImageAnalysis::recognizeDigitWithTesseract(Mat& textColorSegmentationDigitROI) {
	tesseract::TessBaseAPI tess;
	//tesseract::TessBaseAPI::SetVariable("tessedit_char_whitelist", "0123456789");
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
	tess.SetVariable("tessedit_char_whitelist", "0123456789");
	tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
	tess.SetImage((uchar*)textColorSegmentationDigitROI.data, textColorSegmentationDigitROI.cols, textColorSegmentationDigitROI.rows, 1, textColorSegmentationDigitROI.cols);

	char* out = tess.GetUTF8Text();
	if (out == NULL) {
		return -1;
	}

	stringstream ss;
	ss << out;
	int number = -1;
	ss >> number;

	tess.Clear();
	tess.End();

	delete[] out;

	return number;
}
#endif

Mat ImageAnalysis::textSkeletonization(Mat& image, Mat& kernel, int numberIterations) {
	if (numberIterations < 1) {
		return image.clone();
	}
	
	Mat skel(image.size(), CV_8UC1, cv::Scalar(0));
	Mat erodedImage;	
	Mat temp;
	
	for(int iterNumber = 0; iterNumber < numberIterations; ++iterNumber) {
		erode(image, erodedImage, kernel);
		dilate(erodedImage, temp, kernel);
		subtract(image, temp, temp);
		bitwise_or(skel, temp, skel);
		erodedImage.copyTo(image);
	}

	return skel;
}


void ImageAnalysis::skeletonizeTemplates() {
	digitsImagesTemplatesSkeletons.clear();
	for(size_t imageNumber = 0; imageNumber < digitsImagesTemplates.size(); ++imageNumber) {
		Mat& templateImage = digitsImagesTemplates[imageNumber];
		int templateKernelSizeX = (std::max)(templateImage.size().width * textSkeletonKernelPercentageX / 100, 3);
		int templateKernelSizeY = (std::max)(templateImage.size().height * textSkeletonKernelPercentageY / 100, 3);
		Mat templateSkeletonizationStructuringElement = getStructuringElement(cv::MORPH_ELLIPSE, Size(templateKernelSizeX, templateKernelSizeY));
		Mat digitTemplateSkeleton;
		if (useSkeletonizationOnDigits) {
			digitTemplateSkeleton = textSkeletonization(templateImage.clone(), templateSkeletonizationStructuringElement, textSkeletonIterations);
		} else {
			digitTemplateSkeleton = templateImage.clone();
			for(int iterNumber = 0; iterNumber < textSkeletonIterations; ++iterNumber) {
				erode(digitTemplateSkeleton, digitTemplateSkeleton, templateSkeletonizationStructuringElement);
			}
		}
		digitsImagesTemplatesSkeletons.push_back(digitTemplateSkeleton);		
	}

	if (useCVHiGUI) {
		int windowXPos = 0;
		int windowYPos = screenHeight - WINDOW_DIGITS_HEIGHT;
		
		int digitsWindowSpace = screenWidth - WINDOW_OPTIONS_WIDTH;
		int digitWidth = digitsWindowSpace / 10;
		int digitHeight = WINDOW_DIGITS_HEIGHT;
		
		if (digitHeight * 10 * 0.7 > digitsWindowSpace) {
			digitHeight = (int)(digitsWindowSpace / (10 * 0.7));
		}

		for(size_t imageNumber = 0; imageNumber < digitsImagesTemplates.size(); ++imageNumber) {
			Mat& templateImageSkeleton = digitsImagesTemplatesSkeletons[imageNumber];
			
			stringstream ss;
			ss << WINDOW_NAME_DIGITS_SKELETON(imageNumber);
			string windowName = ss.str();

			if (!windowsInitialized) {				
				namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			}
			
			int windowHeight = digitHeight - WINDOW_FRAME_THICKNESS - WINDOW_HEADER_HEIGHT;
			int windowWidth = templateImageSkeleton.size().width * digitHeight / templateImageSkeleton.size().height;
						
			resizeWindow(windowName, windowWidth, windowHeight);
			moveWindow(windowName, windowXPos, windowYPos);			
			imshow(windowName, templateImageSkeleton);

			windowXPos += windowWidth + 2 * WINDOW_FRAME_THICKNESS;
		}
	}
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

	loadDigitsTemplateImages();

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


void updateImageAnalysisAndTemplates(int position, void* userData) {		
	ImageAnalysis* imgAnalysis = ((ImageAnalysis*)userData);
	imgAnalysis->skeletonizeTemplates();
	imgAnalysis->updateImage();
}


void ImageAnalysis::drawTrafficSignLabel(string text, Mat& image, const Rect& signBoundingRect) {
	int textBoxHeight = (int)(signBoundingRect.height * 0.15);
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = (double)textBoxHeight / 46.0;
	int thickness = (std::max)(1, (int)(textBoxHeight * 0.05));
	int baseline = 0;

	Rect textBoundingRect = signBoundingRect;
	textBoundingRect.height = (std::max)(textBoxHeight, TEXT_MIN_SIZE);
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
	addHighGUIWindow(0, 1, WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION);
	//addHighGUIWindow(1, 1, WINDOW_NAME_SIGNAL_CANNY);
	addHighGUIWindow(1, 1, WINDOW_NAME_SIGNAL_RECOGNITION);
	addHighGUIWindow(2, 1, WINDOW_NAME_TEXT_COLOR_SEGMENTATION);
	addHighGUIWindow(3, 1, WINDOW_NAME_SIGNAL_ROI);	
	
	if (optionsOneWindow) {		
		namedWindow(WINDOW_NAME_OPTIONS, CV_WINDOW_NORMAL);
		resizeWindow(WINDOW_NAME_OPTIONS, WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2, WINDOW_OPTIONS_HIGHT);
		moveWindow(WINDOW_NAME_OPTIONS, screenWidth - WINDOW_OPTIONS_WIDTH, 0);
	} else {						
		addHighGUITrackBarWindow(WINDOW_NAME_BILATERAL_FILTER_OPTIONS, 3, 0, 0);
		addHighGUITrackBarWindow(WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS, 2, 3, 1, 0, WINDOW_HEADER_HEIGHT);
		addHighGUITrackBarWindow(WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS, 3, 3, 1, 2 * WINDOW_FRAME_THICKNESS);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS, 4, 6, 2, 0, 0);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS, 6, 6, 2, 2 * WINDOW_FRAME_THICKNESS);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS, 6, 10, 3, 2 * WINDOW_FRAME_THICKNESS, 0);
		addHighGUITrackBarWindow(WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS, 6, 10, 3, 0, WINDOW_HEADER_HEIGHT);
		addHighGUITrackBarWindow(WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS, 10, 10, 3, 2 * WINDOW_FRAME_THICKNESS, WINDOW_HEADER_HEIGHT);
		/*addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_CANNY_OPTIONS, 3, 14, 4);
		addHighGUITrackBarWindow(WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS, 6, 17, 5);*/	
	}	
	
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_DIST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterDistance, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_COLOR_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaColor, 200, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BI_FILTER_SPACE_SIG, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_BILATERAL_FILTER_OPTIONS), &bilateralFilterSigmaSpace, 200, updateImageAnalysis, (void*)this);
	
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_CLIP, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehClipLimit, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileXSize, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CLAHE_TILE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_HISTOGRAM_EQUALIZATION_CLAHE_OPTIONS), &claehTileYSize, 20, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_CONTRAST, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &contrast, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_BRIGHTNESS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_CONTRAST_AND_BRIGHTNESS_OPTIONS), &brightness, 1000, updateImageAnalysis, (void*)this);
		
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationLowerHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationUpperHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationLowerSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationUpperSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MIN_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationLowerValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_COLOR_SEG_MAX_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_COLOR_SEGMENTATION_OPTIONS), &signalColorSegmentationUpperValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_MORPH_OPERATOR, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS), &signalColorSegmentationMorphType, 1, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_MORPH_KERNEL_SIZE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS), &signalColorSegmentationMorphKernelSizeX, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_MORPH_KERNEL_SIZE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS), &signalColorSegmentationMorphKernelSizeY, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_SIGNAL_MORPH_ITERATIONS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_MORPHOLOGY_OPERATORS_OPTIONS), &signalColorSegmentationMorphIterations, 20, updateImageAnalysis, (void*)this);

	/*cv::createTrackbar(TRACK_BAR_NAME_CANNY_LOWER_HYSTERESIS_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannyLowerHysteresisThreshold, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CANNY_HIGHER_HYSTERESIS_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannyHigherHysteresisThreshold, 300, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_CANNY_KERNEL_SIZE_SOBEL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_CANNY_OPTIONS), &cannySobelOperatorKernelSize, 7, updateImageAnalysis, (void*)this);*/

	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_DP, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesDP, 10, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_DIST_CENTERS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMinDistanceCenters, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_CANNY_HIGHER_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesCannyHigherThreshold, 500, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_ACCUMULATOR_THRESHOLD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesAccumulatorThreshold, 250, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MIN_RADIUS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMinRadius, 100, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_HOUGH_CIRCLES_MAX_RADIUS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_SIGNAL_RECOGNITION_OPTIONS), &houghCirclesMaxRadius, 100, updateImageAnalysis, (void*)this);

	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationLowerHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_HUE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationUpperHue, 180, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationLowerSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_SAT, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationUpperSaturation, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MIN_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationLowerValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_COLOR_SEG_MAX_VAL, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_COLOR_SEGMENTATION_OPTIONS), &textColorSegmentationUpperValue, 255, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_MORPH_OPERATOR, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textColorSegmentationMorphType, 1, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_MORPH_KERNEL_SIZE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textColorSegmentationMorphKernelSizeX, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_MORPH_KERNEL_SIZE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textColorSegmentationMorphKernelSizeY, 20, updateImageAnalysis, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_MORPH_ITERATIONS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textColorSegmentationMorphIterations, 20, updateImageAnalysis, (void*)this);	
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_SKELETONIZATION_KERNEL_SIZE_X, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textSkeletonKernelPercentageX, 100, updateImageAnalysisAndTemplates, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_SKELETONIZATION_KERNEL_SIZE_Y, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textSkeletonKernelPercentageY, 100, updateImageAnalysisAndTemplates, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_SKELETONIZATION_ITERATIONS, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textSkeletonIterations, 20, updateImageAnalysisAndTemplates, (void*)this);
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_MIN_MATCH_PERCENTAGE, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textMinMatchPercentage, 100, updateImageAnalysis, (void*)this);
#ifdef USE_TESSERACT
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_DIGIT_RECOGNITION_METHOD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &digitRecognitionMethod, 2, updateImageAnalysis, (void*)this);
#else
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_DIGIT_RECOGNITION_METHOD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &digitRecognitionMethod, 1, updateImageAnalysis, (void*)this);
#endif		
	cv::createTrackbar(TRACK_BAR_NAME_TEXT_TEMPLATE_MATCH_METHOD, (optionsOneWindow? WINDOW_NAME_OPTIONS : WINDOW_NAME_TEXT_MORPHOLOGY_OPERATORS_OPTIONS), &textTemplateMatchMethod, 5, updateImageAnalysis, (void*)this);	
}


pair< pair<int, int>, pair<int, int> > ImageAnalysis::addHighGUIWindow(int column, int row, string windowName, int numberColumns, int numberRows, int xOffset, int yOffset, int windowWidth, int windowHeight, int imageWidth, int imageHeight) {
	if (numberColumns < 1 || numberRows < 1)
		return pair< pair<int, int>, pair<int, int> >(pair<int, int>(0, 0), pair<int, int>(0, 0));
	
	int imageWidthFinal = (imageWidth > 0? imageWidth : originalImage.size().width);
	if (imageWidthFinal < 10)
		imageWidthFinal = (screenWidth - WINDOW_OPTIONS_WIDTH) / 2;

	int imageHeightFinal = (imageHeight > 0? imageHeight : originalImage.size().height);
	if (imageHeightFinal < 10)
		imageHeightFinal = (screenHeight - WINDOW_DIGITS_HEIGHT)/ 2;	


	int windowHeightFinal = (windowHeight > 0? windowHeight : ((screenHeight - WINDOW_DIGITS_HEIGHT) / numberRows));
	int windowWidthFinal = (windowWidth > 0? windowWidth : (imageWidthFinal * windowHeightFinal / imageHeightFinal));

	if ((windowWidthFinal * numberColumns + WINDOW_OPTIONS_WIDTH) > screenWidth) {		
		windowWidthFinal = ((screenWidth - WINDOW_OPTIONS_WIDTH) / numberColumns);
		windowHeightFinal = imageHeightFinal * windowWidthFinal / imageWidthFinal;
	}

	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
	resizeWindow(windowName, windowWidthFinal - 2 * WINDOW_FRAME_THICKNESS, windowHeightFinal - WINDOW_FRAME_THICKNESS - WINDOW_HEADER_HEIGHT);
	
	int x = 0;
	if (column != 0) {
		x = windowWidthFinal * column;
	}

	int y = 0;
	if (row != 0) {
		y = windowHeightFinal * row;
	}

	x += xOffset;
	y += yOffset;
	
	moveWindow(windowName, x, y);

	return pair< pair<int, int>, pair<int, int> >(pair<int, int>(x, y), pair<int, int>(windowWidthFinal, windowHeightFinal));
}


pair< pair<int, int>, pair<int, int> > ImageAnalysis::addHighGUITrackBarWindow(string windowName, int numberTrackBars, int cumulativeTrackBarPosition, int trackBarWindowNumber, int xOffset, int yOffset) {
	namedWindow(windowName, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);

	int width = WINDOW_OPTIONS_WIDTH - WINDOW_FRAME_THICKNESS * 2;
	int height = numberTrackBars * WINDOW_OPTIONS_TRACKBAR_HEIGHT;
	resizeWindow(windowName, width, height);

	int x = (screenWidth - WINDOW_OPTIONS_WIDTH) + xOffset;
	int y = ((WINDOW_HEADER_HEIGHT + WINDOW_FRAME_THICKNESS) * trackBarWindowNumber + WINDOW_OPTIONS_TRACKBAR_HEIGHT * cumulativeTrackBarPosition) + yOffset;
	
	moveWindow(windowName, x, y);

	return pair< pair<int, int>, pair<int, int> >(pair<int, int>(x, y), pair<int, int>(width, height));
}
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  </OpenCV HighGUI>  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
