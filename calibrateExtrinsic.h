#ifndef CVLEVMARQ_H
#define CVLEVMARQ_H

#include <opencv2/opencv.hpp>
using namespace cv;
#include <vector>
using namespace std;

int calibration(std::vector<cv::Mat> &vecImg, std::vector<cv::Mat > &vecPoint,
                vector<vector<float> > vvecPlane_, cv::Size boardSize, float squareSize, Mat &cameraMatrix, Mat &distCoeffs, Mat &rVecL, Mat &tVecL);
#endif // CVLEVMARQ_H
