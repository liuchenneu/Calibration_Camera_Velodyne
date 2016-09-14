#ifndef POINTGREYCAMERA_H
#define POINTGREYCAMERA_H


#include <iostream>
#include <sstream>
using namespace std;
#include <flycapture/FlyCapture2.h>
using namespace FlyCapture2;

class PointGreyCamera
{
public:
    PointGreyCamera();
    ~PointGreyCamera();
    void printBuildInfo();
    void printCameraInfo(CameraInfo* pCamInfo);
    void printFormat7Capabilities(Format7Info fmt7Info);
    int setFrame(float fps=10.0);
    int setImageAttr(int width=640,int heigth=480);
    int init();
    int getAnImage();
private:
    void printError(FlyCapture2::Error error);

public:
FlyCapture2::Image img;

private:
int width;
int heigth;
float fps;
FlyCapture2::Error error;
FlyCapture2::BusManager busMgr;
unsigned int numCameras;
FlyCapture2::PGRGuid guid;
FlyCapture2::Camera cam;
};

#endif // POINTGREYCAMERA_H
