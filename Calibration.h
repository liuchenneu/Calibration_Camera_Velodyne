#ifndef CALIBRATION_H
#define CALIBRATION_H
//Qt
#include <QMainWindow>
#include <QLabel>
#include <QVTKWidget.h>
//STL
#include <vector>
#include <iostream>
//opencv
#include <opencv2/opencv.hpp>
//PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h> //fps calculations
#include <pcl/io/pcd_io.h>
#include <pcl/io/hdl_grabber.h>
#include <pcl/io/vlp_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/boost.h>
#include <pcl/visualization/mouse_event.h>
//VTK
#include <vtkRenderWindow.h>
//Boost
#include <boost/function/function0.hpp>
#include <boost/thread.hpp>
//
#include "pointgreycamera.h"
#include "simplevlpviewer.hpp"


namespace Ui {
class Calibration;
}

class Calibration : public QMainWindow
{
    Q_OBJECT
public:
    explicit Calibration(int argc,char** argv,QWidget *parent = 0);
    ~Calibration();
private:
    void pickCallback(const pcl::visualization::PointPickingEvent& event);
    void cloudCallback (const PointCloud<PointXYZI>::ConstPtr& cloud);
    int processImageAndPointcloud(cv::Mat &image,pcl::PointCloud<PointXYZI>::ConstPtr pointcloud);
    void imgThreadFun();

private:
    Ui::Calibration *ui;
    PointGreyCamera *camera;

    bool updateWindowFlag;
    int imgHeight;
    int imgWidth;
    QImage img;

    vector<cv::Mat> rvecs;
    vector<cv::Mat> tvecs;
    vector<float> reprojErrs;

    boost::mutex cloud_mutex_;
    pcl::PointCloud<PointXYZI>::ConstPtr cloud_;
    vector<pcl::PointCloud<pcl::PointXYZI> >vecPointCloud;
    boost::function<void(const PointCloud<PointXYZI>::ConstPtr&)> cloud_cb;
    boost::signals2::connection cloud_connection;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    vector<boost::shared_ptr<QVTKWidget> > vecVtkViewer;
    vector<boost::shared_ptr<pcl::visualization::PCLVisualizer> >vecPCLViewer;
    PointCloudColorHandlerGenericField<PointXYZI> *color_handler;
    VLPGrabber grabber;
    vector<vector<float> >vvecPlane;
    vector<cv::Mat >vecLaserPoint;
    vector<cv::Mat >vecImg;

    boost::function0<void> imgFun;
    boost::thread imgThread;

signals:
    void updateRealtimeImage_();
    void updateRealtimePointCloud_();

private slots:
    void updateRealtimeImage();
    void updateRealtimePointCloud();
    void on_btnCapture_clicked();
    void on_btnCalibration_clicked();
    void on_radioButtonFixAspectRatio_clicked();
    void on_btnLoad_clicked();
};

#endif // CALIBRATION_H
