#include "Calibration.h"
#include "ui_Calibration.h"
#include "calibrateExtrinsic.h"
#include <boost/bind.hpp>

#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <sstream>

#include <Eigen/Eigen>
#include <Eigen/SVD>
#include <Eigen/Dense>

Calibration::Calibration(int argc,char** argv,QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Calibration),
    updateWindowFlag(1)
{
    ui->setupUi(this);
    ui->doubleSpinBoxAspectRatio->setVisible(false);
    viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
    ui->qvtkRealtimePointcloud->SetRenderWindow (viewer->getRenderWindow());
    viewer->setupInteractor (ui->qvtkRealtimePointcloud->GetInteractor (), ui->qvtkRealtimePointcloud->GetRenderWindow ());

    boost::function<void (const pcl::visualization::PointPickingEvent&)> pickFun;
    pickFun=boost::bind(&Calibration::pickCallback,this,_1);
    viewer->registerPointPickingCallback(pickFun);
    ui->qvtkRealtimePointcloud->update ();
    color_handler = new PointCloudColorHandlerGenericField<PointXYZI> ("intensity");

    cloud_cb = boost::bind (&Calibration::cloudCallback, this, _1);
    cloud_connection = grabber.registerCallback (cloud_cb);

    viewer->addCoordinateSystem (1.0, "global");
    grabber.start();

    camera=new PointGreyCamera;
    if(0!=camera->init())
    {
        return;
    };
    imgFun=boost::bind(&Calibration::imgThreadFun,this);
    imgThread=boost::thread(imgFun);
    connect(this,SIGNAL(updateRealtimeImage_()),this,SLOT(updateRealtimeImage()));
    connect(this,SIGNAL(updateRealtimePointCloud_()),this,SLOT(updateRealtimePointCloud()));
}

Calibration::~Calibration()
{
    delete camera;
    delete ui;
}

void Calibration::pickCallback(const PointPickingEvent &event)
{
    float x,y,z;
    event.getPoint(x,y,z);
    pcl::PointXYZ origin(x,y,z);
    int index=ui->tabPointCloud->currentIndex()-1;
    if(index<0)
        return;

    double threshold=ui->doubleSpinBoxCalibrationboardSquareSize->value()/1000.0
            *min(ui->spinBoxCalibrationboardHeight->value(),ui->spinBoxCalibrationBoardWidth->value());

    pcl::PointCloud<pcl::PointXYZI>::Ptr board(new pcl::PointCloud<pcl::PointXYZI>);
    for(PointCloud<PointXYZI>::const_iterator it=vecPointCloud.at(index).begin();it!=vecPointCloud.at(index).end();++it)
    {
        if(sqrt(pow((x-it->x),2)+pow((y-it->y),2)+pow((z-it->z),2))<threshold)
        {
            board->push_back(*it);
        }
    }

    if(board->size()==0)
        return;

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.03);
    seg.setInputCloud (board);
    seg.segment (*inliers, *coefficients);

    stringstream name;
    name<<"plane"<<index;
    vecPCLViewer.at(index)->removeShape(name.str());
    vecPCLViewer.at(index)->addPlane(*coefficients,name.str());

    pcl::PointCloud<pcl::PointXYZI>::Ptr pOnPlaneCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for(int i=0;i<inliers->indices.size();i++)
    {
        pOnPlaneCloud->push_back(board->points[inliers->indices[i]]);
    }
    stringstream pcdfile;
    pcdfile<<index<<"_.pcd";
    pcl::io::savePCDFile(pcdfile.str().c_str(),*pOnPlaneCloud);

    ofstream plane(name.str().c_str(),ios::binary);
    double *point=new double[3*inliers->indices.size()];
    for(int i=0;i<inliers->indices.size();i++)
    {
        point[3*i]=board->points[inliers->indices[i]].x;
        point[3*i+1]=board->points[inliers->indices[i]].y;
        point[3*i+2]=board->points[inliers->indices[i]].z;
    }
    Mat laserPoint=Mat(inliers->indices.size(),3,CV_64F,point).t();
    laserPoint=1000*laserPoint;
    Mat lastrow=Mat::ones(1,laserPoint.cols,CV_64F);
    laserPoint.push_back(lastrow);
    if(index+1>vecLaserPoint.size())
    {
        vecLaserPoint.resize(index+1);
        vvecPlane.resize(index+1);
    }
    laserPoint.copyTo(vecLaserPoint[index]);

    vvecPlane[index].resize(4);
    vvecPlane[index][0]=coefficients->values[0];
    vvecPlane[index][1]=coefficients->values[1];
    vvecPlane[index][2]=coefficients->values[2];
    vvecPlane[index][3]=coefficients->values[3]*1000.0;
}

void Calibration::cloudCallback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud)
{
    boost::mutex::scoped_lock lock (cloud_mutex_);
    cloud_ = cloud;
    if(updateWindowFlag)
        emit updateRealtimePointCloud_();
}

int Calibration::processImageAndPointcloud(cv::Mat &image, PointCloud<pcl::PointXYZI>::ConstPtr pointcloud)
{
    static int count=0;
    cv::Mat grayImg;
    if(image.type()==CV_8UC3)
    {
        cv::cvtColor(image,grayImg,CV_BGR2GRAY);
    }
    else if(image.type()==CV_8UC1)
    {
        grayImg=image;
    }
    else
    {
        cerr<<"can not detected original image formate!"<<endl;
        return -1;
    }
    bool found;
    cv::Size boardSize(ui->spinBoxCalibrationboardHeight->value(),ui->spinBoxCalibrationBoardWidth->value());
    vector<cv::Point2f> pointbuf;
    if(ui->rdBtnChessBoard->isChecked())
        found=findChessboardCorners( grayImg, boardSize, pointbuf,
                                     CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
    if(ui->rdBtnCircleGrid->isChecked())
        found = findCirclesGrid(grayImg, boardSize, pointbuf );

    if(!found)
    {
        cerr<<"found no chessboard or circlegrid"<<endl;
        return -1;
    }
    vecImg.push_back(image);

    QLabel *lblImg=new QLabel;

    cv::Mat rgbImage;
    cv::cvtColor(image,rgbImage,CV_BGR2RGB);

    QImage tmpImg=QImage((const unsigned char*)(rgbImage.data),rgbImage.cols,rgbImage.rows,QImage::Format_RGB888);
    QImage scaledImg=tmpImg.scaled(ui->realTimeDisplay->width(),ui->realTimeDisplay->height());
    lblImg->setPixmap(QPixmap::fromImage(scaledImg));
    stringstream tabname;
    tabname<<count;
    ui->tabImg->addTab(lblImg,tabname.str().c_str());

    if(ui->rdBtnCamVel->isChecked())
    {
        boost::shared_ptr<QVTKWidget> vtkViewer;
        boost::shared_ptr<pcl::visualization::PCLVisualizer> newViewer;
        vtkViewer.reset(new QVTKWidget());
        newViewer.reset (new pcl::visualization::PCLVisualizer (tabname.str().c_str(), false));
        vtkViewer->SetRenderWindow (newViewer->getRenderWindow());
        newViewer->setupInteractor (vtkViewer->GetInteractor (),vtkViewer->GetRenderWindow ());

        boost::function<void (const pcl::visualization::PointPickingEvent&)> pickFun;
        pickFun=boost::bind(&Calibration::pickCallback,this,_1);
        newViewer->registerPointPickingCallback(pickFun);
        ui->tabPointCloud->addTab(vtkViewer.get(),tabname.str().c_str());
        newViewer->addCoordinateSystem (1.0, "global");

        color_handler->setInputCloud(pointcloud);
        if (!newViewer->updatePointCloud (pointcloud, *color_handler, "cloud"))
            newViewer->addPointCloud (pointcloud, *color_handler, "cloud");
        vtkViewer->update ();
        vecVtkViewer.push_back(vtkViewer);
        vecPCLViewer.push_back(newViewer);
        pcl::PointCloud<pcl::PointXYZI> tmpcloud;
        tmpcloud=*pointcloud;
        vecPointCloud.push_back(tmpcloud);
    }
    ++count;
    return 0;
}

void Calibration::updateRealtimeImage()
{
    QImage scaledImg;
    scaledImg=img.scaled(ui->realTimeDisplay->width(),ui->realTimeDisplay->height());
    ui->lblRealtimeImg->setPixmap(QPixmap::fromImage(scaledImg));
}

pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud;
void Calibration::updateRealtimePointCloud()
{
    if (cloud_mutex_.try_lock ())
    {
        cloud_.swap (cloud);
        cloud_mutex_.unlock();
    }
    if(cloud)
    {
        color_handler->setInputCloud(cloud);
        if (!viewer->updatePointCloud (cloud, *color_handler, "cloud"))
            viewer->addPointCloud (cloud, *color_handler, "cloud");
        ui->qvtkRealtimePointcloud->update ();
    }
}

void Calibration::imgThreadFun()
{
    while(updateWindowFlag)
    {
        if(0==camera->getAnImage())
        {
            {
                if(1==camera->img.GetBitsPerPixel())
                {
                    img=QImage(camera->img.GetCols(),camera->img.GetRows(),QImage::Format_Mono);
                    memcpy(img.bits(),(const unsigned char*)(camera->img.GetData()),camera->img.GetCols()*camera->img.GetRows());
                    emit updateRealtimeImage_();
                }
                else
                {
                    img=QImage(camera->img.GetCols(),camera->img.GetRows(),QImage::Format_RGB888);
                    memcpy(img.bits(),(const unsigned char*)(camera->img.GetData()),camera->img.GetCols()*camera->img.GetRows()*3);
                    emit updateRealtimeImage_();
                }
            }
            usleep(50000);
        };
    }
}

void Calibration::on_btnCapture_clicked()
{
    static int count=0;
    if(count==0)
    {
        imgHeight=img.height();
        imgWidth=img.width();
    }
    else
    {
        if((imgHeight!=img.height())||(imgWidth!=img.width()))
        {
            cerr<<"there is something wrong with image size!"<<endl;
            return;
        }
    }
    updateWindowFlag=false;
    imgThread.join();

    cv::Mat image;
    if(img.format()==QImage::Format_RGB888)
    {
        cv::Mat rgbimage=cv::Mat(img.height(),img.width(),CV_8UC3,img.bits());
        cv::cvtColor(rgbimage,image,CV_RGB2BGR);
    }
    else if(img.format()==QImage::Format_Mono)
    {
        image=cv::Mat(img.height(),img.width(),CV_8UC1,img.bits());
    }
    if(0==processImageAndPointcloud(image,cloud))
    {
        stringstream filename;
        filename<<"image"<<count<<".jpg";
        cv::imwrite(filename.str().c_str(),image);
        filename.str("");
        filename<<"velodyne"<<count<<".pcd";
        pcl::io::savePCDFile(filename.str().c_str(),*cloud);
        ++count;
    }

    updateWindowFlag=true;
    imgThread=boost::thread(imgFun);
}

void Calibration::on_btnCalibration_clicked()
{
    if(vecLaserPoint.size()<10)
    {
        cerr<<"not enough images,at least 10 images needed"<<endl;
        return;
    }
    cv::Mat rVecL,tVecL,RVecL;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size boardSize(ui->spinBoxCalibrationboardHeight->value(),ui->spinBoxCalibrationBoardWidth->value());
    float squareSize=ui->doubleSpinBoxCalibrationboardSquareSize->value();
    calibration(vecImg,vecLaserPoint,vvecPlane,boardSize,squareSize,cameraMatrix,distCoeffs,rVecL,tVecL);
    cv::Rodrigues(rVecL,RVecL);

    double *ptr;
    CV_Assert(cameraMatrix.isContinuous());
    ptr=(double*)cameraMatrix.data;
    cout<<"camera matrix:"<<endl;
    cout<<ptr[0]<<"\t"<<ptr[1]<<"\t"<<ptr[2]<<endl
               <<ptr[3]<<"\t"<<ptr[4]<<"\t"<<ptr[5]<<endl
              <<ptr[6]<<"\t"<<ptr[7]<<"\t"<<ptr[8]<<endl<<endl;

    CV_Assert(distCoeffs.isContinuous());
    ptr=(double*)distCoeffs.data;
    cout<<"distCoeffs matrix:"<<endl;
    cout<<ptr[0]<<endl
               <<ptr[1]<<endl
              <<ptr[2]<<endl
             <<ptr[3]<<endl
            <<ptr[4]<<endl<<endl;

    CV_Assert(RVecL.isContinuous());
    ptr=(double*)RVecL.data;
    cout<<"lidar to camera roate matrix:"<<endl;
    cout<<ptr[0]<<"\t"<<ptr[1]<<"\t"<<ptr[2]<<endl
               <<ptr[3]<<"\t"<<ptr[4]<<"\t"<<ptr[5]<<endl
              <<ptr[6]<<"\t"<<ptr[7]<<"\t"<<ptr[8]<<endl<<endl;
    CV_Assert(tVecL.isContinuous());

    ptr=(double*)tVecL.data;
    cout<<"lidar to camera translation vector:"<<endl;
    cout<<ptr[0]<<endl
               <<ptr[1]<<endl
              <<ptr[2]<<endl<<endl;
}

void Calibration::on_radioButtonFixAspectRatio_clicked()
{
    ui->doubleSpinBoxAspectRatio->setVisible(ui->radioButtonFixAspectRatio->isChecked());
}

void Calibration::on_btnLoad_clicked()
{
    int count=0;
    cv::Mat image;
    while(1)
    {
        stringstream filename;
        filename<<"image"<<count<<".jpg";
        image=cv::imread(filename.str().c_str());

        if(image.empty())
            break;
        if(count==0)
        {
            imgHeight=image.rows;
            imgWidth=image.cols;
        }
        else
        {
            if((imgHeight!=image.rows)||(imgWidth!=image.cols))
            {
                cerr<<"there is something wrong with image size!"<<endl;
                return;
            }
        }

        filename.str("");
        filename<<"velodyne"<<count<<".pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud (new pcl::PointCloud<pcl::PointXYZI>);
        if(0!=pcl::io::loadPCDFile(filename.str().c_str(), *pointcloud))
        {
            break;
        };
        processImageAndPointcloud(image,pointcloud);
        ++count;
    }
}
