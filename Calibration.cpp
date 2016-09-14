#include "Calibration.h"
#include "ui_Calibration.h"
#include <boost/bind.hpp>

#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <sstream>

Calibration::Calibration(int argc,char** argv,QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Calibration),
    updateWindowFlag(1)
{
    ui->setupUi(this);
    camera=new PointGreyCamera;
    if(0!=camera->init())
    {
        return;
    };

    ui->doubleSpinBoxAspectRatio->setVisible(false);


    imgFun=boost::bind(&Calibration::imgThreadFun,this);
    imgThread=boost::thread(imgFun);
    connect(this,SIGNAL(updateRealtimeImage_()),this,SLOT(updateRealtimeImage()));

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
    cout<<"the origin of the "<<index<<"th pointcloud's chessboard is:"<<x<<"    "<<y<<"     "<<z<<endl;
    cout<<vecPointCloud.at(index).width<<" "<<vecPointCloud.at(index).height<<endl;

    double threshold=ui->doubleSpinBoxCalibrationboardSquareSize->value()/100.0/2.0
            *min(ui->spinBoxCalibrationboardHeight->value(),ui->spinBoxCalibrationBoardWidth->value());

    pcl::PointCloud<pcl::PointXYZI>::Ptr board(new pcl::PointCloud<pcl::PointXYZI>);
    for(PointCloud<PointXYZI>::const_iterator it=vecPointCloud.at(index).begin();it!=vecPointCloud.at(index).end();++it)
    {
        if(sqrt(pow((x-it->x),2)+pow((y-it->y),2)+pow((z-it->z),2))<threshold)
        {
            board->push_back(*it);
//            it->intensity=255;
        }
    }

//    PointCloudColorHandlerGenericField<PointXYZI> *color_handler;
//    color_handler = new PointCloudColorHandlerGenericField<PointXYZI> ("intensity");
//    color_handler->setInputCloud(vecPointCloud.at(index));
//    if (!viewer->updatePointCloud (vecPointCloud.at(index), *color_handler, "cloud"))
//        viewer->addPointCloud (vecPointCloud.at(index), *color_handler, "cloud");
//    vecVtkViewer.at(index)->update ();

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (board);
    seg.segment (*inliers, *coefficients);

    stringstream name;
    name<<"plane"<<index;
    vecPCLViewer.at(index)->removeShape(name.str());
    vecPCLViewer.at(index)->addPlane(*coefficients,name.str());
}

void Calibration::cloudCallback(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud)
{
    boost::mutex::scoped_lock lock (cloud_mutex_);
    cloud_ = cloud;
    if(updateWindowFlag)
        emit updateRealtimePointCloud_();
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
    updateWindowFlag=false;
    imgThread.join();
    cv::Mat grayImg;
    if(img.format()==QImage::Format_RGB888)
    {
        cv::Mat rgbImg=cv::Mat(img.height(),img.width(),CV_8UC3,img.bits());
        cv::cvtColor(rgbImg,grayImg,CV_RGB2GRAY);
    }
    else if(img.format()==QImage::Format_Mono)
    {
        grayImg=cv::Mat(img.height(),img.width(),CV_8UC1,img.bits());
    }
    else
    {
        cerr<<"can not detected original image formate!"<<endl;
        return;
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
        updateWindowFlag=true;
        imgThread=boost::thread(imgFun);
        return;
    }

    if(ui->rdBtnChessBoard->isChecked() && found)
        cornerSubPix(grayImg, pointbuf, cv::Size(11,11),cv::Size(-1,-1), cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
    imagePoints.push_back(pointbuf);
    QLabel *lblImg=new QLabel;
    QImage scaledImg=img.scaled(ui->realTimeDisplay->width(),ui->realTimeDisplay->height());
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

        if (!newViewer->updatePointCloud (cloud, *color_handler, "cloud"))
            newViewer->addPointCloud (cloud, *color_handler, "cloud");
        vtkViewer->update ();
        vecVtkViewer.push_back(vtkViewer);
        vecPCLViewer.push_back(newViewer);
        pcl::PointCloud<pcl::PointXYZI> tmpcloud;
        tmpcloud=*cloud;
        vecPointCloud.push_back(tmpcloud);
    }

    ++count;
    updateWindowFlag=true;
    imgThread=boost::thread(imgFun);
}



void Calibration::on_btnCalibration_clicked()
{
    if(imagePoints.size()<10)
    {
        cerr<<"not enough images,at least 10 images needed"<<endl;
        return;
    }

    cv::Size imageSize(img.height(),img.width());
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    int flags;
    if(ui->radioButtonFixAspectRatio->isChecked())
    {
        flags |= CV_CALIB_FIX_ASPECT_RATIO;
        cameraMatrix.at<double>(0,0) = ui->doubleSpinBoxAspectRatio->value();
    }

    cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    vector<vector<cv::Point3f> > objectPoints(1);
    cv::Size boardSize(ui->spinBoxCalibrationboardHeight->value(),ui->spinBoxCalibrationBoardWidth->value());
    float squareSize=(float)ui->doubleSpinBoxCalibrationboardSquareSize->value();
    objectPoints[0].resize(0);
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            objectPoints[0].push_back(cv::Point3f(float(j*squareSize),
                                                  float(i*squareSize), 0));
    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                     distCoeffs, rvecs, tvecs, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    reprojErrs.resize(objectPoints.size());
    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
                          cameraMatrix, distCoeffs, imagePoints2);
        err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), CV_L2);
        int n = (int)objectPoints[i].size();
        reprojErrs[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }
    double totalAvgErr = std::sqrt(totalErr/totalPoints);

    cv::FileStorage fs("result", cv::FileStorage::WRITE );
    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime(buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << ui->doubleSpinBoxAspectRatio->value();

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
                 flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                 flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                 flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                 flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            cv::Mat r = bigmat(cv::Range(i, i+1), cv::Range(0,3));
            cv::Mat t = bigmat(cv::Range(i, i+1), cv::Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            cv::Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            cv::Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
    return;
}

void Calibration::on_radioButtonFixAspectRatio_clicked()
{
    ui->doubleSpinBoxAspectRatio->setVisible(ui->radioButtonFixAspectRatio->isChecked());
}
