#include "calibrateExtrinsic.h"
#include <iostream>
#include "stdio.h"
static const char* cvDistCoeffErr = "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8 or 8x1 floating-point vector";

const double factor=1.0/1000.0;
void collectCalibrationData( InputArrayOfArrays objectPoints,
                             InputArrayOfArrays imagePoints1,
                             InputArrayOfArrays imagePoints2,
                             Mat& objPtMat, Mat& imgPtMat1, Mat* imgPtMat2,
                             Mat& npoints )
{
    int nimages = (int)objectPoints.total();
    int i, j = 0, ni = 0, total = 0;
    CV_Assert(nimages > 0 && nimages == (int)imagePoints1.total() &&
              (!imgPtMat2 || nimages == (int)imagePoints2.total()));

    for( i = 0; i < nimages; i++ )
    {
        ni = objectPoints.getMat(i).checkVector(3, CV_32F);
        CV_Assert( ni >= 0 );
        total += ni;
    }

    npoints.create(1, (int)nimages, CV_32S);
    objPtMat.create(1, (int)total, CV_32FC3);
    imgPtMat1.create(1, (int)total, CV_32FC2);
    Point2f* imgPtData2 = 0;

    if( imgPtMat2 )
    {
        imgPtMat2->create(1, (int)total, CV_32FC2);
        imgPtData2 = imgPtMat2->ptr<Point2f>();
    }

    Point3f* objPtData = objPtMat.ptr<Point3f>();
    Point2f* imgPtData1 = imgPtMat1.ptr<Point2f>();

    for( i = 0; i < nimages; i++, j += ni )
    {
        Mat objpt = objectPoints.getMat(i);
        Mat imgpt1 = imagePoints1.getMat(i);
        ni = objpt.checkVector(3, CV_32F);
        int ni1 = imgpt1.checkVector(2, CV_32F);
        CV_Assert( ni > 0 && ni == ni1 );
        npoints.at<int>(i) = ni;
        memcpy( objPtData + j, objpt.data, ni*sizeof(objPtData[0]) );
        memcpy( imgPtData1 + j, imgpt1.data, ni*sizeof(imgPtData1[0]) );

        if( imgPtData2 )
        {
            Mat imgpt2 = imagePoints2.getMat(i);
            int ni2 = imgpt2.checkVector(2, CV_32F);
            CV_Assert( ni == ni2 );
            memcpy( imgPtData2 + j, imgpt2.data, ni*sizeof(imgPtData2[0]) );
        }
    }
}

cv::Mat prepareCameraMatrix(cv::Mat& cameraMatrix0, int rtype)
{
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, rtype);
    if( cameraMatrix0.size() == cameraMatrix.size() )
        cameraMatrix0.convertTo(cameraMatrix, rtype);
    return cameraMatrix;
}

cv::Mat prepareDistCoeffs(cv::Mat& distCoeffs0, int rtype)
{
    cv::Mat distCoeffs = cv::Mat::zeros(distCoeffs0.cols == 1 ? cv::Size(1, 8) : cv::Size(8, 1), rtype);
    if( distCoeffs0.size() == cv::Size(1, 4) ||
            distCoeffs0.size() == cv::Size(1, 5) ||
            distCoeffs0.size() == cv::Size(1, 8) ||
            distCoeffs0.size() == cv::Size(4, 1) ||
            distCoeffs0.size() == cv::Size(5, 1) ||
            distCoeffs0.size() == cv::Size(8, 1) )
    {
        cv::Mat dstCoeffs(distCoeffs, cv::Rect(0, 0, distCoeffs0.cols, distCoeffs0.rows));
        distCoeffs0.convertTo(dstCoeffs, rtype);
    }
    return distCoeffs;
}

void optimizeRoate(Mat &r_)
{
    double *r[4];
    for(int i=0;i<4;i++)
        r[i]=r_.ptr<double>(i);
    double b1[16]={0,1-r[0][0],-r[1][0],-r[2][0],
                   r[0][0]-1,0,-r[2][0],r[1][0],
                   r[1][0],r[2][0],0,-(r[0][0]+1),
                   r[2][0],-r[1][0],r[0][0]+1,0};
    double b2[16]={0,-r[0][1],1-r[1][1],-r[2][1],
                   r[0][1],0,-r[2][1],r[1][1]+1,
                   r[1][1]-1,r[2][1],0,-r[0][1],
                   r[2][1],-r[1][1]+1,r[0][1],0};
    double b3[16]={0,-r[0][2],-r[1][2],1-r[2][2],
                   r[0][2],0,-(r[2][2]+1),r[1][2],
                   r[1][2],r[2][2]+1,0,-r[0][2],
                   r[2][2]-1,-r[1][2],-r[0][2],0};
    Mat b1_(4,4,CV_64F,b1);
    Mat b2_(4,4,CV_64F,b2);
    Mat b3_(4,4,CV_64F,b3);
    Mat b=b1_.t()*b1_+b2_.t()*b2_+b3_.t()*b3_;

    cv::Mat eValuesMat;
    cv::Mat eVectorsMat;
    cv::eigen(b, eValuesMat, eVectorsMat);

    double* q;
    Mat matQ=eVectorsMat(Range(3,4),Range::all());
    q=matQ.ptr<double>(0);

    double roate_[9];
    roate_[0]=q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3];
    roate_[1]=2*(q[1]*q[2]-q[0]*q[3]);
    roate_[2]=2*(q[3]*q[1]+q[0]*q[2]);
    roate_[3]=2*(q[1]*q[2]+q[0]*q[3]);
    roate_[4]=q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3];
    roate_[5]=2*(q[3]*q[2]-q[0]*q[1]);
    roate_[6]=2*(q[3]*q[1]-q[0]*q[2]);
    roate_[7]=2*(q[3]*q[2]+q[0]*q[1]);
    roate_[8]=q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
    r_=Mat(3,3,CV_64F,roate_).clone();
}

void reprojectLidarPoints(Mat& rCVecs, Mat &tCVecs,Mat &rLVecs, Mat& tLVecs,Mat& points,Mat *delta,
                          Mat* dpdrC=NULL,Mat* dpdtC =NULL,
                          Mat* dpdrL =NULL,Mat* dpdtL=NULL)
{
    double boardInCamera[4];
    double dRC[9],drCdT[27];

    CV_Assert(points.rows==4);
    Mat matrCdT=Mat(3, 9, CV_64F,drCdT),matRC=Mat(3, 3, CV_64F,dRC);
    Rodrigues(rCVecs,matRC,matrCdT);
    //    cout<<"r"<<rCVecs.at<double>(0,0)<<"\t"<<rCVecs.at<double>(1,0)<<"\t"<<rCVecs.at<double>(2,0)<<"\t"<<endl;
    //    cout<<tCVecs.at<double>(0,0)<<"\t"<<tCVecs.at<double>(1,0)<<"\t"<<tCVecs.at<double>(2,0)<<"\t"<<endl;
    boardInCamera[0]=dRC[2];
    boardInCamera[1]=dRC[5];
    boardInCamera[2]=dRC[8];
    boardInCamera[3]=(-dRC[2]*tCVecs.at<double>(0,0)-dRC[5]*tCVecs.at<double>(1,0)-dRC[8]*tCVecs.at<double>(2,0));
    //    matRC*drCdT;
    Mat matBoardInCamera=Mat(4,1,CV_64F,boardInCamera);

    double dC2L[16]={0};
    Mat matL2C(4,4,CV_64F,dC2L);
    Mat matRC2L=matL2C(Range(0,3),Range(0,3));
    Mat matdrLdR;
    Rodrigues(rLVecs,matRC2L,matdrLdR);
    matL2C.at<double>(0,3)=tLVecs.at<double>(0,0);
    matL2C.at<double>(1,3)=tLVecs.at<double>(1,0);
    matL2C.at<double>(2,3)=tLVecs.at<double>(2,0);
    matL2C.at<double>(3,3)=1.0;

    double *boardInLidar;
    Mat  matBoardInLidar(1,4,CV_64F,boardInLidar);
    matBoardInLidar=matL2C.t()*matBoardInCamera;
    CV_Assert(matBoardInLidar.isContinuous());
    boardInLidar=(double*)matBoardInLidar.data;

    double rr=boardInLidar[0]*boardInLidar[0]+boardInLidar[1]*boardInLidar[1]+boardInLidar[2]*boardInLidar[2];
    //    Mat error=abs(matBoardInLidar.t()*points)/sqrt(rr);


    //d(d^2) /dpi
    Mat dd=matBoardInLidar.t()*points;

    *delta=factor*dd.mul(dd)/rr;

    //    double min,max;
    //    minMaxIdx(*delta,&min,&max);
    //    cerr<<min<<"    "<<max<<endl;

    if(dpdrC!=NULL)
    {

        Mat dd4,pi4;
        repeat(dd,4,1,dd4);
        repeat(matBoardInLidar,1,points.cols,pi4);
        Mat df2dpi=(2*rr*dd4.mul(points)-2*dd4.mul(dd4).mul(pi4))/rr/rr;
        df2dpi=df2dpi.t();
        //    Mat df2dpiL=((2*rr*rr*points.mul(dd4)-2*matBoardInLidar*dd)/rr/rr).t();

        //the effect of chessboard to camera roate and translation
        Mat dTCdr=Mat(3,3,CV_64F);
        static int indexs[3] = { 2, 5, 8 };
#define INV(x) x?0:1/x
        for (int i=0;i<3;i++)
        {
            dTCdr.at<double>(i,0)=INV(matrCdT.at<double>(0,indexs[i]));
            dTCdr.at<double>(i,1)=INV(matrCdT.at<double>(1,indexs[i]));
            dTCdr.at<double>(i,2)=INV(matrCdT.at<double>(2,indexs[i]));
        }
        Mat dpidrC=matL2C(Range(0,3),Range(0,4)).t();
        dpidrC.at<double>(3,0)-=tCVecs.at<double>(0,0);
        dpidrC.at<double>(3,1)-=tCVecs.at<double>(0,1);
        dpidrC.at<double>(3,2)-=tCVecs.at<double>(0,2);
        dpidrC=dpidrC*dTCdr;

        Mat dpidtC=Mat::zeros(4,3,CV_64F);
        dpidtC.at<double>(3,0)=-matRC.at<double>(0,2);
        dpidtC.at<double>(3,1)=-matRC.at<double>(1,2);
        dpidtC.at<double>(3,2)=-matRC.at<double>(2,2);


        *dpdrC=factor*df2dpi*dpidrC;
        *dpdtC=factor*df2dpi*dpidtC;
        //the effect of  camera to lidar roate and translation
        Mat dpidTL=Mat::zeros(4,12,CV_64F);
        dpidTL.at<double>(0,0)=dpidTL.at<double>(1,3)=dpidTL.at<double>(2,6)=dpidTL.at<double>(3,9)=matRC.at<double>(0,2);
        dpidTL.at<double>(0,1)=dpidTL.at<double>(1,4)=dpidTL.at<double>(2,7)=dpidTL.at<double>(3,10)=matRC.at<double>(2,2);
        dpidTL.at<double>(0,2)=dpidTL.at<double>(1,5)=dpidTL.at<double>(2,8)=dpidTL.at<double>(3,11)=matRC.at<double>(3,2);

        Mat dTdrL=Mat::zeros(12,3,CV_64F);
        int index2[]={0,3,6,1,4,7,2,5,8};
        for(int i=0;i<9;i++)
        {
            dTdrL.at<double>(index2[i],0)=INV(matdrLdR.at<double>(0,i));
            dTdrL.at<double>(index2[i],1)=INV(matdrLdR.at<double>(1,i));
            dTdrL.at<double>(index2[i],2)=INV(matdrLdR.at<double>(2,i));
        }

        *dpdrL=factor*df2dpi*dpidTL*dTdrL;

        Mat dTdtL=Mat::zeros(12,3,CV_64F);
        dTdtL.at<double>(9,0)=1.;
        dTdtL.at<double>(10,1)=1.;
        dTdtL.at<double>(11,2)=1.;

        *dpdtL=factor*df2dpi*dpidTL*dTdtL;

        //    Mat test=points.t();
        //    for(int i=0;i<test.rows;i++)
        //    {
        //        for(int j=0;j<test.cols;j++)
        //            cout<<test.at<double>(i,j)<<"\t";
        //        cout<<endl;
        //    }
    }
};



double calibrateCameraAndLidar2( const CvMat* objectPoints,
                                 const CvMat* imagePoints, const CvMat* npoints,
                                 CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
                                 CvMat* rvecs, CvMat* tvecs, int flags, CvTermCriteria termCrit,
                                 std::vector<cv::Mat > &vecPoint,
                                 Mat &matPlaneInLidar,Mat &rVecL,Mat &tVecL)
{
    const int NINTRINSIC = 12;
    const int NEXTRINSIC = 6;

    Ptr<CvMat> matM, _m, _Ji, _Je, _err,_JiL, _JeL, _errL;
    CvLevMarq solver;
    double reprojErr = 0;
    double reprojErrL=0;

    double A[9], k[8] = {0,0,0,0,0,0,0,0};
    CvMat matA = cvMat(3, 3, CV_64F, A), _k;
    int i, nimages, maxPoints = 0, ni = 0, pos, total = 0, nparams, npstep, cn;
    double aspectRatio = 0.;

    // 0. check the parameters & allocate buffers
    if( !CV_IS_MAT(objectPoints) || !CV_IS_MAT(imagePoints) ||
            !CV_IS_MAT(npoints) || !CV_IS_MAT(cameraMatrix) || !CV_IS_MAT(distCoeffs) )
        CV_Error( CV_StsBadArg, "One of required vector arguments is not a valid matrix" );

    if( imageSize.width <= 0 || imageSize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "image width and height must be positive" );

    if( CV_MAT_TYPE(npoints->type) != CV_32SC1 ||
            (npoints->rows != 1 && npoints->cols != 1) )
        CV_Error( CV_StsUnsupportedFormat,
                  "the array of point counters must be 1-dimensional integer vector" );

    nimages = npoints->rows*npoints->cols;
    npstep = npoints->rows == 1 ? 1 : npoints->step/CV_ELEM_SIZE(npoints->type);

    if( rvecs )
    {
        cn = CV_MAT_CN(rvecs->type);
        if( !CV_IS_MAT(rvecs) ||
                (CV_MAT_DEPTH(rvecs->type) != CV_32F && CV_MAT_DEPTH(rvecs->type) != CV_64F) ||
                ((rvecs->rows != nimages || (rvecs->cols*cn != 3 && rvecs->cols*cn != 9)) &&
                 (rvecs->rows != 1 || rvecs->cols != nimages || cn != 3)) )
            CV_Error( CV_StsBadArg, "the output array of rotation vectors must be 3-channel "
                                    "1xn or nx1 array or 1-channel nx3 or nx9 array, where n is the number of views" );
    }

    if( tvecs )
    {
        cn = CV_MAT_CN(tvecs->type);
        if( !CV_IS_MAT(tvecs) ||
                (CV_MAT_DEPTH(tvecs->type) != CV_32F && CV_MAT_DEPTH(tvecs->type) != CV_64F) ||
                ((tvecs->rows != nimages || tvecs->cols*cn != 3) &&
                 (tvecs->rows != 1 || tvecs->cols != nimages || cn != 3)) )
            CV_Error( CV_StsBadArg, "the output array of translation vectors must be 3-channel "
                                    "1xn or nx1 array or 1-channel nx3 array, where n is the number of views" );
    }

    if( (CV_MAT_TYPE(cameraMatrix->type) != CV_32FC1 &&
         CV_MAT_TYPE(cameraMatrix->type) != CV_64FC1) ||
            cameraMatrix->rows != 3 || cameraMatrix->cols != 3 )
        CV_Error( CV_StsBadArg,
                  "Intrinsic parameters must be 3x3 floating-point matrix" );

    if( (CV_MAT_TYPE(distCoeffs->type) != CV_32FC1 &&
         CV_MAT_TYPE(distCoeffs->type) != CV_64FC1) ||
            (distCoeffs->cols != 1 && distCoeffs->rows != 1) ||
            (distCoeffs->cols*distCoeffs->rows != 4 &&
             distCoeffs->cols*distCoeffs->rows != 5 &&
             distCoeffs->cols*distCoeffs->rows != 8) )
        CV_Error( CV_StsBadArg, cvDistCoeffErr );

    for( i = 0; i < nimages; i++ )
    {
        ni = npoints->data.i[i*npstep];
        if( ni < 4 )
        {
            char buf[100];
            sprintf( buf, "The number of points in the view #%d is < 4", i );
            CV_Error( CV_StsOutOfRange, buf );
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    matM = cvCreateMat( 1, total, CV_64FC3 );
    _m = cvCreateMat( 1, total, CV_64FC2 );

    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );

    nparams = NINTRINSIC + nimages*6+NEXTRINSIC;
    _Ji = cvCreateMat( maxPoints*2, NINTRINSIC, CV_64FC1 );
    _Je = cvCreateMat( maxPoints*2, 6, CV_64FC1 );
    _err = cvCreateMat( maxPoints*2, 1, CV_64FC1 );
    cvZero( _Ji );

    _k = cvMat( distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), k);
    if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 8 )
    {
        if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 5 )
            flags |= CV_CALIB_FIX_K3;
        flags |= CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5 | CV_CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CV_CALIB_USE_INTRINSIC_GUESS )
    {
        cvConvert( cameraMatrix, &matA );
        if( A[0] <= 0 || A[4] <= 0 )
            CV_Error( CV_StsOutOfRange, "Focal length (fx and fy) must be positive" );
        if( A[2] < 0 || A[2] >= imageSize.width ||
                A[5] < 0 || A[5] >= imageSize.height )
            CV_Error( CV_StsOutOfRange, "Principal point must be within the image" );
        if( fabs(A[1]) > 1e-5 )
            CV_Error( CV_StsOutOfRange, "Non-zero skew is not supported by the function" );
        if( fabs(A[3]) > 1e-5 || fabs(A[6]) > 1e-5 ||
                fabs(A[7]) > 1e-5 || fabs(A[8]-1) > 1e-5 )
            CV_Error( CV_StsOutOfRange,
                      "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
        A[1] = A[3] = A[6] = A[7] = 0.;
        A[8] = 1.;

        if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A[0]/A[4];

            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( CV_StsOutOfRange,
                          "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        cvConvert( distCoeffs, &_k );
    }
    else
    {
        CvScalar mean, sdv;
        cvAvgSdv( matM, &mean, &sdv );
        if( fabs(mean.val[2]) > 1e-5 || fabs(sdv.val[2]) > 1e-5 )
            CV_Error( CV_StsBadArg,
                      "For non-planar calibration rigs the initial intrinsic matrix must be specified" );
        for( i = 0; i < total; i++ )
            ((CvPoint3D64f*)matM->data.db)[i].z = 0.;

        if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = cvmGet(cameraMatrix,0,0);
            aspectRatio /= cvmGet(cameraMatrix,1,1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( CV_StsOutOfRange,
                          "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        cvInitIntrinsicParams2D( matM, _m, npoints, imageSize, &matA, aspectRatio );
    }

    solver.init( nparams, 0, termCrit);

    {
        double* param = solver.param->data.db;
        uchar* mask = solver.mask->data.ptr;

        param[0] = A[0]; param[1] = A[4]; param[2] = A[2]; param[3] = A[5];
        param[4] = k[0]; param[5] = k[1]; param[6] = k[2]; param[7] = k[3];
        param[8] = k[4]; param[9] = k[5]; param[10] = k[6]; param[11] = k[7];

        if( flags & CV_CALIB_FIX_FOCAL_LENGTH )
            mask[0] = mask[1] = 0;
        if( flags & CV_CALIB_FIX_PRINCIPAL_POINT )
            mask[2] = mask[3] = 0;
        if( flags & CV_CALIB_ZERO_TANGENT_DIST )
        {
            param[6] = param[7] = 0;
            mask[6] = mask[7] = 0;
        }
        if( !(flags & CV_CALIB_RATIONAL_MODEL) )
            flags |= CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5 + CV_CALIB_FIX_K6;
        if( flags & CV_CALIB_FIX_K1 )
            mask[4] = 0;
        if( flags & CV_CALIB_FIX_K2 )
            mask[5] = 0;
        if( flags & CV_CALIB_FIX_K3 )
            mask[8] = 0;
        if( flags & CV_CALIB_FIX_K4 )
            mask[9] = 0;
        if( flags & CV_CALIB_FIX_K5 )
            mask[10] = 0;
        if( flags & CV_CALIB_FIX_K6 )
            mask[11] = 0;
    }

    // 2. initialize chessboard to camera  extrinsic parameters
    for( i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        CvMat _Mi, _mi, _ri, _ti;
        ni = npoints->data.i[i*npstep];

        cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
        cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

        cvGetCols( matM, &_Mi, pos, pos + ni );
        cvGetCols( _m, &_mi, pos, pos + ni );

        cvFindExtrinsicCameraParams2( &_Mi, &_mi, &matA, &_k, &_ri, &_ti );
    }

    //3. initialize lidar to camera  extrinsic parameters
    Mat matPlaneInCamera(4,nimages,CV_64F);
    for( int i = 0; i < nimages; i++)
    {
        CvMat _ri, _ti;
        double *pRi,*pTi;
        cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
        cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );
        pRi=_ri.data.db;
        pTi=_ti.data.db;
        Mat ri;
        Rodrigues(Mat(&_ri),ri);
        matPlaneInCamera.at<double>(0,i)=ri.at<double>(0,2);
        matPlaneInCamera.at<double>(1,i)=ri.at<double>(1,2);
        matPlaneInCamera.at<double>(2,i)=ri.at<double>(2,2);
        matPlaneInCamera.at<double>(3,i)=(-ri.at<double>(0,2)*pTi[0]
                -ri.at<double>(1,2)*pTi[1]-ri.at<double>(2,2)*pTi[2]);
    }

    Mat flag=matPlaneInLidar(Range(3,4),Range::all()).clone();
    divide(flag,abs(flag),flag);
    repeat(flag, 4, 1, flag);
    multiply(matPlaneInLidar,flag,matPlaneInLidar);
    flag=matPlaneInCamera(Range(3,4),Range::all()).clone();
    divide(flag,abs(flag),flag);
    repeat(flag, 4, 1, flag);
    multiply(matPlaneInCamera,flag,matPlaneInCamera);

    Mat Ax=Mat::zeros(3*nimages,9,CV_64F);
    Mat Bx=Mat::zeros(3*nimages,1,CV_64F);
    for(int i=0;i<nimages;i++)
    {
        Ax.at<double>(3*i,0)=Ax.at<double>(3*i+1,3)=Ax.at<double>(3*i+2,6)=matPlaneInLidar.at<double>(0,i);
        Ax.at<double>(3*i,1)=Ax.at<double>(3*i+1,4)=Ax.at<double>(3*i+2,7)=matPlaneInLidar.at<double>(1,i);
        Ax.at<double>(3*i,2)=Ax.at<double>(3*i+1,5)=Ax.at<double>(3*i+2,8)=matPlaneInLidar.at<double>(2,i);
        Bx.at<double>(3*i,0)=matPlaneInCamera.at<double>(0,i);
        Bx.at<double>(3*i,1)=matPlaneInCamera.at<double>(1,i);
        Bx.at<double>(3*i,2)=matPlaneInCamera.at<double>(2,i);
    }
    Mat X;
    bool ret=solve(Ax,Bx,X,DECOMP_SVD);
    Mat r=X.reshape(0,3);
    optimizeRoate(r);

    flag=matPlaneInLidar(Range(3,4),Range::all()).clone();
    repeat(flag, 4, 1, flag);
    divide(matPlaneInLidar,flag,matPlaneInLidar);
    flag=matPlaneInCamera(Range(3,4),Range::all()).clone();
    repeat(flag, 4, 1, flag);
    divide(matPlaneInCamera,flag,matPlaneInCamera);

    Mat At=matPlaneInLidar(Range(0,3),Range::all()).t();
    Mat temp,temp1;
    pow(matPlaneInLidar(Range(0,3),Range::all()),2,temp);
    reduce(temp, temp, 0, CV_REDUCE_SUM);
    pow(r*matPlaneInCamera(Range(0,3),Range::all()),2,temp1);
    reduce(temp1, temp1, 0, CV_REDUCE_SUM);
    divide(temp,temp1,temp);
    pow(temp,0.5,temp);
    Mat Bt=temp-Mat::ones(temp.rows,temp.cols,CV_64F);
    Bt=Bt.t();
    ret=solve(At,Bt,X,DECOMP_SVD);
    Mat t=X.reshape(0,3);
    t=-r*t;
    {
        double* param = solver.param->data.db+NINTRINSIC + nimages*6;
        Mat rodR(3,1,CV_64F,param);
        Rodrigues(r,rodR);
        memcpy((unsigned char*)(param+3),t.data,3*sizeof(double));
    }

    // 4. run the optimization
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;
        bool proceed = solver.updateAlt( _param, _JtJ, _JtErr, _errNorm );
        double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;

        if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        {
            param[0] = param[1]*aspectRatio;
            pparam[0] = pparam[1]*aspectRatio;
        }

        A[0] = param[0]; A[4] = param[1]; A[2] = param[2]; A[5] = param[3];
        k[0] = param[4]; k[1] = param[5]; k[2] = param[6]; k[3] = param[7];
        k[4] = param[8]; k[5] = param[9]; k[6] = param[10]; k[7] = param[11];

        if( !proceed )
            break;

        reprojErr = 0;
        reprojErrL=0;

        for( i = 0, pos = 0; i < nimages; i++, pos += ni )
        {
            CvMat _Mi, _mi, _ri, _ti, _dpdr, _dpdt, _dpdf, _dpdc, _dpdk, _mp, _part,_rL,_tL;
            Mat dpdrC,dpdtC ,dpdrL,dpdtL,delta;
            ni = npoints->data.i[i*npstep];

            cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
            cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

            cvGetRows( solver.param, &_rL, NINTRINSIC +nimages*6 , NINTRINSIC +nimages*6 + 3);
            cvGetRows( solver.param, &_tL, NINTRINSIC +nimages*6+3 , NINTRINSIC +nimages*6 + 6);

            cvGetCols( matM, &_Mi, pos, pos + ni );
            cvGetCols( _m, &_mi, pos, pos + ni );

            _Je->rows = _Ji->rows = _err->rows = ni*2;
            cvGetCols( _Je, &_dpdr, 0, 3 );
            cvGetCols( _Je, &_dpdt, 3, 6 );
            cvGetCols( _Ji, &_dpdf, 0, 2 );
            cvGetCols( _Ji, &_dpdc, 2, 4 );
            cvGetCols( _Ji, &_dpdk, 4, NINTRINSIC );
            cvReshape( _err, &_mp, 2, 1 );

            Mat ri=Mat(&_ri);
            Mat ti=Mat(&_ti);
            Mat rL=Mat(&_rL);
            Mat tL=Mat(&_tL);
            if( _JtJ || _JtErr )
            {
                cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, &_k, &_mp, &_dpdr, &_dpdt,
                                  (flags & CV_CALIB_FIX_FOCAL_LENGTH) ? 0 : &_dpdf,
                                  (flags & CV_CALIB_FIX_PRINCIPAL_POINT) ? 0 : &_dpdc, &_dpdk,
                                  (flags & CV_CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0);
                reprojectLidarPoints(ri,ti,rL,tL,vecPoint[i],&delta,&dpdrC,&dpdtC ,&dpdrL,&dpdtL);
            }
            else
            {
                cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, &_k, &_mp );
                reprojectLidarPoints(ri,ti,rL,tL,vecPoint[i],&delta);
            }

            cvSub( &_mp, &_mi, &_mp );

            if( _JtJ || _JtErr )
            {
                cvGetSubRect( _JtJ, &_part, cvRect(0,0,NINTRINSIC,NINTRINSIC) );
                cvGEMM( _Ji, _Ji, 1, &_part, 1, &_part, CV_GEMM_A_T );

                cvGetSubRect( _JtJ, &_part, cvRect(NINTRINSIC+i*6,NINTRINSIC+i*6,6,6) );
                cvGEMM( _Je, _Je, 1, 0, 0, &_part, CV_GEMM_A_T );

                cvGetSubRect( _JtJ, &_part, cvRect(NINTRINSIC+i*6,0,6,NINTRINSIC) );
                cvGEMM( _Ji, _Je, 1, 0, 0, &_part, CV_GEMM_A_T );

                cvGetRows( _JtErr, &_part, 0, NINTRINSIC );
                cvGEMM( _Ji, _err, 1, &_part, 1, &_part, CV_GEMM_A_T );

                cvGetRows( _JtErr, &_part, NINTRINSIC + i*6, NINTRINSIC + (i+1)*6 );
                cvGEMM( _Je, _err, 1, 0, 0, &_part, CV_GEMM_A_T );

                /*Jacobian come from lidar to camera transformation*/
                Mat merge(vecPoint[i].cols,6,CV_64F);
                Mat submat = merge.colRange(0,3);
                dpdrC.copyTo(submat);
                submat = merge.colRange(3,6);
                dpdtC.copyTo(submat);

                Mat JtJ=Mat(_JtJ);
                Mat JtErr=Mat(_JtErr);
                delta=delta.t();
                submat=JtJ(Range(NINTRINSIC+i*6,NINTRINSIC+i*6+6),Range(NINTRINSIC+i*6,NINTRINSIC+i*6+6));
                gemm(merge,merge,1,submat,1,submat,CV_GEMM_A_T);

                submat=JtErr.rowRange(NINTRINSIC + i*6, NINTRINSIC + (i+1)*6);
                gemm(merge,delta, 1,submat,1,submat, CV_GEMM_A_T );

                submat = merge.colRange(0,3);
                dpdrL.copyTo(submat);
                submat = merge.colRange(3,6);
                dpdtL.copyTo(submat);
                submat=JtJ(Range(NINTRINSIC+nimages*6,NINTRINSIC+nimages*6+6),Range(NINTRINSIC+nimages*6,NINTRINSIC+nimages*6+6));
                gemm(merge,merge,1,submat,1,submat,CV_GEMM_A_T);

                submat=JtErr.rowRange(NINTRINSIC + nimages*6, NINTRINSIC + (nimages+1)*6);
                gemm(merge,delta, 1,submat,1,submat, CV_GEMM_A_T );
            }
            double errNorm = cvNorm( &_mp, 0, CV_L2 );
            reprojErr += errNorm*errNorm;
            reprojErrL+=mean(delta)[0];
            //                                double min,max;
            //                                Mat mp(&_mp);
            //                                minMaxIdx(mp,&min,&max);
            //                                cout<<min<<"    "<<max<<endl;
        }
//        cerr<<"pointcloud reproject error:"<<reprojErrL/nimages<<endl;

        reprojErr=reprojErr/total;
        if( _errNorm )
            *_errNorm = reprojErr;
//        cout<<"camera reproject error:"<<std::sqrt(reprojErr)<<endl;
    }

    // 5. store the results
    cvConvert( &matA, cameraMatrix );
    cvConvert( &_k, distCoeffs );

    for( i = 0; i < nimages; i++ )
    {
        CvMat src, dst;
        if( rvecs )
        {
            src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 );
            if( rvecs->rows == nimages && rvecs->cols*CV_MAT_CN(rvecs->type) == 9 )
            {
                dst = cvMat( 3, 3, CV_MAT_DEPTH(rvecs->type),
                             rvecs->data.ptr + rvecs->step*i );
                cvRodrigues2( &src, &matA );
                cvConvert( &matA, &dst );
            }
            else
            {
                dst = cvMat( 3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
                                 rvecs->data.ptr + i*CV_ELEM_SIZE(rvecs->type) :
                                 rvecs->data.ptr + rvecs->step*i );
                cvConvert( &src, &dst );
            }
        }
        if( tvecs )
        {
            src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 + 3 );
            dst = cvMat( 3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
                             tvecs->data.ptr + i*CV_ELEM_SIZE(tvecs->type) :
                             tvecs->data.ptr + tvecs->step*i );
            cvConvert( &src, &dst );
        }
    }

    Mat optir=Mat(1,3,CV_64F,&solver.param->data.db[NINTRINSIC + nimages*6]);
    optir.copyTo(rVecL);
    Mat optit=Mat(1,3,CV_64F,&solver.param->data.db[NINTRINSIC + nimages*6+3]);
    optit.copyTo(tVecL);

//    double optiT_[16];
//    Mat optiT(4,4,CV_64F,optiT_);
//    Mat optir=Mat(1,3,CV_64F,&solver.param->data.db[NINTRINSIC + nimages*6]);
//    Mat optiR;
//    Rodrigues(optir,optiR);
//    Mat submat=optiT(Range(0,3),Range(0,3));
//    optiR.copyTo(submat);
//    optiT_[3]=solver.param->data.db[NINTRINSIC + nimages*6+3];
//    optiT_[7]=solver.param->data.db[NINTRINSIC + nimages*6+4];
//    optiT_[11]=solver.param->data.db[NINTRINSIC + nimages*6+5];
//    optiT_[12]=optiT_[13]=optiT_[14]=0;
//    optiT_[15]=1;

//    cout<<"optizimed result:";
//    for(int i=0;i<16;i++)
//        cout<<optiT_[i]<<"\t";
//    cout<<endl;


//    double mat_[]={0.999389607504606,0.034300725687374,  -0.006623641680219,
//                   -0.006267271265040,  -0.010486705359104,  -0.999925372376160,
//                   -0.034367626084425,   0.999356537592080,  -0.010265332400381};
//    double tL_[]={-3.807207571682040,-3.807207571682040,-3.807207571682040};

//    double roateVec[]={0.190642131987721,   0.456089910809645,  -0.047571620767705,
//                     0.001268217000157,  -0.402935015682191,  -0.023391403839455,
//                     0.077926133478526,  -0.583137844459217,  -1.618132713724060,
//                    -0.098116920423716,   0.130230043478334,  -1.583111764863171,
//                     0.298620247787020,   0.572349173381591,  -1.575152170422742,
//                     0.074220752761960,  -0.556414518622429,  -0.004399198302642,
//                     0.210404447795883,   0.745131583798004,  -0.080189129582092,
//                     0.325366444646209,   0.342172586822293,  -1.557648214145086,
//                     0.036147343258706,   0.056028010198836,  -1.296380197275425,
//                    -0.036418188307553,   0.239530232061863,  -1.000217919942123,
//                    -0.004358204728471,   0.122168695973289,  -0.870168893456246,
//                     0.339197827935503,  -0.532421366305479,  -0.896657572504623,
//                    -0.096258016688925,   0.644208415356438,  -1.048832793306796};

//    double transVec[]={
//        0.122288203919863,  -0.339144388514384,   2.861482748601342,
//       -0.349562226943821,  -0.316057952453260,   2.089177461467003,
//       -0.168906529198920,   0.330225368373914,   1.703933776981739,
//       -0.204742391346970,   0.413846050148017,   1.620173874867283,
//       -0.107402574778565,   0.236654961918996,   2.269841511291580,
//       -0.287620580874946,  -0.308147998498301,   1.701529577832018,
//       -0.255400699626491,  -0.311217334993108,   1.595140876449788,
//       -0.221307952045118,   0.308794981603408,   2.202535236514681,
//        0.054812258912101,   0.396331708314539,   1.808829187931801,
//       -0.290256710994700,   0.274849140718396,   1.959392266783563,
//       -0.455554393975717,   0.193076833730876,   1.868775440867674,
//       -0.278541961421252,   0.244718756648981,   1.414716960177043,
//       -0.044489308929449,   0.264693661267861,   2.279182927142079
//    };
//    for(i=0;i<sizeof(transVec)/sizeof(double);i++)
//        transVec[i]*=1000.0;


//    Mat mat(3,3,CV_64F,mat_);
//    Mat rL;
//    Rodrigues(mat,rL);

//    Mat tL(3,1,CV_64F,tL_);
//    Mat delta;
//    double sum=0.0;
//    for(int i=0;i<nimages;i++)
//    {
//        Mat ri=Mat(3,1,CV_64F,roateVec+i*3);
//        Mat ti=Mat(3,1,CV_64F,transVec+i*3);
//        reprojectLidarPoints(ri,ti,rL,tL,vecPoint[i],&delta);
//        sum+=mean(delta)[0];
//    }
//    cerr<<"matlab result:"<<sum/nimages<<endl;
    return std::sqrt(reprojErr);
}



double calibrateCameraAndLidar(cv::InputArrayOfArrays _objectPoints,
                               cv::InputArrayOfArrays _imagePoints,
                               cv::Size imageSize, cv::InputOutputArray _cameraMatrix, cv::InputOutputArray _distCoeffs,
                               cv::OutputArrayOfArrays _rvecs, cv::OutputArrayOfArrays _tvecs, int flags, cv::TermCriteria criteria,
                               std::vector<cv::Mat > &vecPoint,
                               Mat &planeInLidar,Mat &rVecL,Mat &tVecL)
{
    int rtype = CV_64F;
    cv::Mat cameraMatrix = _cameraMatrix.getMat();
    cameraMatrix =prepareCameraMatrix(cameraMatrix, rtype);
    cv::Mat distCoeffs = _distCoeffs.getMat();
    distCoeffs = prepareDistCoeffs(distCoeffs, rtype);
    if( !(flags & CV_CALIB_RATIONAL_MODEL) )
        distCoeffs = distCoeffs.rows == 1 ? distCoeffs.colRange(0, 5) : distCoeffs.rowRange(0, 5);

    int    i;
    size_t nimages = _objectPoints.total();
    CV_Assert( nimages > 0 );
    Mat objPt, imgPt, npoints, rvecM((int)nimages, 3, CV_64FC1), tvecM((int)nimages, 3, CV_64FC1);
    collectCalibrationData( _objectPoints, _imagePoints, noArray(),
                            objPt, imgPt, 0, npoints );
    CvMat c_objPt = objPt, c_imgPt = imgPt, c_npoints = npoints;
    CvMat c_cameraMatrix = cameraMatrix, c_distCoeffs = distCoeffs;
    CvMat c_rvecM = rvecM, c_tvecM = tvecM;

    double reprojErr = calibrateCameraAndLidar2(&c_objPt, &c_imgPt, &c_npoints, imageSize,
                                                &c_cameraMatrix, &c_distCoeffs, &c_rvecM,
                                                &c_tvecM, flags, criteria,vecPoint,planeInLidar,rVecL,tVecL);

    bool rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed();

    if( rvecs_needed )
        _rvecs.create((int)nimages, 1, CV_64FC3);
    if( tvecs_needed )
        _tvecs.create((int)nimages, 1, CV_64FC3);

    for( i = 0; i < (int)nimages; i++ )
    {
        if( rvecs_needed )
        {
            _rvecs.create(3, 1, CV_64F, i, true);
            Mat rv = _rvecs.getMat(i);
            memcpy(rv.data, rvecM.ptr<double>(i), 3*sizeof(double));
        }
        if( tvecs_needed )
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            memcpy(tv.data, tvecM.ptr<double>(i), 3*sizeof(double));
        }
    }
    cameraMatrix.copyTo(_cameraMatrix);
    distCoeffs.copyTo(_distCoeffs);

    return reprojErr;
}

static bool runCalibration( std::vector<std::vector<cv::Point2f> > imagePoints,
                            cv::Size imageSize, cv::Size boardSize,
                            float squareSize, float aspectRatio,
                            int flags, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
                            std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs,
                            std::vector<float>& reprojErrs,
                            double& totalAvgErr,
                            std::vector<cv::Mat > &vecPoint,
                            Mat &planeInLidar,Mat &rVecL,Mat &tVecL)
{
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    for( int i = 0; i < boardSize.height; i++ )
        for( int j = 0; j < boardSize.width; j++ )
            objectPoints[0].push_back(cv::Point3f(float(j*squareSize),
                                                  float(i*squareSize), 0));

    objectPoints.resize(imagePoints.size(),objectPoints[0]);


    double rms =calibrateCameraAndLidar(objectPoints, imagePoints, imageSize, cameraMatrix,
                                        distCoeffs, rvecs, tvecs, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5,
                                        TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 1000, DBL_EPSILON),
                                        vecPoint,planeInLidar,rVecL,tVecL);
    ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    std::cout<<"RMS error reported by calibrateCamera:"<< rms<<std::endl;

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    //    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
    //                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


int calibration(std::vector<cv::Mat> &vecImg, std::vector<cv::Mat > &vecPoint,
                vector<vector<float> >vvecPlane, cv::Size boardSize, float squareSize,
                cv::Mat &cameraMatrix,
                cv::Mat &distCoeffs,
                Mat &rVecL,Mat &tVecL)
{
    int nImages=vecImg.size();
    std::vector<std::vector<float> > vvecPlane_;
    std::vector<cv::Mat > vecPoint_;
    std::vector<std::vector<cv::Point2f> > imagePoints;

    for(int i=0;i<nImages;i++)
    {
        std::vector<cv::Point2f> pointbuf;
        bool found=findChessboardCorners(vecImg[i], boardSize, pointbuf,
                                         CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
        if(found)
        {
            cv::Mat viewGray;
            cv::cvtColor(vecImg[i],viewGray,CV_BGR2GRAY);
            cv::cornerSubPix( viewGray, pointbuf, cv::Size(11,11),
                              cv::Size(-1,-1), cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 300, 0.001 ));
            imagePoints.push_back(pointbuf);
            vvecPlane_.push_back(vvecPlane[i]);
            vecPoint_.push_back(vecPoint[i]);
        }
        else
            std::cerr<<"frame "<<i<<" found no corners"<<std::endl;
    }

    Mat planeInLidar(4,vvecPlane_.size(),CV_64F);
    for(int i=0;i<4;i++)
        for(int j=0;j<vvecPlane_.size();j++)
            planeInLidar.at<double>(i,j)=vvecPlane_[j][i];


    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<float> reprojErrs;
    double totalAvgErr = 0;
    cv::Size imageSize=vecImg[0].size();
    float aspectRatio=1.0;
    float flags=0.0f;
    bool ok = runCalibration(imagePoints, imageSize, boardSize, squareSize,
                             aspectRatio, flags,cameraMatrix, distCoeffs,
                             rvecs, tvecs, reprojErrs, totalAvgErr,vecPoint_,planeInLidar,rVecL,tVecL);
    return 0;
};
