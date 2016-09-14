#include "pointgreycamera.h"


PointGreyCamera::PointGreyCamera()
{

}

PointGreyCamera::~PointGreyCamera()
{

}

void PointGreyCamera::printBuildInfo()
{
    FC2Version fc2Version;
    Utilities::GetLibraryVersion( &fc2Version );

    ostringstream version;
    version << "FlyCapture2 library version: " << fc2Version.major << "." << fc2Version.minor << "." << fc2Version.type << "." << fc2Version.build;
    cout << version.str() << endl;

    ostringstream timeStamp;
    timeStamp <<"Application build date: " << __DATE__ << " " << __TIME__;
    cout << timeStamp.str() << endl << endl;

}

void PointGreyCamera::printCameraInfo(CameraInfo *pCamInfo)
{
    cout << endl;
    cout << "*** CAMERA INFORMATION ***" << endl;
    cout << "Serial number -" << pCamInfo->serialNumber << endl;
    cout << "Camera model - " << pCamInfo->modelName << endl;
    cout << "Camera vendor - " << pCamInfo->vendorName << endl;
    cout << "Sensor - " << pCamInfo->sensorInfo << endl;
    cout << "Resolution - " << pCamInfo->sensorResolution << endl;
    cout << "Firmware version - " << pCamInfo->firmwareVersion << endl;
    cout << "Firmware build time - " << pCamInfo->firmwareBuildTime << endl << endl;

}

int PointGreyCamera::getAnImage()
{
    error = cam.RetrieveBuffer(&img);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }
    return 0;
}

void PointGreyCamera::printError(Error error)
{
    error.PrintErrorTrace();
}

void PointGreyCamera::printFormat7Capabilities(Format7Info fmt7Info)
{
    cout << "Max image pixels: (" << fmt7Info.maxWidth << ", " << fmt7Info.maxHeight << ")" << endl;
    cout << "Image Unit size: (" << fmt7Info.imageHStepSize << ", " << fmt7Info.imageVStepSize << ")" << endl;
    cout << "Offset Unit size: (" << fmt7Info.offsetHStepSize << ", " << fmt7Info.offsetVStepSize << ")" << endl;
    cout << "Pixel format bitfield: 0x" << fmt7Info.pixelFormatBitField << endl;
}

int PointGreyCamera::setFrame(float fps)
{
    Property frmRate;
    frmRate.type = FRAME_RATE;
    error= cam.GetProperty(&frmRate);
    if (error != PGRERROR_OK)
    {
        printError(error);
        cerr<<"get camera property failed"<<endl;
        return -1;
    }

    frmRate.onOff = true;
    frmRate.autoManualMode = false;
    frmRate.absValue = fps;
    frmRate.absControl=true;
    error = cam.SetProperty(&frmRate);
    if (error != PGRERROR_OK)
    {
        printError(error);
        cerr<<"set frame failed!"<<endl;
        return -1;
    }
    return 0;

}

int PointGreyCamera::setImageAttr(int width, int heigth)
{
    this->width=width;
    this->heigth=heigth;

    Mode k_fmt7Mode = MODE_0;
    PixelFormat k_fmt7PixFmt ;
    k_fmt7PixFmt= PIXEL_FORMAT_RGB8;

    Format7Info fmt7Info;
    bool supported;
    fmt7Info.mode = k_fmt7Mode;
    error = cam.GetFormat7Info(&fmt7Info, &supported);
    if (error != PGRERROR_OK)
    {
        printError(error);
        cerr<< "get format7 info"<<endl;
        return -1;
    }
    printFormat7Capabilities(fmt7Info);

    if ((k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0 )
    {
        // Pixel format not supported!
        cout << "Pixel format is not supported" << endl;
        return -1;
    }

    Format7ImageSettings fmt7ImageSettings;
    fmt7ImageSettings.mode = k_fmt7Mode;
    fmt7ImageSettings.offsetX =(fmt7Info.maxWidth-width)/2;
    fmt7ImageSettings.offsetY = (fmt7Info.maxHeight-heigth)/2;
    fmt7ImageSettings.width =width /*fmt7Info.maxWidth*/;
    fmt7ImageSettings.height = heigth/*fmt7Info.maxHeight*/;
    fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;

    bool valid;
    Format7PacketInfo fmt7PacketInfo;
    // Validate the settings to make sure that they are valid
    error = cam.ValidateFormat7Settings(
                &fmt7ImageSettings,
                &valid,
                &fmt7PacketInfo );

    if ( !valid )
    {
        // Settings are not valid
        cout << "Format7 settings are not valid" << endl;
        return -1;
    }

    // Set the settings to the camera
    error = cam.SetFormat7Configuration(
                &fmt7ImageSettings,
                fmt7PacketInfo.recommendedBytesPerPacket);
    if (error != PGRERROR_OK)
    {
        printError(error);
        cerr<< "set formate7 configure failed"<<endl;
        return -1;
    }
    return 0;
}

int PointGreyCamera::init()
{
    error = busMgr.GetNumOfCameras(&numCameras);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }
    cout << "Number of cameras detected: " << numCameras << endl;
    if(numCameras==0)
    {
        cerr<<"can not detecte any camera!"<<endl;
        return -1;
    }
    cout<< "use the first camera!"<<endl;

    error = busMgr.GetCameraFromIndex(0, &guid);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }

    error = cam.Connect(&guid);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }

    // Get the camera information
    CameraInfo camInfo;
    error = cam.GetCameraInfo(&camInfo);
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }

    if(-1==setImageAttr())
    {
        cerr<<"set color failed!"<<endl;
        return -1;
    }

    if(-1==setFrame(20.0))
    {
        cerr<<"set frame failed!"<<endl;
        return -1;
    }

    EmbeddedImageInfo info;
    error=cam.GetEmbeddedImageInfo(&info);
    if(error!=PGRERROR_OK)
    {
        cout<<"get embedded image info error!"<<endl;

    }
    else
    {
        info.timestamp.onOff=true;
        error=cam.SetEmbeddedImageInfo(&info);
        if(error!=PGRERROR_OK)
        {
            cerr<<"set timestamp error"<<endl;
        };

    }

    printCameraInfo(&camInfo);

    error = cam.StartCapture();
    if (error != PGRERROR_OK)
    {
        printError( error );
        return -1;
    }
    return 0;
}

