#include <iostream>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

enum Method
{
    FGD_STAT,
    MOG,
    MOG2,
    GMG
};

int main(int argc, const char** argv)
{
    cv::CommandLineParser cmd(argc, argv,
        "{ c | camera | false       | use camera }"
        "{ f | file   | 768x576.avi | input video file }"
        "{ m | method | mog2         | method (fgd, mog, mog2, gmg) }"
        "{ h | help   | false       | print help message }");

    if (cmd.get<bool>("help"))
    {
        cout << "Usage : bgfg_segm [options]" << endl;
        cout << "Avaible options:" << endl;
        cmd.printParams();
        return 0;
    }

    bool useCamera = cmd.get<bool>("camera");
    string file = cmd.get<string>("file");
    string method = cmd.get<string>("method");

    if (method != "fgd"
        && method != "mog"
        && method != "mog2"
        && method != "gmg")
    {
        cerr << "Incorrect method" << endl;
        return -1;
    }

    Method m = method == "fgd" ? FGD_STAT :
               method == "mog" ? MOG :
               method == "mog2" ? MOG2 :
                                  GMG;

    VideoCapture cap;

    if (useCamera)
        cap.open(0);
    else
        cap.open(file);

    if (!cap.isOpened())
    {
        cerr << "can not open camera or video file" << endl;
        return -1;
    }

    Mat frame;
    cap >> frame;

    // imwrite( "background.jpg", frame );

    // Mat background = imread("background.jpg");

    GpuMat d_frame(frame);

    FGDStatModel fgd_stat;
    MOG_GPU mog;
    MOG2_GPU mog2;
    GMG_GPU gmg;
    gmg.numInitializationFrames = 40;

    // double alpha = 0.5; double beta; double input;

    GpuMat d_fgmask;
    GpuMat d_fgimg;
    GpuMat d_bgimg;

    Mat fgmask;
    Mat fgimg;
    Mat bgimg;

    switch (m)
    {
    case FGD_STAT:
        fgd_stat.create(d_frame);
        break;

    case MOG:
        mog(d_frame, d_fgmask, 0.01f);
        break;

    case MOG2:
        mog2(d_frame, d_fgmask);
        break;

    case GMG:
        gmg.initialize(d_frame.size());
        break;
    }

    namedWindow("image", WINDOW_NORMAL);
    namedWindow("foreground mask", WINDOW_NORMAL);
    namedWindow("foreground image", WINDOW_NORMAL);
    if (m != GMG)
    {
        namedWindow("mean background image", WINDOW_NORMAL);
    }
    int pic_seq=0;
    for(;;)
    {
        cap >> frame;
        if (frame.empty())
            break;
        d_frame.upload(frame);

        int64 start = cv::getTickCount();

        //update the model
        switch (m)
        {
        case FGD_STAT:
            fgd_stat.update(d_frame);
            d_fgmask = fgd_stat.foreground;
            d_bgimg = fgd_stat.background;
            break;

        case MOG:
            mog(d_frame, d_fgmask);
            mog.getBackgroundImage(d_bgimg);
            break;

        case MOG2:
            mog2(d_frame, d_fgmask);
            mog2.getBackgroundImage(d_bgimg);
            break;

        case GMG:
            gmg(d_frame, d_fgmask);
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;

        d_fgimg.create(d_frame.size(), d_frame.type());
        d_fgimg.setTo(Scalar::all(0));
        d_frame.copyTo(d_fgimg, d_fgmask);

        d_fgmask.download(fgmask);
        d_fgimg.download(fgimg);

        if (!d_bgimg.empty())
            d_bgimg.download(bgimg);
        // std::cout << "FPS : " << fgimg.size() << std::endl;
        // equalizeHist(fgmask,fgmask);
        medianBlur(fgmask, fgmask, 7);

        // medianBlur(fgimg, fgimg, 7);
         
        // bilateralFilter(fgmask,fgmask,5,1.0,1.0);
        imshow("foreground mask", fgmask);
        // imshow("foreground image", fgimg);

        // Canny( fgimg, fgimg,0 ,0, 3);
        // imshow("foreground mask2", fgimg);

        // Mat dst;
        // addWeighted( fgimg,1, fgmask,1,0, dst,-1);
        // imshow( "Linear Blend", dst );

        vector<vector<Point> > contours;
        findContours(fgmask,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE, Point(0, 0));
        std::cout << "find : " << contours.size() << std::endl;
        // for (size_t idx = 0; idx < contours.size(); idx++) {
        // if(contourArea(contours[idx]) >50){
        //         cout << "Area: " << contourArea(contours[idx]) << std::endl;
        //         rectangle(frame,);
        //         cv::drawContours(frame, contours, idx, Scalar(255,0,0));
        //     }
        // }
        // vector<Moments> mu(contours.size() );
        // for( int i = 0; i < contours.size(); i++ )
        //    { mu[i] = moments( contours[i], false ); }
      
        // ///  Get the mass centers:
        // vector<Point2f> mc( contours.size() );
        // for( int i = 0; i < contours.size(); i++ )
        //    { 
        //       mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); 
        //       cout << "Object: " << i << " center :" << mc[i] << endl;
        //    }

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
            { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
               boundRect[i] = boundingRect( Mat(contours_poly[i]) );
               minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
            }
        /// Draw contours
        // Mat drawing = Mat::zeros( detected_edges.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
           {
             if(contourArea(contours[i]) >1200){
                stringstream seq;
                seq << pic_seq;
                Rect r = boundRect[i];
                cv::Mat croppedImage = frame(r);
                cv::imwrite( "./image/Gray_Image"+seq.str()+".jpg", croppedImage );
                pic_seq++;
                cv::rectangle(frame, boundRect[i].tl(), boundRect[i].br(),  Scalar(255,0,0), 2, 8, 0 );
                // circle( frame, center[i], (int)radius[i], Scalar(255,255,0), 2, 8, 0 );
                // circle( frame, mc[i], 20, Scalar(255,0,0), 2, 8, 0 );
                }
             // circle( frame, mc[i], 20, Scalar(255,0,0), -1, 8, 0 );
           }
        imshow("image", frame);
        // if (!bgimg.empty())
            // imshow("mean background image", bgimg);

        int key = waitKey(30);
        if (key == 27)
            break;
    }

    return 0;
}
