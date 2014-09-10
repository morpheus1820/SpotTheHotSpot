#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;

int skip=5;                     // skip n frames
int segm_thresh=150;            // fixed threshold for modules segmentation
int erosion_size=19;            // kernel size for image opening
bool stop=false;                // to stop after each frames (debug)

char text[10];
string filename_out = "out.mov";// video file for saving results


int main(int argc, char* argv[])
{
    namedWindow("Panels", 1);
    createTrackbar( "segm thresh", "Panels", &segm_thresh, 255, NULL );
    createTrackbar( "element", "Panels", &erosion_size, 21, NULL );

    VideoCapture cap(argv[1]);
    VideoWriter output;
    Mat frame;
    cap >> frame;

    output.open ( filename_out, CV_FOURCC('D','I','V','X'), 15, frame.size(), true );
    cap.release();
    cap.open(filename);
    vector<int> y_dists_total;
    int panel_count=0;

    std::ostringstream oss;
    oss << "Hotspot rows: ";

    do
    {
        for(int i=0;i<skip+1;i++)
            cap >> frame;
        //resize(frame,frame,Size(320,240));
        Mat frame_bw(frame.rows,frame.cols,CV_8UC1);
        cvtColor(frame,frame_bw,CV_BGR2GRAY);
        Mat frame_bw_orig=frame_bw.clone();

        GaussianBlur(frame_bw,frame_bw, Size(19,19),0.5);
        threshold( frame_bw, frame_bw, segm_thresh, 255,0 );


        Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );
        erode( frame_bw, frame_bw, element );
        dilate( frame_bw, frame_bw, element );


        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        int thresh = 100;

        findContours( frame_bw, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        //filter small contours
        for (vector<vector<Point> >::iterator it = contours.begin(); it!=contours.end(); )
        {
            if ( contourArea(*it)<5000)
                it=contours.erase(it);
            else
                ++it;
        }

        /// Draw contours
        Mat drawing = frame.clone();// Mat::zeros( canny_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            drawContours( drawing, contours, i, Scalar(255,255,255), 2, 8, hierarchy, 0, Point() );
        }
        /// Approximate contours to polygons + get bounding rects and circles
        Mat threshold_output;

        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );
        for( int i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }

        // // filter dupicate centers
        // for( int i = 0; i<center_temp.size(); i++ )
        // {
        //     for( int j = 0; j< center_temp.size(); j++ )
        //     if(i!=j)
        //     {

        //         if( sqrt( (center_temp[i].x-center_temp[j].x)*(center_temp[i].x-center_temp[j].x) +
        //             (center_temp[i].y-center_temp[j].y)*(center_temp[i].y-center_temp[j].y) ) < 10 )  // if dist i - j too small
        //                 {center_temp.erase (center_temp.begin()+j);
        //                     center_temp.erase (center_temp.begin()+j);
        //                     j++;}
        //     }
        // }


        // cycle on current contours
        vector<int> y_dists;

        bool found_hotspots=false;
        int last_hotspot=0;

        for( int i = 0; i< contours.size(); i++ )
        {
            // Draw polygonal contour + bonding rects + circles (for debug)

            drawContours( drawing, contours_poly, i, Scalar(255,255,255), 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), Scalar(255,255,255), 2, 8, 0 );
            circle( drawing, center[i], 2, Scalar(255,255,255), 2, 8, 0 );

            int dist=abs(center[i].y-frame.rows/2);
            y_dists.push_back(dist);





            // Mat labels=frame_bw_orig.clone();
            // labels.setTo(0);
            // unsigned char cont_avg;
            // unsigned char cont_std;
            //     cv::drawContours(labels, contours, i, cv::Scalar(i), CV_FILLED);
            //
            //     Scalar mean;
            //     Scalar stdev;
            //      //= cv::mean(frame_bw_orig(roi), labels(roi) == x);
            //      meanStdDev(frame_bw_orig(roi),mean, stdev,labels(roi) == i);
            //     cont_avg = mean[0];
            //     cont_std =stdev[0];


            // find hotspots ---------------------------


            // find peaks
            cv::Rect roi = boundRect[i];
            Mat segmented=frame_bw_orig(roi).clone();



            // Mat peak_img = segmented.clone();
            // dilate(peak_img,peak_img,Mat(),Point(-1,-1),8);
            // peak_img = peak_img - segmented;



            // Mat flat_img ;
            // erode(segmented,flat_img,Mat(),Point(-1,-1),8);
            // flat_img = segmented - flat_img;


            // threshold(peak_img,peak_img,0,255,CV_THRESH_BINARY);
            // threshold(flat_img,flat_img,0,255,CV_THRESH_BINARY);
            // bitwise_not(flat_img,flat_img);

            // peak_img.setTo(Scalar::all(255),flat_img);
            // bitwise_not(peak_img,peak_img);
            // imshow("peaks",peak_img);

            // for(int x=0;x<roi.width;x++)
            //     for(int y=0;y<roi.height;y++)
            //     {
            //         if (pointPolygonTest(contours_poly[i], Point2f(x+roi.x,y+roi.y), true)>5) // point inside
            //         {
            //             unsigned char col=peak_img.at<unsigned char>(y,x);

            //             if(col==255)
            //                 circle( drawing, Point(x+roi.x,y+roi.y), 2, Scalar(50,50,255), -1);


            //         }

            //     }



            // adaptive thresh
            adaptiveThreshold(segmented, segmented, 255, CV_THRESH_BINARY, CV_ADAPTIVE_THRESH_MEAN_C,51,-50);

            // filter out blobls with wrong shape or too small or too large
            vector<vector<Point> > contours2;
            vector<Vec4i> hierarchy2;
            int thresh = 100;
            findContours( segmented, contours2, hierarchy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );

            for( int n = 0; n< contours2.size(); n++ )
            {
                Rect rect=boundingRect(contours2[n]);
                double ratio;
                ratio=(double)rect.width/(double)rect.height;

                // draw hotspots in red, discarded hotspots in white
                if(ratio>0.5 && ratio<1.5 && contourArea(contours2[n])> 50 && contourArea(contours2[n])< 2000
                    && rect.y+roi.y< 400) // for dicarding writings...
                   { drawContours( drawing, contours2, n, Scalar(0,0,255), 2, 8, hierarchy2, 0, Point(roi.x,roi.y) );
                    found_hotspots=true;
                   }
                else
                    drawContours( drawing, contours2, n, Scalar(255,255,255), 2, 8, hierarchy2, 0, Point(roi.x,roi.y) );

            }


            // char text[10];
            // sprintf(text,"%d",i);
            // imshow(text, segmented );




            // histogram--------------------------------
            //     int histSize = 256;

            //     /// Set the ranges ( for B,G,R) )
            //     float range[] = { 0, 256 } ;
            //     const float* histRange = { range };

            //     bool uniform = true; bool accumulate = false;

            //     Mat hist, g_hist, r_hist;

            //     /// Compute the histograms:
            //     Mat temp=frame_bw_orig(boundRect[i]).clone();
            //     calcHist( &temp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

            //     // Draw the histograms for B, G and R
            //     int hist_w = 200; int hist_h = 100;
            //     int bin_w = cvRound( (double) hist_w/histSize );

            //     Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

            //     /// Normalize the result to [ 0, histImage.rows ]
            //     normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );


            //     /// Draw for each channel
            //     for( int n = 1; n < histSize; n++ )
            //     {
            //         line( histImage, Point( bin_w*(n-1), hist_h - cvRound(hist.at<float>(n-1)) ) ,
            //                          Point( bin_w*(n), hist_h - cvRound(hist.at<float>(n)) ),
            //                          Scalar( 255, 255, 255), 1, 8, 0  );

            //     }

            //     /// Display
            //     if(i<9){
            //     sprintf(text,"hist %d",i);
            //     imshow(text, histImage );



            // }
            //------------------------------------------

            // label current panels

            sprintf(text,"%d",i);
            cv::putText(drawing, text, center[i], CV_FONT_HERSHEY_SIMPLEX, 1, Scalar::all(255));
        }


        if(found_hotspots && panel_count!=last_hotspot)
        {
            oss << panel_count << ",";
            last_hotspot=panel_count;
        }
        cv::putText(drawing, oss.str(), Point(10,50), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255));

        // sort panel distances from y=0
        std::sort ( y_dists.begin(),  y_dists.end());
        int avg_dist=(int)((y_dists[0]+y_dists[1]+y_dists[2])/3.0);
        y_dists_total.push_back(avg_dist);

        // detect new row of panels
        if(y_dists_total.size()>2 && y_dists_total[y_dists_total.size()-2]< y_dists_total[y_dists_total.size()-1] &&
            y_dists_total[y_dists_total.size()-2]<y_dists_total[y_dists_total.size()-3])
            panel_count++;

        // write debug info
        char text[10];
        sprintf(text,"Panel rows: %d",panel_count);
        cv::putText(drawing, text, Point(10,30), CV_FONT_HERSHEY_SIMPLEX, 0.8, Scalar::all(255));

        imshow("Panels", drawing);
        output.write ( drawing );

        char c;
        if(stop)
            c=waitKey(0);
        else
            c=waitKey(100);

        if(c==27) break;
        if(c==' ') stop=!stop;

    } while(cap.isOpened());

    output.release();
    return 0;
}


