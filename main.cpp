#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

using namespace cv;
using namespace std;

void drawTestRect(Mat& image, Point2f center, Size2f size, int angle) {
    RotatedRect rRect = RotatedRect(center, size, angle);
    Point2f vertices[4];
    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
        line(image, vertices[i], vertices[(i+1)%4], Scalar(0 , 0, 0), 3);
}

int main(int argc, char** argv)
{
//    Mat img, gray;
//    if( argc != 2 || !(img=imread(argv[1], 1)).data)
//        return -1;
//    cvtColor(img, gray, COLOR_BGR2GRAY);
//    // smooth it, otherwise a lot of false circles may be detected
//    GaussianBlur( gray, gray, Size(9, 9), 2, 2 );
//    vector<Vec3f> circles;
//    HoughCircles(gray, circles, HOUGH_GRADIENT,
//                 2, gray.rows/4, 200, 100 );
//    for( size_t i = 0; i < circles.size(); i++ )
//    {
//        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//        int radius = cvRound(circles[i][2]);
//        // draw the circle center
//        circle( img, center, 3, Scalar(0,255,0), -1, 8, 0 );
//        // draw the circle outline
//        circle( img, center, radius, Scalar(0,0,255), 3, 8, 0 );
//    }
//    namedWindow( "circles", 1 );
//    imshow( "circles", img );

//    waitKey(0);



    Mat img, gray;
    if( argc != 2 || !(img=imread(argv[1], 1)).data)
        return -1;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(9, 9), 2, 2 );
    vector<Vec6f> rects;

    HoughRects(gray, rects, 100, 300, 5, 5);

    return 0;
}