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
        line(image, vertices[i], vertices[(i+1)%4], Scalar(0, 0, 255), 3);
}

int main(int argc, char** argv)
{

    Mat img, gray;
    if( argc != 2 || !(img=imread(argv[1], 1)).data)
        return -1;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(9, 9), 2, 2);
    Mat rects;

    HoughRects(gray, rects, 1, 100, 300);

    for (int i = 0; i < rects.rows; i++) {
        drawTestRect(img, Point2f(rects.at<float>(i, 0), rects.at<float>(i, 1)), Size2f(rects.at<float>(i, 2), rects.at<float>(i, 3)), rects.at<float>(i, 4));
    }

    imshow("image", img);

    waitKey(0);
    return 0;
}