#include <opencv2/opencv.hpp>

using namespace cv;

int main() {

    for (int angle = 0; angle < 181; angle++) {
        Mat image(600, 1000, CV_8U, Scalar(0));
        RotatedRect rRect = RotatedRect(Point2f(1000 / 2,600 / 2), Size2f(300, 100), angle);
        Point2f vertices[4];
        rRect.points(vertices);
        image.setTo(Scalar(255 ,255, 255));
        for (int i = 0; i < 4; i++)
            line(image, vertices[i], vertices[(i+1)%4], Scalar(0 , 0, 0), 3);

        // Rect brect = rRect.boundingRect();
        // rectangle(image, brect, Scalar(255,0,0));
        // imshow("rectangles", image);
        imwrite("testRect" + std::to_string(angle) + ".png", image);
        // waitKey(0);
    }
}