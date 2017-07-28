#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <string>

using namespace std;
using namespace cv;

struct HoughRectSettings {
    string imagePath;
    int height;
    int width;
    int scaleAngle = 5;
    int scale = 5;
};

struct Cell {
    int row; 
    int col;
    Cell(int row = -1, int col = -1): row(row), col(col) {};
};

struct IPoint {
    int value;
    Cell cell;
    int angle;
    IPoint(int value, Cell cell): value(value), cell(cell) {};
    bool operator < (const IPoint& r) const { return value > r.value; }
};

Mat intToUchar(const Mat& mat) {
    double _min = 0, _max = 0;
    minMaxLoc(mat, &_min, &_max);

    Mat res = Mat::zeros(mat.rows, mat.cols, CV_8UC1);
    // assert (_max > 0);
    if (_max <= 0) {
        return res;
    }
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            res.at<uchar>(i, j) = (255 * mat.at<int>(i, j)) / (int)_max;
        }
    }
    return res;
}

Mat intToUcharGlobalMax(const Mat& mat, int _max) {
    
    Mat res = Mat::zeros(mat.rows, mat.cols, CV_8UC1);
    // assert (_max > 0);
    if (_max <= 0) {
        return res;
    }
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            res.at<uchar>(i, j) = (255 * mat.at<int>(i, j)) / (int)_max;
        }
    }
    return res;
}

Mat FindEdges(const Mat& mat, Mat& angles) {
    Mat deltaI = Mat::zeros(mat.rows - 1, mat.cols - 1, CV_32SC1);
    vector<IPoint> filtering;

    int dx = 0;
    int dy = 0;
    int tmp = 0;

    angles = Mat::zeros(deltaI.rows, deltaI.cols, CV_32SC1);

    for (int i = 0; i < mat.rows - 1; i++) {
        for (int j = 0; j < mat.cols - 1; j++) {

            dx = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i, j + 1);
            dy = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i + 1, j);
            tmp = dx * dx + dy * dy;
            double alpha = atan2(dy, dx);

            if (alpha < 0) {
                alpha += CV_PI;
            }

            int alpha_grad = (alpha * 180.0) / CV_PI;
            angles.at<int>(i, j) = alpha_grad;
            deltaI.at<int>(i, j) = tmp;
            filtering.push_back(IPoint(tmp, Cell(i, j)));
        }
    }

    Mat deltaIConvert = intToUchar(deltaI);

    sort(filtering.begin(), filtering.end());

    vector<IPoint> filtered = vector<IPoint>(filtering.begin(), filtering.begin() + filtering.size() * 0.005);

    Mat edges = Mat::zeros(deltaI.rows, deltaI.cols, CV_32SC1);

    for (auto point: filtered) {
        edges.at<int>(point.cell.row, point.cell.col) = point.value;
    }

    Mat edgesChar = intToUchar(edges);

    return edgesChar;
}

vector <Cell> rotateRect(vector <Cell> rectCoords, Cell centr, int angle) {
    vector<Cell> res(rectCoords.size());
    for (int i = 0; i < rectCoords.size(); i++) {
        rectCoords[i].row -= centr.row;
        rectCoords[i].col -= centr.col;
    }

    for (int i = 0; i < res.size(); i++) {
        double theta = (angle * CV_PI) / 180.0;
        res[i].row = rectCoords[i].col * sin(theta) + rectCoords[i].row * cos(theta);
        res[i].col = rectCoords[i].col * cos(theta) - rectCoords[i].row * sin(theta);
    }

    for (int i = 0; i < res.size(); i++) {
        res[i].row += centr.row;

        res[i].col += centr.col;
    }

    return res;
}

void runAlongLine(const Mat& I, vector <Mat>& C, Cell p0, Cell p1, int scale, int scaleAngle, IPoint& curMax, int phi) {
    if (phi < 0) {
        phi += 180;
    } else if (phi > 180) {
        phi -= 180;
    }

    int phiScaled = phi / scaleAngle;
    double _norm = sqrt((p1.row - p0.row) * (p1.row - p0.row) + (p1.col - p0.col) * (p1.col - p0.col));
    for (double t = 0; t <= 1; t += scale / (double)(_norm)) {
        Cell p = Cell(floor(p0.row + (p1.row - p0.row) * t), floor(p0.col + (p1.col - p0.col) * t));
        if (p.row < I.rows && p.row > 0 && p.col < I.cols && p.col > 0) {
            C[phiScaled].at<int>(p.row / scale, p.col / scale)++;
            if (C[phiScaled].at<int>(p.row / scale, p.col / scale) > curMax.value) {
                curMax.value = C[phiScaled].at<int>(p.row / scale, p.col / scale);
                curMax.cell.row = p.row / scale;
                curMax.cell.col = p.col / scale;
                curMax.angle = phi;///?
            }
        }
    }
}

void runRectangle(const Mat& I, vector <Mat>& Cangles, int scale, int scaleAngle, IPoint& curMax, int phi, int R, 
    double k, int i, int j) {
    for (int r = R - 2; r <=  R + 2; r++) {
        int curAngle = phi;
        int bound = 15;

        int start = curAngle - bound;
        int finish = curAngle + bound;
        for (int angle = start; angle <= finish; angle += scaleAngle) {
            int curHeight = r;
            int curWidth = k * r;

            Cell ptl = Cell(i - curHeight, j - curWidth);
            Cell ptr = Cell(i - curHeight, j + curWidth);
            Cell pbr = Cell(i + curHeight, j + curWidth);
            Cell pbl = Cell(i + curHeight, j - curWidth);

            vector <Cell> rectCoords = { ptl, ptr, pbr, pbl };
            vector <Cell> rotateCoords = rotateRect(rectCoords, Cell(i, j), angle);

            runAlongLine(I, Cangles, rotateCoords[0], rotateCoords[1], scale, scaleAngle, curMax, angle);
            runAlongLine(I, Cangles, rotateCoords[1], rotateCoords[2], scale, scaleAngle, curMax, angle);
            runAlongLine(I, Cangles, rotateCoords[2], rotateCoords[3], scale, scaleAngle, curMax, angle);
            runAlongLine(I, Cangles, rotateCoords[3], rotateCoords[0], scale, scaleAngle, curMax, angle);
        }
    }
}

vector <Mat> HoughRectMy(const Mat& I, const Mat& edgesChar, int height, int width, const Mat& angles, 
    int scale, int scaleAngle, IPoint& curMax) {
    Mat C = Mat::zeros(I.rows / scale, I.cols / scale, CV_32SC1);

    int MaxAngle = (180 + 1) / scaleAngle;
    vector <Mat> Cangles(MaxAngle + 1);
    for (int i = 0; i <= MaxAngle; i ++) {
        Cangles[i] = Mat::zeros(I.rows / scale, I.cols / scale, CV_32SC1);
    }

    int R = height;
    double k = ((double)width / (double)height);
    for (int i = 0; i < edgesChar.rows; i++) {
        for (int j = 0; j < edgesChar.cols; j++) {
            if (edgesChar.at<uchar>(i, j) == 0) {
                continue;
            }
            runRectangle(I, Cangles, scale, scaleAngle, curMax, angles.at<int>(i, j), R, k, i, j);
            runRectangle(I, Cangles, scale, scaleAngle, curMax, angles.at<int>(i, j) - 90, R, k, i, j);
        }
    }

    vector <Mat> Hangles(MaxAngle + 1);

    for (int angle = 0; angle < Cangles.size(); angle++) {
        int Cmax = 0;
        for (int i = 0; i < Cangles[angle].rows; i++) {
            for (int j = 0; j < Cangles[angle].cols; j++) {
                Cmax = max(Cmax, (int)Cangles[angle].at<uchar>(i, j));
            }
        }
    }

    for (int i = 0; i <= MaxAngle; i++) {
        Hangles[i] = intToUcharGlobalMax(Cangles[i], curMax.value);
    }

    return Hangles;
}

void drawRect(Mat& mat, vector <Cell> rectCoords, int height, int width, int scale, int angle) {
    for (auto elem: rectCoords) {
        int col = elem.col * scale + scale / 2;
        int row = elem.row * scale + scale / 2;
        RotatedRect rRect = RotatedRect(Point2f(col, row), Size2f(width, height), angle);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(mat, vertices[i], vertices[(i+1)%4], Scalar(0, 0, 255), 3);
        }
    }
}

Mat hsv_filter(const cv::Mat& img, const cv::Mat& hsv_distr) {
    Mat hsv_img;
    cvtColor(img, hsv_img, CV_BGR2HSV);
    Mat res = Mat::zeros(hsv_img.rows, hsv_img.cols, CV_8U);
    for (int i = 0; i < hsv_img.rows; i++) {
        for (int j = 0; j < hsv_img.cols; j++) {
            auto hsv = hsv_img.at<Vec3b>(i, j);
            res.at<uchar>(i, j) = hsv_distr.at<uchar>(hsv[0], hsv[1]);
        }
    }

    return res;
}

int main(int argc, char** argv) {
    if ( argc != 2 ) {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat I, T, src;
    src = imread(argv[1]);
    GaussianBlur(src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    T = src;
    // resize(src, T, Size(400, 300));

    cvtColor(T, I, CV_BGR2GRAY);

    Mat gate_hs_distr = imread("orange_lane_hs.png", 0);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
    Mat dilated, blured;
    dilate(gate_hs_distr, dilated, element);
    blur(dilated, blured, Size(7, 7));
    gate_hs_distr = blured;

    auto filtered = hsv_filter(T, gate_hs_distr);
    Mat angles;

    Mat edgesChar = FindEdges(filtered, angles);
    int offset = 0;

    IPoint curMax2 = IPoint(-1, Cell(-1, -1));
    int scaleAngle = 5;
    vector <Mat> C2 = HoughRectMy(I, edgesChar, 25, 300,
        angles, 5, scaleAngle, curMax2);


    Mat C3;
    int scaledAngle = curMax2.angle / scaleAngle;
    resize(C2[scaledAngle], C3, Size(C2[scaledAngle].cols * 5, 
        C2[scaledAngle].rows * 5));

    vector<Cell> points;
    points.push_back(curMax2.cell);
    drawRect(T, points, 2 * 25, 300, 5, curMax2.angle);

    imshow("T", T);
    imshow("C3", C3);
    waitKey(0);
    return 0;
}
