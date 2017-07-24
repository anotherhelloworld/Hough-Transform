#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <queue>
#include <string>

using namespace std;
using namespace cv;

enum class HoughType {
    LINES,
    CIRCLES,
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

struct Params {
    int phi;
    int r;
    Params(int phi, int r): phi(phi), r(r) {};
    bool operator < (const Params& right) const { return make_pair(r, phi) < make_pair(right.r, right.phi); }
};

struct Bfs {

    Mat used;
    Mat mat;
    vector <Mat> debug;
    vector <vector <Cell>> components;

    Bfs(Mat& mat): mat(mat) {
        used = Mat::zeros(mat.rows, mat.cols, CV_32SC1);
    }


    void bfs(Cell s) {

        int dcols[4] = {1, 0, -1, 0};
        int drows[4] = {0, 1, 0, -1};
        vector<Cell> res;
        queue<Cell> q;
        q.push(s);
        used.at<int>(s.row, s.col) = 1;
        while (!q.empty()) {
            Cell cell = q.front();
            res.push_back(cell);
            q.pop();
            // cout << cell.row << "   " << cell.col << endl;
            for (int i = 0; i < 4; i++) {
                int drow = cell.row + drows[i];
                int dcol = cell.col + dcols[i];
                if (drow >= 0 && drow < mat.rows && dcol >= 0 && dcol < mat.cols && 
                    used.at<int>(drow, dcol) == 0 && mat.at<uchar>(drow, dcol) > 0) {
                    used.at<int>(drow, dcol) = 1;
                    q.push(Cell(drow, dcol));
                }
            }
        }

        Mat tmp;
        if (res.size() > 20) {
            cvtColor(mat, tmp, CV_GRAY2RGB);
            for (auto cell : res) {
                tmp.at<Vec3b>(cell.row, cell.col)[0] = 0;
                tmp.at<Vec3b>(cell.row, cell.col)[1] = 0;
                tmp.at<Vec3b>(cell.row, cell.col)[2] = 255;
            }
            debug.push_back(tmp);
            components.push_back(res);
        }
    }

    void start() {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                if (used.at<int>(i, j) == 0 && mat.at<uchar>(i, j) > 0) {
                    bfs(Cell(i, j));
                }
            }
        }
    }
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

void threshHold(Mat& mat, uchar value) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (mat.at<uchar>(i, j) < value) {
                mat.at<uchar>(i, j) = 0;
            }
        }
    }
}

void drawLine(set <Params> lines, Mat& rgb, int offset) {
    for (auto line_ : lines) {
        double r = line_.r - offset, phi = line_.phi;
        
        Point pt1, pt2;
        double phi_rad = (phi * CV_PI) / 180.0;
        // phi = (phi_rad * 180.0) / cv_pi;
        double a = cos(phi_rad), b = sin(phi_rad);
        double x0 = a*r, y0 = b*r;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(rgb, pt1, pt2, Scalar(0,0,255), 3);
    }
}

set <Params> findLocalMax(const Mat& mat, int windowSize) {
    assert(windowSize <= mat.rows);
    assert(windowSize <= mat.cols);
    set <Params> res;
    for (int i = windowSize; i <= mat.rows - windowSize; i += 1) {
        for (int j = windowSize; j <= mat.cols - windowSize; j += 1) {
            Rect roi(j - windowSize, i - windowSize, windowSize * 2, windowSize * 2);

            Mat roiMat = mat(roi);

            double _min, _max;
            Point minLoc, maxLoc;
            minMaxLoc(roiMat, &_min, &_max, &minLoc, &maxLoc);

            for (int k = i - windowSize; k < i + windowSize; k++) {
                for (int l = j - windowSize; l < j + windowSize; l++) {
                    if ((maxLoc.x + (j - windowSize) == l) && (maxLoc.y + (i - windowSize) == k)) {
                        if (mat.at<uchar>(k, l) > 0) {
                            res.emplace(k, l);
                        }
                    }
                }
            }
        }
    }
    return res;
}

Mat FindEdges(const Mat& mat, Mat& angles, HoughType houghType) {
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

            if (houghType == HoughType::LINES && alpha < 0) {
                alpha += CV_PI;
            }

            int alpha_grad = (alpha * 180.0) / CV_PI;
            // cout << alpha_grad << endl;
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

Mat HoughLineMy(const Mat& I, Mat& edgesChar, int& offset, Mat& angles) {
    int phi = 180;
    offset = cvRound(sqrt((double)(I.rows * I.rows + I.cols * I.cols)));

    Mat C = Mat::zeros(phi + 1, offset * 2, CV_32SC1);

    for (int i = 0; i < edgesChar.rows; i++) {
        for (int j = 0; j < edgesChar.cols; j++) {
            if (edgesChar.at<uchar>(i, j) == 0) {
                continue;
            }

            int curAngle = angles.at<int>(i, j);
            int bound = 5;
            int start = (curAngle <= bound) ? 0 : curAngle - bound;
            int finish = (curAngle + bound >= 180) ? 180 : curAngle + bound;

            // cout << start << "   " << curAngle << "   " << finish << endl;

            for (int angle = start; angle < finish; angle++) {
                double theta = (angle * CV_PI) / 180.0;
                int cur_r = i * sin(theta) + j * cos(theta);
                C.at<int>(angle, cur_r + offset)++;
            }
        }
    }

    Mat H = intToUchar(C);

    return H;
}

Mat HoughCirclesMy(const Mat& I, Mat& edgesChar, int R, Mat& angles) {

    Mat C = Mat::zeros(I.rows, I.cols, CV_32SC1);

    for (int i = 0; i < edgesChar.rows; i++) {
        for (int j = 0; j < edgesChar.cols; j++) {
            if (edgesChar.at<uchar>(i, j) == 0) {
                continue;
            }

            int curAngle = angles.at<int>(i, j);
            int bound = 15;
            int start = curAngle - 15;
            int finish = curAngle + 15;

            for (int r = R - 10; r < R + 10; r++) {
                for (int angle = start; angle < finish; angle++) {
                    double theta = (angle * CV_PI) / 180.0;
                    int x = r * cos(theta) + j;
                    int y = r * sin(theta) + i;
                    if (x >= 0 && x < C.cols && y < C.rows && y >= 0)
                        C.at<int>(y, x)++;
                }

                start = curAngle - 15 + 180;
                finish = curAngle + 15 + 180;

                for (int angle = start; angle < finish; angle++) {
                    double theta = (angle * CV_PI) / 180.0;
                    int x = r * cos(theta) + j;
                    int y = r * sin(theta) + i;
                    if (x >= 0 && x < C.cols && y < C.rows && y >= 0)
                        C.at<int>(y, x)++;
                }
            }
        }
    }

    Mat H = intToUchar(C);

    return H;
}

vector <Cell> countAverage(Bfs& bfs) {
    vector <Cell> coords;
    for (auto component: bfs.components) {
        int averagerow = 0, averagecol = 0;

        for (auto cell: component) {
            averagerow += cell.row;
            averagecol += cell.col;
        }
        averagerow /= (int)component.size();
        averagecol /= (int)component.size();

        coords.emplace_back(averagerow, averagecol);
    }
    return coords;
}

void drawCircles(Mat& mat, vector <Cell> circleCoords, int R) {
    for (auto elem: circleCoords) {
        circle(mat, Point(elem.col, elem.row), R, Scalar(0, 0, 255));
    }
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
    // cout << "lol" << endl;

    int phiScaled = phi / scaleAngle;
    // cout << phiScaled << "    " << C.size() << endl;
    double _norm = sqrt((p1.row - p0.row) * (p1.row - p0.row) + (p1.col - p0.col) * (p1.col - p0.col));
    for (double t = 0; t <= 1; t += scale / (double)(_norm)) {
    // for (double t = 0; t <= 1; t += 0.02) {
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
    // cout << "kek" << endl;
}

void runRectangle(const Mat& I, vector <Mat>& Cangles, int scale, int scaleAngle, IPoint& curMax, int phi, int R, 
    double k, int i, int j) {
    for (int r = R - 2; r <=  R + 2; r++) {

        // int curAngle = angles.at<int>(i, j) + phi;
        int curAngle = phi;
        int bound = 15;

        int start = curAngle - bound;
        int finish = curAngle + bound;
        // cout << "123" << endl;
        for (int angle = start; angle <= finish; angle += scaleAngle) {
            int curHeight = r;
            int curWidth = k * r;
            // cout << curHeight << "  " << curWidth << "  " << k << endl;

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
        // cout << "asdasdas" << endl;
    }
}

vector <Mat> HoughRectMy(const Mat& I, const Mat& edgesChar, int height, int width, const Mat& angles, 
    int scale, int scaleAngle, IPoint& curMax) {
    Mat C = Mat::zeros(I.rows / scale, I.cols / scale, CV_32SC1);

    int MaxAngle = (180 + 1) / scaleAngle;
    // cout << MaxAngle << endl;
    // cout << "das;fk;sldkf;lsdk;lfk;dslkf;lksd;k" << endl;
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
        // cout << angle << "  " << Cmax << endl;
    }

    for (int i = 0; i <= MaxAngle; i++) {
        Hangles[i] = intToUcharGlobalMax(Cangles[i], curMax.value);
    }
    // Mat H = intToUchar(C);

    return Hangles;
}


void drawRect(Mat& mat, vector <Cell> rectCoords, int height, int width, int scale, int angle) {
    for (auto elem: rectCoords) {
        int col = elem.col * scale + scale / 2;
        int row = elem.row * scale + scale / 2;
        // rectangle(mat, Point(col - width, row - height), Point(col + width, row + height), Scalar(0, 0, 255));
        RotatedRect rRect = RotatedRect(Point2f(col, row), Size2f(width, height), angle);
        Point2f vertices[4];
        rRect.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(mat, vertices[i], vertices[(i+1)%4], Scalar(0, 0, 255), 3);
        }
    }
}

Mat SobelGrad(Mat& mat) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    Mat grad_x, grad_y;
    Mat angles = Mat::zeros(mat.rows, mat.cols, CV_32SC1);
    Sobel(mat, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(mat, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    for (int i = 0; i < angles.rows - 1; i++) {
        for (int j = 0; j < angles.cols - 1; j++) {

            int dx = grad_x.at<short>(i, j);
            int dy = grad_y.at<short>(i, j);
            double alpha = atan2(dy, dx);

            if (alpha < 0) {
                alpha += CV_PI;
            }

            int alpha_grad = (alpha * 180.0) / CV_PI;
            angles.at<int>(i, j) = alpha_grad;
        }
    }
    return angles;
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
    // blured = dilated;
    gate_hs_distr = blured;

    auto filtered = hsv_filter(T, gate_hs_distr);
    // auto filtered = imread(argv[1], 0);

    // imshow("filtered", filtered);
    Mat angles;

    Mat edgesChar = FindEdges(filtered, angles, HoughType::LINES);

    // imshow("grad", edgesChar);
    //Mat angles1 = SobelGrad(I);


    int offset = 0;

    //ширина 340
    //высота 308

    // IPoint curMax1 = IPoint(-1, Cell(-1, -1));
    // vector <Mat> C1 = HoughRectMy(I, edgesChar, 100 / 2, 300 / 2, angles, 5, curMax1);

    IPoint curMax2 = IPoint(-1, Cell(-1, -1));
    // cout << "as,hfdkjdf" << endl;
    int scaleAngle = 5;
    vector <Mat> C2 = HoughRectMy(I, edgesChar, 25, 300, 
        angles, 5, scaleAngle, curMax2);
    // vector <Mat> C2 = HoughRectMy(I, edgesChar, 300, 60, 
    //     angles, 5, scaleAngle, curMax2);


    Mat C3;
    int scaledAngle = curMax2.angle / scaleAngle;
    resize(C2[scaledAngle], C3, Size(C2[scaledAngle].cols * 5, 
        C2[scaledAngle].rows * 5));

    vector<Cell> points;
    points.push_back(curMax2.cell);
    // cout << curMax2.cell.row << "    " << curMax2.cell.col << "    " << curMax2.angle << endl;
    // cout << "dkasj;kasj" << endl;
    // cout << curMax2.angle << endl;
    drawRect(T, points, 2 * 25, 300, 5, curMax2.angle);
   // drawRect(T, points, 300, 60, 5, curMax2.angle);

    // Mat C1 = HoughLineMy(I, edgesChar, offset, angles);
    // threshHold(C1, 200);
    // set <Params> params = findLocalMax(C1, 4);
    // drawLine(params, T, offset);

    imshow("T", T);
    // imshow("C1", C1);
    // imshow("C1", C1[curMax1.angle]);
    // imshow("C2", C2[scaledAngle]);
    imshow("C3", C3);
    // imwrite("out" + string(argv[1]), T);
    waitKey(0);
    return 0;
}
