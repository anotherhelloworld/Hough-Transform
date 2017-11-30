#include <iostream>
#include <type_traits>
#include <functional>
#include <sstream>
#include <map>
#include <iomanip>
#include <exception>
#include <iostream>
#include <cmath>
#include <array>
#include <set>
#include <stdio.h>
#include <vector>
#include <queue>
#include <string>

#include <opencv2/opencv.hpp>

using namespace cv;

class HoughLaneRecognizer
{
public:

    int hough_height;
    int hough_width;
    int hough_scale;
    int hough_scale_angle;

    HoughLaneRecognizer(
            int hough_height,
            int hough_width,
            int hough_scale,
            int hough_scale_angle
            )
            : hough_height(hough_height)
            , hough_width(hough_width)
            , hough_scale(hough_scale)
            , hough_scale_angle(hough_scale_angle)
    {
    }

    struct Cell
    {
        int row;
        int col;
        Cell(int row = -1, int col = -1)
                : row(row)
                , col(col)
        {};
    };

    struct AccumPoint
    {
        int value;
        int angle;
        Cell cell;
        AccumPoint(
                int value,
                Cell cell)
                : value(value)
                , cell(cell)
        {};
        bool operator < (const AccumPoint& r) const { return value > r.value; }
    };

    cv::Mat normalize_mat(const cv::Mat& mat, int _max)
    {
        cv::Mat res = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
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

    cv::Mat int_to_char(const cv::Mat& mat)
    {
        double _min = 0, _max = 0;
        cv::minMaxLoc(mat, &_min, &_max);
        return normalize_mat(mat, _max);
    }

    cv::Mat int_to_char_global_max(const cv::Mat& mat, int _max)
    {
        cv::Mat res = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
        return normalize_mat(mat, _max);
    }

    cv::Mat find_edges(const cv::Mat& mat, cv::Mat& angles)
    {
        cv::Mat delta_i = cv::Mat::zeros(mat.rows - 1, mat.cols - 1, CV_32SC1);
        std::vector<AccumPoint> filtering;

        angles = cv::Mat::zeros(delta_i.rows, delta_i.cols, CV_32SC1);

        for (int i = 0; i < mat.rows - 1; i++) {
            for (int j = 0; j < mat.cols - 1; j++) {

                int dx = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i, j + 1);
                int dy = (int)mat.at<uchar>(i, j) - (int)mat.at<uchar>(i + 1, j);
                int laplas = dx * dx + dy * dy;
                double alpha = atan2(dy, dx);

                if (alpha < 0) {
                    alpha += CV_PI;
                }

                int alpha_grad = (alpha * 180.0) / CV_PI;
                angles.at<int>(i, j) = alpha_grad;
                delta_i.at<int>(i, j) = laplas;
                filtering.push_back(AccumPoint(laplas, Cell(i, j)));
            }
        }

        sort(filtering.begin(), filtering.end());

        std::vector<AccumPoint> filtered(filtering.begin(), filtering.begin() + filtering.size() * 0.5);

        cv::Mat edges = cv::Mat::zeros(delta_i.rows, delta_i.cols, CV_32SC1);

        for (auto point: filtered) {
            edges.at<int>(point.cell.row, point.cell.col) = point.value;
        }

        cv::Mat edges_char = int_to_char(edges);

        return edges_char;
    }

    std::vector <Cell> rotate_rect(std::vector <Cell> rect_coords, Cell centr, int angle)
    {
        std::vector<Cell> res(rect_coords.size());
        for (int i = 0; i < rect_coords.size(); i++) {
            rect_coords[i].row -= centr.row;
            rect_coords[i].col -= centr.col;
        }

        for (int i = 0; i < res.size(); i++) {
            double theta = (angle * CV_PI) / 180.0;
            res[i].row = rect_coords[i].col * sin(theta) + rect_coords[i].row * cos(theta);
            res[i].col = rect_coords[i].col * cos(theta) - rect_coords[i].row * sin(theta);
        }

        for (int i = 0; i < res.size(); i++) {
            res[i].row += centr.row;
            res[i].col += centr.col;
        }

        return res;
    }

    void run_along_line(
            const cv::Mat& image,
            std::vector <cv::Mat>& accum,
            Cell start,
            Cell finish,
            int scale,
            int scale_angle,
            AccumPoint& max_accum,
            int angle)
    {
        if (angle < 0) angle += 180; else if (angle > 180) angle -= 180;

        int angleScaled = angle / scale_angle;
        double _norm = sqrt((finish.row - start.row) * (finish.row - start.row) +
                            (finish.col - start.col) * (finish.col - start.col));
        for (double i = 0; i <= 1; i += scale / (double)(_norm)) {
            Cell p = Cell(floor(start.row + (finish.row - start.row) * i),
                          floor(start.col + (finish.col - start.col) * i));
            if (p.row < image.rows && p.row > 0 && p.col < image.cols && p.col > 0) {
                accum[angleScaled].at<int>(p.row / scale, p.col / scale)++;
                if (accum[angleScaled].at<int>(p.row / scale, p.col / scale) > max_accum.value) {
                    max_accum.value = accum[angleScaled].at<int>(p.row / scale, p.col / scale);
                    max_accum.cell.row = p.row / scale;
                    max_accum.cell.col = p.col / scale;
                    max_accum.angle = angle;///?
                }
            }
        }
    }

    void run_rectangle(
            const cv::Mat& image,
            std::vector <cv::Mat>& accum,
            int scale,
            int scale_angle,
            AccumPoint& max_accum,
            int angle,
            int rad,
            double k,
            int row,
            int col)
    {
        for (int r = rad - 2; r <=  rad + 2; r++) {
            int bound = 15;

            int start = angle - bound;
            int finish = angle + bound;
            for (int angle = start; angle <= finish; angle += scale_angle) {
                int cur_height = r;
                int cur_width = k * r;

                Cell ptl = Cell(row - cur_height, col - cur_width);
                Cell ptr = Cell(row - cur_height, col + cur_width);
                Cell pbr = Cell(row + cur_height, col + cur_width);
                Cell pbl = Cell(row + cur_height, col - cur_width);

                std::vector <Cell> rect_coords = { ptl, ptr, pbr, pbl };
                std::vector <Cell> rotate_coords = rotate_rect(rect_coords, Cell(row, col), angle);

                run_along_line(image, accum, rotate_coords[0], rotate_coords[1], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[1], rotate_coords[2], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[2], rotate_coords[3], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[3], rotate_coords[0], scale, scale_angle, max_accum, angle);
            }
        }
    }

    std::vector <cv::Mat> hough_rect(
            const cv::Mat& image,
            const cv::Mat& edges_char,
            const cv::Mat&
            angles, AccumPoint& max_accum)
    {
        int max_angle = (180 + 1) / hough_scale_angle;
        std::vector <cv::Mat> angles_accum(max_angle + 1);
        for (int i = 0; i <= max_angle; i ++) {
            angles_accum[i] = cv::Mat::zeros(image.rows / hough_scale,
                                             image.cols / hough_scale, CV_32SC1);
        }

        double k = ((double)hough_width / (double)hough_height);
        for (int row = 0; row < edges_char.rows; row++) {
            for (int col = 0; col < edges_char.cols; col++) {
                if (edges_char.at<uchar>(row, col) == 0) {
                    continue;
                }
                run_rectangle(image, angles_accum, hough_scale, hough_scale_angle,
                              max_accum, angles.at<int>(row, col), hough_height, k, row, col);
                run_rectangle(image, angles_accum, hough_scale, hough_scale_angle,
                              max_accum, angles.at<int>(row, col) - 90, hough_height, k, row, col);
            }
        }

        std::vector <cv::Mat> angles_accum_char(max_angle + 1);

        for (int angle = 0; angle <= max_angle; angle++) {
            angles_accum_char[angle] = int_to_char_global_max(angles_accum[angle], max_accum.value);
        }

        return angles_accum_char;
    }

    void draw_rect(
            cv::Mat& mat,
            std::vector <Cell> rect_coords,
            int height,
            int width,
            int scale,
            int angle)
    {
        for (auto elem: rect_coords) {
            int col = elem.col * scale + scale / 2;
            int row = elem.row * scale + scale / 2;
            cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(col, row), cv::Size2f(width, height), angle);
            cv::Point2f vertices[4];
            rRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                line(mat, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 3);
            }
        }
    }

    cv::Mat hsv_filter(const cv::Mat& img, const cv::Mat& hsv_distr)
    {
        cv::Mat hsv_img;
        cv::cvtColor(img, hsv_img, CV_BGR2HSV);
        cv::Mat res = cv::Mat::zeros(hsv_img.rows, hsv_img.cols, CV_8U);
        for (int i = 0; i < hsv_img.rows; i++) {
            for (int j = 0; j < hsv_img.cols; j++) {
                auto hsv = hsv_img.at<cv::Vec3b>(i, j);
                res.at<uchar>(i, j) = hsv_distr.at<uchar>(hsv[0], hsv[1]);
            }
        }
        return res;
    }

    cv::Mat recognize(InputArray _src)
    {
        cv::Mat src;
        cv::Mat image;
        cv::GaussianBlur(_src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);

        cv::cvtColor(src, src, CV_BGR2GRAY);

        cv::Mat angles;

        cv::Mat edges_char = find_edges(src, angles);

        AccumPoint max_accum = AccumPoint(-1, Cell(-1, -1));
        std::vector <cv::Mat> accum = hough_rect(src, edges_char, angles,  max_accum);

//        cv::Mat resized_accum;
//        int scaled_angle = max_accum.angle / hough_scale_angle;
//        resize(accum[scaled_angle], resized_accum, cv::Size(accum[scaled_angle].cols * 5,
//                                                 accum[scaled_angle].rows * 5));

        std::vector<Cell> points;
        points.push_back(max_accum.cell);
        draw_rect(src, points, 100, 300, 5, max_accum.angle);

        imshow("src", src);
//        imshow("max_accum", resized_accum);
//        imshow("edges", edges_char);
//        imshow("filtered", filtered);
        return src;
    }
};

void HoughRect(InputArray src_image, int rect_height, int rect_width, int accum_scale, int angle_scale) {
    HoughLaneRecognizer hr(rect_height, rect_width, accum_scale, angle_scale);
//    cv::Mat src = cv::cvarrToMat(src_image);
    hr.recognize(src_image);
}

int main(int argc, char* argv[]) {
    cv::Mat src = cv::imread(argv[1], 1);
    HoughRect(src, 50, 150, 5, 5);
    waitKey(0);
    return 0;
}