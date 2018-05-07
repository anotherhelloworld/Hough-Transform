#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

cv::Mat equalize_channel(cv::Mat src, double unused) {
    cv::Mat res = src.clone();

    std::vector<double> hist(256);

    int all_count = src.rows * src.cols;

    int left_bound = 1000;
    int right_bound = 0;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int val = src.at<uchar>(i, j);
            hist[val]+=1;
            right_bound = std::max(right_bound, val);
            left_bound = std::min(left_bound, val);
        }
    }

    int need_count = all_count * unused;
    int left_count = 0;
    int right_count = 0;
    int k = 0;
    while(left_count < need_count) {
        left_count += hist[k];
        k++;
    }
    left_bound = std::max(left_bound, k);
    k = hist.size() - 1;
    while(right_count < need_count) {
        right_count += hist[k];
        k--;
    }
    right_bound = std::min(right_bound, k);
    all_count -= left_count + right_count;

    if (right_bound == left_bound) {
        return res;
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int oldval = src.at<uchar>(i, j);
            int newval = oldval;

            newval = 255.0 * (oldval - left_bound) / (right_bound - left_bound);
            newval = std::min(newval, 255);
            newval = std::max(newval, 0);
            res.at<uchar>(i, j) = newval;
        }
    }

    return res;
}

cv::Mat equalize_hist(const cv::Mat& frame) {
    std::vector<cv::Mat> channels, channels_eq;
    split(frame, channels);

    channels_eq = channels;

    int channels_count = 3;

    for (int i = 0; i < channels_count; i++) {
        channels_eq[i] = equalize_channel(channels[i], 0.01);
    }

    cv::Mat equalized;
    merge(channels_eq, equalized);
    return equalized;
}

int main(int argc, char* argv[]) {
    cv::Mat raw;
    raw = cv::imread(argv[1]);
    auto img = equalize_hist(raw);
    imwrite(string(argv[2]) + argv[3] + string(".png"), img);

    return 0;
}