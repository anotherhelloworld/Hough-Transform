/****************************************************************************************\
*                                     Rect Detection                                     *
\****************************************************************************************/

class HoughRectRecognizer
{
public:

    int src_height;
    int src_width;
    int hough_height;
    int hough_width;
    int hough_scale;
    int hough_scale_angle;

    HoughRectRecognizer(
            int src_height,
            int src_width,
            int hough_height,
            int hough_width,
            int hough_scale,
            int hough_scale_angle
            )
            : src_height(src_height)
            , src_width(src_width)
            , hough_height(hough_height)
            , hough_width(hough_width)
            , hough_scale(hough_scale)
            , hough_scale_angle(hough_scale_angle)
    {};

    struct Cell
    {
        int row;
        int col;
        Cell(
            int row = -1,
            int col = -1
            )
            : row(row)
            , col(col)
        {};
    };

    struct AccumPoint
    {
        int value;
        int angle;
        //int angle_scaled = 0;
        Cell cell;
        AccumPoint(
            int value = 0,
            int angle = 0,
            Cell cell = Cell(-1, -1)
            )
            : value(value)
            , angle(angle)
            , cell(cell)
        {};
        bool operator < (const AccumPoint& r) const { return value > r.value; }
    };

    struct Accum
    {
        int counter;
        int angle_scaled;
        cv::Mat accum;
        std::vector<AccumPoint> local_max;
        Accum(
            int counter = 0
            )
            : counter(counter)
        {};
        bool operator < (const Accum& r) const { return counter > r.counter; }
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

    // std::set <AccumPoint> find_local_max(const std::vector<Accum>& accums, int windowSize, AccumPoint& max_accum, int scale_angle)
    // {
    //     std::set <AccumPoint> res;
    //     for (int i = 0; i < accums.size() * 5 / 10; i++) {
    //         assert(windowSize <= accums[i].accum.rows);
    //         assert(windowSize <= accums[i].accum.cols);

    //         for (int row = 0; row < accums[i].accum.rows - windowSize; row += windowSize) {
    //             for (int col = windowSize; col <= accums[i].accum.cols - windowSize; col += windowSize) {

    //                 // std::cout << row << " " << col << std::endl;

    //                 cv::Rect roi(
    //                     col,
    //                     row,
    //                     ((row + windowSize) > accums[i].accum.rows ? accums[i].accum.rows - row : windowSize),
    //                     ((col + windowSize) > accums[i].accum.cols ? accums[i].accum.cols - col : windowSize));

    //                 cv::Mat roiMat = accums[i].accum(roi);

    //                 double _min, _max;
    //                 cv::Point minLoc, maxLoc;
    //                 minMaxLoc(roiMat, &_min, &_max, &minLoc, &maxLoc);

    //                 if ((max_accum.value / 2 < roiMat.at<int>(maxLoc.x, maxLoc.y))
    //                     && ((max_accum.cell.row - maxLoc.y > windowSize)
    //                         || (max_accum.cell.col - maxLoc.x > windowSize)
    //                         || (max_accum.angle - accums[i].angle_scaled * scale_angle > 15))) {
    //                     res.emplace(0, accums[i].angle_scaled * scale_angle, Cell(maxLoc.y, maxLoc.x));
    //                 }
    //             }
    //         }
    //     }
    //     return res;
    // }

    cv::Mat find_edges(const cv::Mat& mat, cv::Mat& angles, bool& empty)
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
                filtering.push_back(AccumPoint(laplas, 0, Cell(i, j)));
            }
        }

        sort(filtering.begin(), filtering.end());

        std::vector<AccumPoint> filtered(filtering.begin(), filtering.begin() + filtering.size() * 5 / 10);

        if (filtered.size() == 0) {
            empty = true;
        }

        cv::Mat edges = cv::Mat::zeros(delta_i.rows, delta_i.cols, CV_32SC1);

        // for (auto &point: filtered) {
        //     edges.at<int>(point.cell.row, point.cell.col) = point.value;
        // }

        for (size_t i = 0; i < filtered.size(); i++) {
            edges.at<int>(filtered[i].cell.row, filtered[i].cell.col) = filtered[i].value;
        }

        cv::Mat edges_char = int_to_char(edges);

        return edges_char;
    }

    std::vector <Cell> rotate_rect(Cell rect_coords[4], Cell centr, int angle)
    {
        std::vector<Cell> res(4);
        for (int i = 0; i < 4; i++) {
            rect_coords[i].row -= centr.row;
            rect_coords[i].col -= centr.col;
        }

        for (size_t i = 0; i < res.size(); i++) {
            double theta = (angle * CV_PI) / 180.0;
            res[i].row = rect_coords[i].col * sin(theta) + rect_coords[i].row * cos(theta);
            res[i].col = rect_coords[i].col * cos(theta) - rect_coords[i].row * sin(theta);
        }

        for (size_t i = 0; i < res.size(); i++) {
            res[i].row += centr.row;
            res[i].col += centr.col;
        }

        return res;
    }

    void run_along_line(
            const cv::Mat& image,
            std::vector <Accum>& accum,
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
                accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale)++;
                accum[angleScaled].counter++;
                if (accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale) > max_accum.value) {
                    max_accum.value = accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale);
                    max_accum.cell.row = p.row / scale;
                    max_accum.cell.col = p.col / scale;
                    max_accum.angle = angle;///?
                    //max_accum.angle_scaled = angleScaled;
                }
            }
        }
    }

    void run_rectangle(
            const cv::Mat& image,
//            std::vector <cv::Mat>& accum,
//            std::vector <int>& accum_counter,
            std::vector<Accum>& accum,
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


                Cell rect_coords[4] = { ptl, ptr, pbr, pbl };
                // std::array<Cell, 4> rect_coords = { ptl, ptr, pbr, pbl };
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
            const cv::Mat& angles,
            AccumPoint& max_accum)
    {
        int max_angle = (180 + 1) / hough_scale_angle;
//        std::vector <cv::Mat> angles_accum(max_angle + 1);
//        std::vector <int> accum_counter(max_angle + 1);
        std::vector <Accum> accums(max_angle + 1);

        for (int i = 0; i <= max_angle; i ++) {
            accums[i].accum = cv::Mat::zeros(image.rows / hough_scale,
                                             image.cols / hough_scale, CV_32SC1);
            accums[i].angle_scaled = i;
        }

        double k = ((double)hough_width / (double)hough_height);
        for (int row = 0; row < edges_char.rows; row++) {
            for (int col = 0; col < edges_char.cols; col++) {
                if (edges_char.at<uchar>(row, col) == 0) {
                    continue;
                }
//                run_rectangle(image, angles_accum, accum_counter, hough_scale, hough_scale_angle,
//                              max_accum, angles.at<int>(row, col), hough_height, k, row, col);
//                run_rectangle(image, angles_accum, accum_counter, hough_scale, hough_scale_angle,
//                              max_accum, angles.at<int>(row, col) - 90, hough_height, k, row, col);
                run_rectangle(image, accums, hough_scale, hough_scale_angle,
                              max_accum, angles.at<int>(row, col), hough_height, k, row, col);
                run_rectangle(image, accums, hough_scale, hough_scale_angle,
                              max_accum, angles.at<int>(row, col) - 90, hough_height, k, row, col);

            }
        }

        int scaled_angle = max_accum.angle / hough_scale_angle;

        if ((size_t)scaled_angle > accums.size()) {
            std::vector <cv::Mat> angles_accum_char;
            return angles_accum_char;
        }
//        cv::Mat mat_max_accum = accum[scaled_angle].accum;
//
//        int sum = 0;
//
//        for (int i = 0; i < mat_max_accum.rows; i++) {
//            for (int j = 0; j < mat_max_accum.cols; j++) {
//                sum += mat_max_accum.at<int>(i, j);
//            }
//        }
//
//        int average = sum / (mat_max_accum.rows + mat_max_accum.cols - 2);
        sort(accums.begin(), accums.end());

        // find_local_max(accums, 5, max_accum, hough_scale_angle);

        std::vector <cv::Mat> angles_accum_char(max_angle + 1);

//        if (max_accum.value * 3 < average) {
//            max_accum.value = -1;
//            return angles_accum_char;
//        }

        for (int angle = 0; angle <= max_angle; angle++) {
            angles_accum_char[angle] = int_to_char_global_max(accums[angle].accum, max_accum.value);
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
        for (size_t i = 0; i < rect_coords.size(); i++) {
            int col = rect_coords[i].col * scale + scale / 2;
            int row = rect_coords[i].row * scale + scale / 2;
            cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(col, row), cv::Size2f(width, height), angle);
            cv::Point2f vertices[4];
            rRect.points(vertices);
            for (int i = 0; i < 4; i++) {
                line(mat, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 3);
            }
        }
    }

    void recognize(cv::InputArray _src, std::vector<cv::Vec6d>& rects)
    {
        cv::Mat src;
        _src.copyTo(src);
        cv::Mat image;

        cv::Mat angles;

        bool empty = false;
        cv::Mat edges_char = find_edges(src, angles, empty);

        // CvMemStorage* seqMem = cvCreateMemStorage(0);

        // CvSeq* seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(cv::RotatedRect), seqMem);

        if (empty) {
            return;
        }

        AccumPoint max_accum = AccumPoint(-1, 0, Cell(-1, -1));

        std::vector <cv::Mat> accum = hough_rect(src, edges_char, angles,  max_accum);

        if (max_accum.value == -1) {
            return;
        }

        std::vector<Cell> points;
        points.push_back(max_accum.cell);

        points[0].col = points[0].col * this->hough_scale + this->hough_scale / 2;
        points[0].row = points[0].row * this->hough_scale + this->hough_scale / 2;

        // cv::RotatedRect rRect(cv::Point2f(points[0].col, points[0].row),
        //                             cv::Size2f(this->src_width, this->src_height),
        //                             max_accum.angle);

        cv::Vec6f rRect(points[0].col, points[0].row,
                        this->src_width, this->src_height,
                        max_accum.angle, 0);

        // cvSeqPush(seq, rRect);
       rects.push_back(rRect);

        cv::Mat C3;
        int scaled_angle = max_accum.angle / hough_scale_angle;
        resize(accum[scaled_angle], C3, cv::Size(accum[scaled_angle].cols * 5,
                                                 accum[scaled_angle].rows * 5));
        // imshow("accum", C3);

        return;
    }
};

void HoughRects(cv::InputArray src_image, cv::OutputArray _output, int rect_height,
            int rect_width, int accum_scale, int angle_scale)
{
    HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale);
    std::vector<cv::Vec6d> rects;
    hr.recognize(src_image, rects);
    if (rects.size() > 0) {
        _output.create(rects.size(), 6, CV_32FC1);
        Mat(1, rects.size(), CV_32SC1, &rects[0]).copyTo(_output.getMat());
    }
    return;
    // Mat(1, rects.size(), CV_32SC6, &rects[0]);
    // tmp->copyTo(output);
    // _output.create(rects.size(), 5, CV_32FC1);
    // cv::Mat _rects(rects.size(), 5, CV_32S);
    // for (size_t i = 0; i < rects.size(); i++) {
    //         _rects.at<int>(i, 0) = rects[i].center.x;
    //         _rects.at<int>(i, 1) = rects[i].center.y;
    //         _rects.at<int>(i, 2) = rects[i].angle;
    //         _rects.at<int>(i, 3) = rects[i].size.width;
    //         _rects.at<int>(i, 4) = rects[i].size.height;
    // }
    // _rects.copyTo(_output);
}

// CV_IMPL CvSeq*
// cvHoughRect(cv::InputArray src_image, int rect_height,
//             int rect_width, int accum_scale, int angle_scale, int min_angle, int max_angle)
// {
//     HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale);
//     return NULL;
// }