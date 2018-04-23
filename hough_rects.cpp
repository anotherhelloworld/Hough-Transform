/****************************************************************************************\
*                                     Rect Detection                                     *
\****************************************************************************************/
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

int get_distance(Cell& c1, Cell& c2)
{
    return (c2.row - c1.row) * (c2.row - c1.row) + (c2.col - c1.col) * (c2.col - c1.col);
}

Cell get_real_cell(Cell& c, int hough_scale)
{
    return Cell(c.row * hough_scale + hough_scale / 2, c.col * hough_scale + hough_scale / 2);
}

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

class FindEdgesInvoker : public ParallelLoopBody
{
private:
    const cv::Mat& mat;
    cv::Mat& angles;
    bool& empty;
public:
    FindEdgesInvoker(
        const cv::Mat& mat,
        cv::Mat& angles,
        bool& empty
        )
        : mat(mat)
        , angles(angles)
        , empty(empty)
        {};

    ~FindEdgesInvoker();

    // cv::Mat int_to_char(const cv::Mat& mat)
    // {
    //     double _min = 0, _max = 0;
    //     cv::minMaxLoc(mat, &_min, &_max);
    //     return normalize_mat(mat, _max);
    // }

    // cv::Mat normalize_mat(const cv::Mat& mat, int _max)
    // {
    //     cv::Mat res = cv::Mat::zeros(mat.rows, mat.cols, CV_8UC1);
    //     if (_max <= 0) {
    //         return res;
    //     }
    //     for (int i = 0; i < mat.rows; i++) {
    //         for (int j = 0; j < mat.cols; j++) {
    //             res.at<uchar>(i, j) = (255 * mat.at<int>(i, j)) / (int)_max;
    //         }
    //     }
    //     return res;
    // }

    void operator() (const Range& boundaries) const
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

        for (size_t i = 0; i < filtered.size(); i++) {
            edges.at<int>(filtered[i].cell.row, filtered[i].cell.col) = filtered[i].value;
        }

        // cv::Mat edges_char = int_to_char(edges);

        // return edges_char;
    }
};

class HoughRectsAccumInvoker : public ParallelLoopBody
{
private:
    const cv::Mat& image;
    const std::vector<AccumPoint>& edges;
    const cv::Mat& angles;
    std::vector<AccumPoint>& max_accums;
    std::vector<Accum>& accums;
    int hough_width;
    int hough_height;
    int hough_scale;
    int hough_scale_angle;

public:
    HoughRectsAccumInvoker(
        const cv::Mat& image,
        const std::vector<AccumPoint>& edges,
        const cv::Mat& angles,
        std::vector<AccumPoint>& max_accums,
        std::vector<Accum>& accums,
        int hough_width,
        int hough_height,
        int hough_scale,
        int hough_scale_angle
        )
        : image(image)
        , edges(edges)
        , angles(angles)
        , max_accums(max_accums)
        , accums(accums)
        , hough_width(hough_width)
        , hough_height(hough_height)
        , hough_scale(hough_scale)
        , hough_scale_angle(hough_scale_angle)
    {}

    ~HoughRectsAccumInvoker() { }

    std::vector <Cell> rotate_rect(Cell rect_coords[4], Cell centr, int angle) const
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
            int angle) const
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
                if (accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale) > max_accums[0].value) {
                    max_accums[0].value = accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale);
                    max_accums[0].cell.row = p.row / scale;
                    max_accums[0].cell.col = p.col / scale;
                    max_accums[0].angle = angle;//?
                }
            }
        }
    }

    void run_rectangle(
            const cv::Mat& image,
            std::vector<Accum>& accum,
            int scale,
            int scale_angle,
            int angle,
            int rad,
            double k,
            int row,
            int col) const
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
                std::vector <Cell> rotate_coords = rotate_rect(rect_coords, Cell(row, col), angle);

                run_along_line(image, accum, rotate_coords[0], rotate_coords[1], scale, scale_angle, angle);
                run_along_line(image, accum, rotate_coords[1], rotate_coords[2], scale, scale_angle, angle);
                run_along_line(image, accum, rotate_coords[2], rotate_coords[3], scale, scale_angle, angle);
                run_along_line(image, accum, rotate_coords[3], rotate_coords[0], scale, scale_angle, angle);
            }
        }
    }

    void operator() (const Range& boundaries) const
    {
        int start = boundaries.start;
        int end = boundaries.end;
        double k = ((double)hough_width / (double)hough_height);
        for (int i = start; i < end; i++) {
            if (edges[i].value == 0) {
                continue;
            }

            AccumPoint tmp_max_accum = AccumPoint(-1, 0, Cell(-1, -1));

            run_rectangle(image, accums, hough_scale, hough_scale_angle,
                          angles.at<int>(edges[i].cell.row, edges[i].cell.col), hough_height, k, edges[i].cell.row, edges[i].cell.col);
            run_rectangle(image, accums, hough_scale, hough_scale_angle,
                          angles.at<int>(edges[i].cell.row, edges[i].cell.col) - 90, hough_height, k, edges[i].cell.row, edges[i].cell.col);
        }
        // for (int row = start; row < end; row++) {
        //     for (int col = 0; col < edges_char.cols; col++) {
        //         if (edges_char.at<uchar>(row, col) == 0) {
        //             continue;
        //         }

        //         AccumPoint tmp_max_accum = AccumPoint(-1, 0, Cell(-1, -1));

        //         run_rectangle(image, accums, hough_scale, hough_scale_angle,
        //                       angles.at<int>(row, col), hough_height, k, row, col);
        //         run_rectangle(image, accums, hough_scale, hough_scale_angle,
        //                       angles.at<int>(row, col) - 90, hough_height, k, row, col);
        //     }
        // }
    }
};

class HoughRectRecognizer
{
public:

    int src_height;
    int src_width;
    int hough_height;
    int hough_width;
    int hough_scale;
    int hough_scale_angle;
    int rects_num;

    HoughRectRecognizer(
        int src_height,
        int src_width,
        int hough_height,
        int hough_width,
        int hough_scale,
        int hough_scale_angle,
        int rects_num
        )
        : src_height(src_height)
        , src_width(src_width)
        , hough_height(hough_height)
        , hough_width(hough_width)
        , hough_scale(hough_scale)
        , hough_scale_angle(hough_scale_angle)
        , rects_num(rects_num)
    {};

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

    std::vector<AccumPoint> find_edges(const cv::Mat& mat, cv::Mat& angles, bool& empty)
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

        return filtered;
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
                    max_accum.angle = angle;//?
                }
            }
        }
    }

    void run_rectangle(
            const cv::Mat& image,
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
                std::vector <Cell> rotate_coords = rotate_rect(rect_coords, Cell(row, col), angle);

                run_along_line(image, accum, rotate_coords[0], rotate_coords[1], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[1], rotate_coords[2], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[2], rotate_coords[3], scale, scale_angle, max_accum, angle);
                run_along_line(image, accum, rotate_coords[3], rotate_coords[0], scale, scale_angle, max_accum, angle);
            }
        }
    }

    void hough_rect_parallel(
            const cv::Mat& image,
            const std::vector<AccumPoint>& edges,
            const cv::Mat& angles,
            std::vector<AccumPoint>& max_accums)
    {
        int global_max_counter = 0;
        int max_angle = (180 + 1) / hough_scale_angle;
        std::vector <Accum> accums(max_angle + 1);

        max_accums[0] = AccumPoint(-1, 0, Cell(-1, -1));

        for (int i = 0; i <= max_angle; i ++) {
            accums[i].accum = cv::Mat::zeros(image.rows / hough_scale,
                                             image.cols / hough_scale, CV_32SC1);
            accums[i].angle_scaled = i;
        }

        double k = ((double)hough_width / (double)hough_height);
        int rect_diag = hough_width * hough_width + hough_height * hough_height;

        int numThreads = std::max(1, getNumThreads());
        parallel_for_(Range(0, edges.size()),
                  HoughRectsAccumInvoker(
                    image, edges, angles,
                    max_accums, accums, hough_width,
                    hough_height, hough_scale, hough_scale_angle),
                  numThreads);
    }

    void hough_rect(
            const cv::Mat& image,
            const cv::Mat& edges_char,
            const cv::Mat& angles,
            std::vector<AccumPoint>& max_accums)
    {

        int global_max_counter = 0;
        int max_angle = (180 + 1) / hough_scale_angle;
        std::vector <Accum> accums(max_angle + 1);

        for (int i = 0; i <= max_angle; i ++) {
            accums[i].accum = cv::Mat::zeros(image.rows / hough_scale,
                                             image.cols / hough_scale, CV_32SC1);
            accums[i].angle_scaled = i;
        }

        double k = ((double)hough_width / (double)hough_height);
        int rect_diag = hough_width * hough_width + hough_height * hough_height;
        for (int row = 0; row < edges_char.rows; row++) {
            for (int col = 0; col < edges_char.cols; col++) {
                if (edges_char.at<uchar>(row, col) == 0) {
                    continue;
                }

                AccumPoint tmp_max_accum = AccumPoint(-1, 0, Cell(-1, -1));

                run_rectangle(image, accums, hough_scale, hough_scale_angle,
                              tmp_max_accum, angles.at<int>(row, col), hough_height, k, row, col);
                run_rectangle(image, accums, hough_scale, hough_scale_angle,
                              tmp_max_accum, angles.at<int>(row, col) - 90, hough_height, k, row, col);

                if (rects_num == 1) {
                    if (max_accums[0].value < tmp_max_accum.value) {
                        max_accums[0] = tmp_max_accum;
                    }
                } else {
                    bool updated = false;
                    for (int i = 0; i < max_accums.size() && global_max_counter > 0 && !updated; i++) {
                        Cell c1 = get_real_cell(max_accums[i].cell, hough_scale);
                        Cell c2 = get_real_cell(tmp_max_accum.cell, hough_scale);
                        int dist = get_distance(c1, c2);
                        if (dist * 4 < rect_diag && max_accums[i].value < tmp_max_accum.value) {
                            max_accums[i] = tmp_max_accum;
                            updated = true;
                        }
                    }

                    if (!updated && global_max_counter < max_accums.size()) {
                        max_accums[global_max_counter] = tmp_max_accum;
                        global_max_counter++;
                    }
                    sort(max_accums.begin(), max_accums.end());
                }
            }
        }
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

    void recognize(cv::Mat& src, std::vector<cv::Vec6f>& rects, std::vector<AccumPoint>& edges, cv::Mat& angles)
    {
        std::vector<AccumPoint> max_accums(rects_num);

        // hough_rect(src, edges, angles, max_accums);
        hough_rect_parallel(src, edges, angles, max_accums);

        if (max_accums.size() == 0) {
            return;
        }

        if (rects_num == 1) {
            cv::Vec6f rRect(
                max_accums[0].cell.col * this->hough_scale + this->hough_scale / 2,
                max_accums[0].cell.row * this->hough_scale + this->hough_scale / 2,
                this->src_width, this->src_height, max_accums[0].angle, max_accums[0].value);
            rects.push_back(rRect);
        }
    }
};

void HoughRects(cv::InputArray src_image, cv::OutputArray _output, int rects_num,
                int rect_height, int rect_width, int accum_scale, int angle_scale)
{

    if ((rect_height == -1 || rect_width == -1) && rects_num == 1) {
        // std::vector<cv::Vec6f> res_rects(rects_num);
        // res_rects[0][5] = -1;
        // for (int rect_height = src_image.rows() / 10; rect_height < src_image.rows(); rect_height += src_image.rows() / 10) {
        //     for (int rect_width = src_image.cols() / 10; rect_width < src_image.cols(); rect_width += src_image.cols() / 10) {
        //         HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale, rects_num);
        //         std::vector<cv::Vec6f> rects;
        //         hr.recognize(src_image, rects);
        //         if (rects[0][5] > res_rects[0][5]) {
        //             res_rects[0] = rects[0];
        //         }
        //     }
        // }

        // int rows = (int)res_rects.size();
        // cv::Mat _rects(rows, 6, CV_32FC1);
        // for (int i = 0; i < res_rects.size(); i++) {
        //     for (int j = 0; j < 6; j++) {
        //         _rects.at<float>(i, j) = res_rects[i][j];
        //     }
        // }

        // if (rows > 0) {
        //     _output.create(rows, 6, CV_32FC1);
        //     _output.assign(_rects);
        // }
        // return;
    } else if (rects_num == 1) {
        HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale, rects_num);
        std::vector<cv::Vec6f> rects;

        cv::Mat src;
        cv::Mat angles;
        src_image.copyTo(src);

        bool empty = false;
        std::vector<AccumPoint> edges = hr.find_edges(src, angles, empty);

        if (empty) {
            return;
        }

        hr.recognize(src, rects, edges, angles);

        int rows = (int)rects.size();
        cv::Mat _rects(rows, 6, CV_32FC1);
        for (int i = 0; i < rects.size(); i++) {
            for (int j = 0; j < 6; j++) {
                _rects.at<float>(i, j) = rects[i][j];
            }
        }

        if (rows > 0) {
            _output.create(rows, 6, CV_32FC1);
            _output.assign(_rects);
        }
        return;
    }
}

// CV_IMPL CvSeq*
// cvHoughRect(cv::InputArray src_image, int rect_height,
//             int rect_width, int accum_scale, int angle_scale, int min_angle, int max_angle)
// {
//     HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale);
//     return NULL;
// }
