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
    int height;
    int width;
    AccumPoint(
        int value = 0,
        int angle = 0,
        Cell cell = Cell(-1, -1),
        int height = 0,
        int width = 0
        )
        : value(value)
        , angle(angle)
        , cell(cell)
        , height(height)
        , width(width)
    {};
    bool operator < (const AccumPoint& r) const { return value > r.value; }
};

struct RectSize
{
    int height;
    int width;
    AccumPoint accum;
    RectSize(
        int height,
        int width,
        AccumPoint accum
        )
        : height(height)
        , width(width)
        , accum(accum)
    {};
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

class HoughRectsAccumInvoker : public ParallelLoopBody
{
private:
    const cv::Mat& image;
    const cv::Mat& edges;
    const cv::Mat& angles;
    AccumPoint& globalMaxAccum;
    std::vector<Accum>& accums;
    int width;
    int height;
    int accumScale;
    int angleStep;
    int sideSize;
    int aspectRatio;

public:
    HoughRectsAccumInvoker(
        const cv::Mat& image,
        const cv::Mat& edges,
        const cv::Mat& angles,
        AccumPoint& globalMaxAccum,
        std::vector<Accum>& accums,
        int width,
        int height,
        int accumScale,
        int angleStep,
        int sideSize,
        int aspectRatio
        )
        : image(image)
        , edges(edges)
        , angles(angles)
        , globalMaxAccum(globalMaxAccum)
        , accums(accums)
        , width(width)
        , height(height)
        , accumScale(accumScale)
        , angleStep(angleStep)
        , sideSize(sideSize)
        , aspectRatio(aspectRatio)
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
            int angleStep,
            int angle,
            int height,
            int width) const
    {
        if (angle < 0) angle += 180; else if (angle > 180) angle -= 180;

        int angleScaled = angle / angleStep;
        double _norm = sqrt((finish.row - start.row) * (finish.row - start.row) +
                            (finish.col - start.col) * (finish.col - start.col));
        for (double i = 0; i <= 1; i += scale / (double)(_norm)) {
            Cell p = Cell(floor(start.row + (finish.row - start.row) * i),
                          floor(start.col + (finish.col - start.col) * i));
            if (p.row < image.rows && p.row > 0 && p.col < image.cols && p.col > 0) {
                accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale)++;
                accum[angleScaled].counter++;
                if (accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale) > globalMaxAccum.value) {
                    globalMaxAccum.value = accum[angleScaled].accum.at<int>(p.row / scale, p.col / scale);
                    globalMaxAccum.cell.row = p.row / scale;
                    globalMaxAccum.cell.col = p.col / scale;
                    globalMaxAccum.angle = angle;//?
                    globalMaxAccum.height = height;
                    globalMaxAccum.width = width;
                }
            }
        }
    }

    void run_rectangle(
            const cv::Mat& image,
            std::vector<Accum>& accum,
            int accumScale,
            int angleStep,
            int angle,
            int rad,
            double k,
            int row,
            int col,
            int height,
            int width) const
    {
        for (int r = rad - 2; r <=  rad + 2; r++) {
            int bound = 15;

            int start = angle - bound;
            int finish = angle + bound;
            for (int angle = start; angle <= finish; angle += angleStep) {
                int cur_height = r;
                int cur_width = k * r;

                Cell ptl = Cell(row - cur_height, col - cur_width);
                Cell ptr = Cell(row - cur_height, col + cur_width);
                Cell pbr = Cell(row + cur_height, col + cur_width);
                Cell pbl = Cell(row + cur_height, col - cur_width);


                Cell rect_coords[4] = { ptl, ptr, pbr, pbl };
                std::vector <Cell> rotate_coords = rotate_rect(rect_coords, Cell(row, col), angle);

                run_along_line(image, accum, rotate_coords[0], rotate_coords[1], accumScale, angleStep, angle, height, width);
                run_along_line(image, accum, rotate_coords[1], rotate_coords[2], accumScale, angleStep, angle, height, width);
                run_along_line(image, accum, rotate_coords[2], rotate_coords[3], accumScale, angleStep, angle, height, width);
                run_along_line(image, accum, rotate_coords[3], rotate_coords[0], accumScale, angleStep, angle, height, width);
            }
        }
    }

    void operator() (const Range& boundaries) const
    {
        int start = boundaries.start;
        int end = boundaries.end;
        double k = ((double)width / (double)height);
        for (int row = start; row < end; row++) {
            for (int col = 0; col < edges.cols; col++) {
                if (edges.at<uchar>(row, col) == 0) {
                    continue;
                }
                run_rectangle(image, accums, accumScale, angleStep,
                              angles.at<int>(row, col), height, k, row, col, height, width);
                run_rectangle(image, accums, accumScale, angleStep,
                              angles.at<int>(row, col) - 90, height, k, row, col, height, width);
            }
        }
    }
};

class HoughRectRecognizer
{
public:

    int sideSize;
    int minAspectRatio;
    int maxAspectRatio;
    int angleStep;
    int accumScale;

    HoughRectRecognizer(
        int sideSize,
        int minAspectRatio,
        int maxAspectRatio,
        int angleStep,
        int accumScale
        )
        : sideSize(sideSize)
        , minAspectRatio(minAspectRatio)
        , maxAspectRatio(maxAspectRatio)
        , angleStep(angleStep)
        , accumScale(accumScale)
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

    void find_local_max(const std::vector<Accum>& accums, int windowSize, std::vector<AccumPoint>& localMax)
    {
        for (int i = 0; i < accums.size() * 5 / 10; i++) {
            assert(windowSize <= accums[i].accum.rows);
            assert(windowSize <= accums[i].accum.cols);

            for (int row = 0; row < accums[i].accum.rows - windowSize; row += windowSize) {
                for (int col = windowSize; col <= accums[i].accum.cols - windowSize; col += windowSize) {


                    cv::Rect roi(
                        col,
                        row,
                        ((row + windowSize) > accums[i].accum.rows ? accums[i].accum.rows - row : windowSize),
                        ((col + windowSize) > accums[i].accum.cols ? accums[i].accum.cols - col : windowSize));

                    cv::Mat roiMat = accums[i].accum(roi);

                    double _min, _max;
                    cv::Point minLoc, maxLoc;
                    minMaxLoc(roiMat, &_min, &_max, &minLoc, &maxLoc);
                    localMax.push_back(AccumPoint((int)_max, accums[i].angle_scaled, Cell(maxLoc.x + col, maxLoc.y + row)));
                }
            }
        }
    }


    void houghRectParallel(
            const cv::Mat& image,
            const cv::Mat& edges,
            const cv::Mat& angles,
            AccumPoint& globalMaxAccum,
            std::vector<AccumPoint>& localMax)
    {
        int max_angle = (180 + 1) / angleStep;
        std::vector <Accum> accums(max_angle + 1);

        for (int i = 0; i <= max_angle; i ++) {
            accums[i].accum = cv::Mat::zeros(image.rows / accumScale,
                                             image.cols / accumScale, CV_32SC1);
            accums[i].angle_scaled = i;
        }

        int numThreads = std::max(1, getNumThreads());

        // for (int ratio = minAspectRatio; ratio <= maxAspectRatio; ratio++) {
        //     int height = sideSize;
        //     int width = sideSize*ratio;
        //     parallel_for_(Range(0, edges.rows),
        //         HoughRectsAccumInvoker(
        //             image, edges, angles, globalMaxAccum,
        //             accums, width, height, accumScale,
        //             angleStep, sideSize, ratio),
        //         numThreads);
        // }

        int height = 100;
        int width = 300;
        parallel_for_(Range(0, edges.rows),
            HoughRectsAccumInvoker(
                image, edges, angles, globalMaxAccum,
                accums, width, height, accumScale,
                angleStep, sideSize, 3),
            numThreads);

        std::sort(accums.begin(), accums.end());
        find_local_max(accums, sideSize / accumScale, localMax);
        std::sort(localMax.begin(), localMax.end());
        int a = a + 1;
    }

    void recognize(cv::Mat& src, std::vector<cv::Vec6f>& rects, cv::Mat& edges, cv::Mat& angles)
    {
        AccumPoint globalMaxAccum;
        std::vector<AccumPoint> localMax;
        houghRectParallel(src, edges, angles, globalMaxAccum, localMax);

        // for (int i = 0; i < 2 && i < localMax.size(); i++) {
        //     cv::Vec6f rRect(
        //         localMax[0].cell.col * accumScale + accumScale / 2,
        //         localMax[0].cell.row * accumScale + accumScale / 2,
        //         sideSize * minAspectRatio, sideSize, localMax[0].angle, localMax[0].value);
        //     rects.push_back(rRect);
        // }

        cv::Vec6f rRect(
            globalMaxAccum.cell.col * accumScale + accumScale / 2,
            globalMaxAccum.cell.row * accumScale + accumScale / 2,
            sideSize * minAspectRatio, sideSize, globalMaxAccum.angle, globalMaxAccum.value);
        rects.push_back(rRect);
    }
};

void HoughRects(cv::InputArray image, cv::OutputArray rects, int sideSize, int minAspectRatio,
                int maxAspectRatio, int accumScale, int angleStep)
{

    HoughRectRecognizer hr(sideSize, minAspectRatio, minAspectRatio, accumScale, angleStep);
    std::vector<cv::Vec6f> _rects;

    cv::Mat src;
    cv::Mat angles;
    image.copyTo(src);

    bool empty = false;
    cv::Mat edges = hr.find_edges(src, angles, empty);

    if (empty) {
        return;
    }

    hr.recognize(src, _rects, edges, angles);

    int rows = (int)_rects.size();
    cv::Mat resRects(rows, 6, CV_32FC1);
    for (int i = 0; i < _rects.size(); i++) {
        for (int j = 0; j < 6; j++) {
            resRects.at<float>(i, j) = _rects[i][j];
        }
    }

    if (rows > 0) {
        rects.create(rows, 6, CV_32FC1);
        rects.assign(resRects);
    }
    return;
}

// CV_IMPL CvSeq*
// cvHoughRect(cv::InputArray src_image, int rect_height,
//             int rect_width, int accum_scale, int angle_scale, int min_angle, int max_angle)
// {
//     HoughRectRecognizer hr(rect_height, rect_width, rect_height / 2, rect_width / 2, accum_scale, angle_scale);
//     return NULL;
// }
