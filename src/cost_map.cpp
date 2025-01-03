#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "MPPI/cost_map.hpp"
#include "MPPI/obstacles.hpp"

CostMap::CostMap(const cv::Mat& map, const double robot_radius, const float cost_decay_rate)
    : robot_radius_(robot_radius), cost_decay_rate_(cost_decay_rate) {
    if (!update_cost_map(map)) {
        throw std::runtime_error("Failed to initialize the cost map.");
    }
};

bool CostMap::update_cost_map(const cv::Mat & map) {

    auto update_obstacle_mask = [this] (const cv::Mat& map) -> void {
        // Create a mask where obstacles are black (0) and the background is white (255)
        cv::Mat gray_map;
        cv::cvtColor(map, gray_map, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_map, obstacle_mask_, 127, 255, cv::THRESH_BINARY);
        obstacle_mask_.convertTo(obstacle_mask_, CV_8UC1); // Align to 8-bit single channel just in case
    };

    auto update_distance_image = [this] () -> void{
        double min_val, max_val;
        cv::minMaxLoc(distance_map_, &min_val, &max_val);
        distance_map_.convertTo(distance_img_, CV_8U, 255.0 / max_val);
    };

    // Validate the input map
    if (map.empty()) {
        std::cerr << "The input map is empty." << std::endl;
        return false;
    }
    if (map.channels() != 3) {
        std::cerr << "The input map must be a 3-channel image." << std::endl;
        return false;
    }

    update_obstacle_mask(map);
    cv::distanceTransform(obstacle_mask_, distance_map_, cv::DIST_L2, 3);

    update_distance_image();

    cost_map_ = create_exp_cost_map();
    cost_color_ = create_cost_color_map();
    return true;
}

float CostMap::get_cost(const int x, const int y) const {
    // validate the input
    if (x < 0 || x >= cost_map_.cols || y < 0 || y >= cost_map_.rows) {
        throw std::runtime_error(
            "The input coordinates are out of range: " + std::to_string(x) + ", " + std::to_string(y));
    }
    return cost_map_.at<uchar>(y, x);
}
const cv::Mat & CostMap::get_distance_image() const {
    return distance_img_;
}

const cv::Mat & CostMap::get_cost_color_map() const {
    return cost_color_;
}

cv::Mat CostMap::create_exp_cost_map() const {
    // Assert if the input is not a float image
    CV_Assert(distance_map_.type() == CV_32FC1);

    cv::Mat cost_map_float = cv::Mat::zeros(distance_map_.size(), CV_32FC1);
    cost_map_float.setTo(255.0f);

    cv::Mat mask = (distance_map_ >= robot_radius_); // 255 for (d >= r), 0 otherwise

    // Subtract distance_map_ - r only where mask is true
    cv::Mat dist_minus_r;
    cv::subtract(distance_map_, robot_radius_, dist_minus_r, mask, CV_32F);

    cv::Mat cost_exp; // = e^{ -decay_rate * (d-r) }
    cv::exp(-cost_decay_rate_ * dist_minus_r, cost_exp);
    cv::Mat cost_normalized = 255.0f * cost_exp;

    // Copy the cost values only where mask is true
    cost_normalized.copyTo(cost_map_float, mask);

    cv::Mat cost_map_8U;
    cost_map_float.convertTo(cost_map_8U, CV_8U);  // clip to the range [0,255]

    return cost_map_8U;
}

cv::Mat CostMap::create_cost_color_map() const {
    cv::Mat cost_color(cost_map_.size(), CV_8UC3, cv::Scalar::all(0));

    for(int y = 0; y < cost_map_.rows; ++y)
    {
        const uchar* cost_ptr = cost_map_.ptr<uchar>(y);
        const uchar* obstacle_ptr  = obstacle_mask_.ptr<uchar>(y);
        cv::Vec3b* color_ptr  = cost_color.ptr<cv::Vec3b>(y);

        for(int x = 0; x < cost_map_.cols; ++x)
        {
            // If it is an obstacle, paint it black
            if(obstacle_ptr[x] == 0)
            {
                color_ptr[x] = cv::Vec3b(0, 0, 0); // B=0, G=0, R=0
                continue;
            }

            uchar cost = cost_ptr[x];
            double ratio = static_cast<double>(cost) / 255.0;

            // If the cost is 1.0, paint it sky blue
            if (ratio == 1.0) {
                color_ptr[x] = cv::Vec3b(255, 200, 0);
                continue;
            }

            // Paint the cost in a gradient from white to blue
            uchar B = 255;
            uchar G = static_cast<uchar>(255.0 * (1.0 - ratio));
            uchar R = static_cast<uchar>(255.0 * (1.0 - ratio));
            color_ptr[x] = cv::Vec3b(B, G, R);
        }
    }
    return cost_color;
}

int main()
{
    const cv::Scalar white{255,255, 255};
    const int image_width = 800;
    const int image_height = 800;
    cv::Mat img = cv::Mat(image_height, image_width, CV_8UC3, white);

    std::vector<Rectangle> rectangles;
    rectangles.emplace_back(std::array<double, 2>{200., 400.}, 200., 100., 45.);
    rectangles.emplace_back(std::array<double, 2>{600., 600.}, 30., 100., -20.);

    std::vector<Circle> circles;
    circles.emplace_back(std::array<double, 2>{400., 200.}, 50.);
    circles.emplace_back(std::array<double, 2>{600., 250.}, 80.);

    auto draw_background = [&img, &white, &rectangles, &circles](){
        img.setTo(white);
        for (const auto& obstacle : rectangles) {
            obstacle.draw(img);
        }
        for (const auto& obstacle : circles) {
            obstacle.draw(img);
        }
    };
    draw_background();

    CostMap cost_map(img, 20.0, 0.02f);

    std::cout << "cost at (400, 400): " << static_cast<int>(cost_map.get_cost(400, 400)) << std::endl;
    std::cout << "cost at (600, 600): " << static_cast<int>(cost_map.get_cost(600, 600)) << std::endl;

    cv::imshow("img", img);
    cv::imshow("distance_map", cost_map.get_distance_image());
    cv::imshow("costColor (white->red) + obstacle=black", cost_map.get_cost_color_map());
    cv::waitKey(0);

    return EXIT_SUCCESS;
}