#pragma once

#include <opencv2/opencv.hpp>


class CostMap {

public:

    CostMap(const cv::Mat& map, const double robot_radius, const float cost_decay_rate);

    bool update_cost_map(const cv::Mat & map);
    float get_cost(const int x, const int y) const;
    const cv::Mat& get_distance_image() const;
    const cv::Mat& get_cost_color_map() const;

private:
    const double robot_radius_;
    const double cost_decay_rate_;

    cv::Mat obstacle_mask_;
    cv::Mat distance_map_;
    cv::Mat distance_img_;
    cv::Mat cost_map_;
    cv::Mat cost_color_;

    cv::Mat create_exp_cost_map() const;
    cv::Mat create_cost_color_map() const;
};
