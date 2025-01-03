#pragma once
#include <array>
#include <opencv2/opencv.hpp>


struct Rectangle {
    Rectangle(
        const std::array<double, 2> &center,
        const double width,
        const double height,
        const double angle_deg = 0.0);

    void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(100, 100, 100)) const;

    std::vector<cv::Point> vertices_;
};

struct Circle {
    Circle(
        const std::array<double, 2> &center,
        const double radius);

    void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(100, 100, 100)) const;

    std::array<double, 2> center_;
    double radius_;
};