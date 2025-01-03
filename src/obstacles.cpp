#include "MPPI/obstacles.hpp"

Rectangle::Rectangle(
    const std::array<double, 2> &center,
    const double width,
    const double height,
    const double angle_deg)
{
    const double angle_rad = angle_deg * M_PI / 180.0;
    const double cos_theta = std::cos(angle_rad);
    const double sin_theta = std::sin(angle_rad);

    const double half_width = width / 2.0;
    const double half_height = height / 2.0;

    const std::array<std::array<double, 2>, 4> corners = {
        std::array<double, 2>{-half_width, -half_height},
        std::array<double, 2>{half_width, -half_height},
        std::array<double, 2>{half_width, half_height},
        std::array<double, 2>{-half_width, half_height}
    };

    for (const auto& c : corners) {
        double x_rot = c[0] * cos_theta - c[1] * sin_theta;
        double y_rot = c[0] * sin_theta + c[1] * cos_theta;

        double x = x_rot + center[0];
        double y = y_rot + center[1];

        vertices_.emplace_back(cv::Point(cvRound(x), cvRound(y)));
    }
}

void Rectangle::draw(cv::Mat& image, const cv::Scalar& color) const
{
    std::vector<std::vector<cv::Point>> fill_contours{vertices_};
    cv::fillPoly(image, fill_contours, color, cv::LINE_AA);
}

Circle::Circle(
    const std::array<double, 2> &center,
    const double radius)
    : center_(center), radius_(radius)
{}

void Circle::draw(cv::Mat& image, const cv::Scalar& color) const {
    cv::circle(image, cv::Point(center_[0], center_[1]), cvRound(radius_), color, cv::FILLED, cv::LINE_AA);
}
