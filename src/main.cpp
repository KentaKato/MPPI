#include <opencv2/opencv.hpp>
#include <array>


struct Rectangle {
    Rectangle(
        const std::array<double, 2> &center,
        const double width,
        const double height,
        const double angle_deg = 0.0)
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

    void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(100, 100, 100)) const
    {
        std::vector<std::vector<cv::Point>> fill_contours{vertices_};
        cv::fillPoly(image, fill_contours, color, cv::LINE_AA);
    }

    std::vector<cv::Point> vertices_;

};

class DiffDriveRobot {
public:
    DiffDriveRobot(const std::array<double, 2> & center, const double init_yaw = 0.0, const double radius = 10.0)
    : radius_(radius), center_(center), yaw_(init_yaw)
    {
    }

    void move(const double v, const double omega, const double dt)
    {
        center_[0] += v * std::cos(yaw_) * dt;
        center_[1] += v * std::sin(yaw_) * dt;
        yaw_ += omega * dt;
    }

    void draw(cv::Mat& image) const
    {
        const auto center_cv = cv::Point(cvRound(center_[0]), cvRound(center_[1]));
        cv::circle(
            image,
            center_cv,
            radius_,
            robot_color_,
            cv::FILLED,
            cv::LINE_AA);

        // draw the orientation
        const double dx = radius_ * std::cos(yaw_);
        const double dy = radius_ * std::sin(yaw_);
        cv::line(
            image,
            center_cv,
            cv::Point(center_[0] + dx, center_[1] + dy),
            orientation_color_,
            2,
            cv::LINE_AA);
    }

private:
    const double radius_;
    const cv::Scalar robot_color_ = cv::Scalar(0, 0, 200);
    const cv::Scalar orientation_color_ = cv::Scalar(0, 0, 0);

    std::array<double, 2> center_;
    double yaw_;
};


int main()
{
    const cv::Scalar white{255,255, 255};
    const int image_width = 800;
    const int image_height = 800;
    cv::Mat img = cv::Mat(image_height, image_width, CV_8UC3, white);

    std::vector<Rectangle> obstacles;
    obstacles.emplace_back(std::array<double, 2>{200., 400.}, 200., 100., 45.);
    obstacles.emplace_back(std::array<double, 2>{600., 600.}, 30., 100., -20.);

    auto draw_background = [&img, &white, &obstacles](){
        img.setTo(white);
        for (const auto& obstacle : obstacles) {
            obstacle.draw(img);
        }
    };

    DiffDriveRobot robot{{400, 400}};
    robot.draw(img);

    const int dt_ms = 10;
    const int ESC_KEY = 27;
    for (int i = 0; i < 1000; ++i)
    {
        draw_background();

        robot.move(10.0, 0.1, static_cast<double>(dt_ms / 1000.0));
        robot.draw(img);

        cv::imshow("Robot", img);
        cv::waitKey(10);

        const int key = cv::waitKey(dt_ms);

        if (key == ESC_KEY) {
            break;
        }
    }

    return EXIT_SUCCESS;
}