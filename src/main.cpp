#include <cstddef>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <Eigen/Dense>
#include <random>

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

struct Circle {
    Circle(
        const std::array<double, 2> &center,
        const double radius)
        : center_(center), radius_(radius)
    {}

    void draw(cv::Mat& image, const cv::Scalar& color = cv::Scalar(100, 100, 100)) const
    {
        cv::circle(image, cv::Point(center_[0], center_[1]), cvRound(radius_), color, cv::FILLED, cv::LINE_AA);
    }

    std::array<double, 2> center_;
    double radius_;
};


class DiffDriveRobot {
public:
    DiffDriveRobot(const std::array<double, 2> & center, const double init_theta = 0.0, const double radius = 10.0)
    : radius_(radius), center_(center), theta_(init_theta)
    {
    }

    void move(const double v, const double omega, const double dt)
    {
        center_[0] += v * std::cos(theta_) * dt;
        center_[1] += v * std::sin(theta_) * dt;
        theta_ += omega * dt;
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
        const double dx = radius_ * std::cos(theta_);
        const double dy = radius_ * std::sin(theta_);
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
    double theta_;
};


template <int Dim>
Eigen::MatrixXd sample_normal_dist_multiple(
    const Eigen::Matrix<double, Dim, Eigen::Dynamic> & mean /* dim  x num_samples */,
    const Eigen::Matrix<double, Dim, Dim> & cov)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    const size_t num_samples = mean.cols();

    //  L is a lower triangular matrix (Cholesky decomposition)
    //  cov = L * L^T
    Eigen::LLT<Eigen::MatrixXd> cholesky_solver(cov);
    Eigen::MatrixXd L = cholesky_solver.matrixL(); // d×d

    std::normal_distribution<> dist(0.0, 1.0);

    Eigen::MatrixXd Z(Dim, num_samples);
    for(int i = 0; i < Dim; ++i) {
        for(int j = 0; j < num_samples; ++j) {
            Z(i, j) = dist(gen);
        }
    }

    Eigen::MatrixXd X = mean + L * Z; // dim × num_samples
    return X;
}

class MPPIController {
public:
    // alias
    using State = Eigen::Vector3d;

public:
    MPPIController(const State & init_state, const size_t horizon_length, const size_t num_samples)
    : horizon_length_(horizon_length), num_samples_(num_samples), current_state_(init_state)
    {
        noise_cov_ = Eigen::Matrix2d::Identity();
        noise_cov_(0, 0) = 0.1;
        noise_cov_(1, 1) = 0.1;

        last_inputs_ = Eigen::MatrixXd::Zero(2, horizon_length);
        for (int i = 0; i < horizon_length; ++i) {
            last_inputs_(0, i) = 10.0;
        }
    }

    void control(cv::Mat & image)
    {
        for (size_t i = 0; i < num_samples_; ++i) {
            const auto next_inputs = sample_normal_dist_multiple<2>(last_inputs_, noise_cov_);
            // TODO: clamp the inputs

            const auto next_states = simulate(next_inputs, 0.1);

            const cv::Scalar gray{100, 100, 100};
            for (int i = 0; i < horizon_length_; ++i) {
                const auto state = next_states.col(i);
                const auto center = cv::Point(cvRound(state[0]), cvRound(state[1]));
                cv::circle(image, center, 1, gray, cv::FILLED, cv::LINE_AA);
            }
        }
    }


private:
    const size_t horizon_length_;
    const size_t num_samples_;
    State current_state_;
    Eigen::Matrix2d noise_cov_;

    Eigen::MatrixXd last_inputs_;


    State dynamics(const State & state, const double v, const double omega, const double dt)
    {
        State next_state;
        next_state[0] = state[0] + v * std::cos(state[2]) * dt;
        next_state[1] = state[1] + v * std::sin(state[2]) * dt;
        next_state[2] = state[2] + omega * dt;
        return next_state;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> simulate(const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs, const double dt)
    {
        if (inputs.cols() != horizon_length_) {
            throw std::runtime_error("The number of columns of inputs must be equal to the horizon length.");
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> states(3, horizon_length_);
        states.col(0) = current_state_;

        for (size_t i = 1; i < horizon_length_; ++i) {
            const auto prev_state = states.col(i - 1);
            const auto input = inputs.col(i);
            states.col(i) = dynamics(prev_state, input[0], input[1], dt);
        }

        return states;
    }


};


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

    const std::array<double, 2> init_robot_pos{100., 100.};
    const double init_robot_theta = 0.0;
    DiffDriveRobot robot{init_robot_pos, init_robot_theta};
    robot.draw(img);


    const int ESC_KEY = 27;
#if 0
    const int dt_ms = 10;
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
#else

    MPPIController controller{Eigen::Vector3d{init_robot_pos[0], init_robot_pos[1], init_robot_theta}, 100, 100};
    for (int i = 0; i < 100; ++i)
    {
        draw_background();
        controller.control(img);
        cv::imshow("Robot", img);
        const int key = cv::waitKey(0);
        if (key == ESC_KEY) {
            break;
        }
    }
#endif

    return EXIT_SUCCESS;
}