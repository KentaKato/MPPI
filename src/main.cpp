#include <cstddef>
#include <opencv2/opencv.hpp>
#include <array>
#include <Eigen/Dense>
#include <random>

#include "MPPI/obstacles.hpp"


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

    Eigen::Vector3d get_state() const
    {
        return Eigen::Vector3d(center_[0], center_[1], theta_);
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
    MPPIController(const size_t horizon_length, const size_t num_samples)
    : horizon_length_(horizon_length), num_samples_(num_samples)
    {
        init_members();
    }

    Eigen::MatrixXd control(cv::Mat & image, const State & current_state)
    {
        current_state_ = current_state;

        std::vector<double> costs(num_samples_);
        Eigen::MatrixXd all_inputs = Eigen::MatrixXd::Zero(2 * num_samples_, horizon_length_);

        for (size_t i = 0; i < num_samples_; ++i) {
            const auto next_inputs = sample_normal_dist_multiple<2>(last_inputs_, noise_cov_);
            // TODO: clamp the inputs

            const auto next_states = simulate(next_inputs, dt_);
            draw_states(next_states, image);
            all_inputs.block(2 * i, 0, 2, horizon_length_) = next_inputs;

            // first term of exp
            double cost = compute_cost(next_inputs, next_states);

            // second term of exp
            double nominal_input_penalty = compute_nominal_input_penalty(next_inputs);

            // FIXME:
            nominal_input_penalty = 0.0;
            double total_cost = cost + nominal_input_penalty;

            costs[i] = total_cost;
        }

        const double rho = *std::min_element(costs.begin(), costs.end());

        Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(1, num_samples_);
        for (size_t i = 0; i < num_samples_; ++i) {
            weights(0, i) = std::exp(-1.0 / temperature_ * (costs[i] - rho));
        }

        const double eta = weights.sum();
        std::cout << "weights before: " << weights << std::endl;
        weights /= eta;
        std::cout << "weights: " << weights << std::endl;

        Eigen::MatrixXd new_inputs = Eigen::MatrixXd::Zero(2, horizon_length_);
        for (size_t i = 0; i < num_samples_; ++i) {
            new_inputs += weights(0, i) * all_inputs.block(2 * i, 0, 2, horizon_length_);
        }
        last_inputs_ = new_inputs;
        const auto next_states = simulate(new_inputs, dt_);
        draw_states(next_states, image, cv::Scalar{0, 0, 200});
        const auto first_input = new_inputs.col(0);
        std::cout <<"new_inputs: " << new_inputs << std::endl;
        return first_input;
    }


private:

    // TODO: remove the task specific parameters
    // task specific parameters
    const double ref_vel_ = 10.0;
    Eigen::Matrix<double, 2, 1> goal_;
    const double dt_ = 0.1;

    // constants
    const double gamma_ = 1.0;
    const double temperature_ = 200.0;

    const size_t horizon_length_;
    const size_t num_samples_;
    State current_state_;
    Eigen::Matrix2d noise_cov_, noise_cov_inv_;

    Eigen::Matrix<double, 2, Eigen::Dynamic> last_inputs_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> nominal_inputs_;

    void init_members() {
        noise_cov_ = Eigen::Matrix2d::Identity();
        noise_cov_(0, 0) = 1;
        noise_cov_(1, 1) = 0.1;
        noise_cov_inv_ = noise_cov_.inverse();
        std::cout << "noise cov: " << noise_cov_ << std::endl;

        const auto zero_inputs = Eigen::MatrixXd::Zero(2, horizon_length_);

        // TODO:ここなんかおかしい
        last_inputs_ = zero_inputs;
        for (int i = 0; i < horizon_length_; ++i) {
            last_inputs_.col(i)(0) = ref_vel_;
        }

        nominal_inputs_ = zero_inputs;

        goal_ << 600.0, 600.0;
    }

    double compute_cost(
        const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs,
        const Eigen::Matrix<double, 3, Eigen::Dynamic> & states) {
        static const double w_ref_vel = 1.0;
        static const double w_smoothness = 10.0;
        static const double w_dist_from_goal = 100.0;

        double cost = 0.0;
        for (int i = 0; i < horizon_length_ - 1; ++i) {
            const auto input = inputs.col(i); // v, omega

            // difference from the reference velocity
            // cost += w_ref_vel * std::pow(ref_vel_ - input(0), 2);

            // smoothness
            // cost += w_smoothness * (inputs.col(i + 1) - input).squaredNorm();
        }

        // distance from the goal
        const auto last_state = states.col(horizon_length_ - 1);
        const double dist_from_goal = (goal_ - last_state.head(2)).norm();
        cost += w_dist_from_goal * dist_from_goal;

        return cost;
    }

    double compute_nominal_input_penalty(const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs) {
        double  penalty = 0.0;
        for (int i = 0; i < horizon_length_ - 1; ++i) {
            const auto input = inputs.col(i);
            const auto last_input = last_inputs_.col(i);
            const auto nominal_input = nominal_inputs_.col(i);
            penalty += (last_input.transpose() - nominal_input.transpose()) * noise_cov_inv_ * input;
        }
        penalty *= gamma_;
        return penalty;
    }

    State dynamics(const State & state, const double v, const double omega, const double dt) {
        State next_state;
        next_state[0] = state[0] + v * std::cos(state[2]) * dt;
        next_state[1] = state[1] + v * std::sin(state[2]) * dt;
        next_state[2] = state[2] + omega * dt;
        return next_state;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> simulate(const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs, const double dt) {
        if (inputs.cols() != horizon_length_) {
            throw std::runtime_error("The number of columns of inputs must be equal to the horizon length.");
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> states(3, horizon_length_);
        states.col(0) = current_state_;

        for (size_t i = 1; i < horizon_length_; ++i) {
            const auto prev_state = states.col(i - 1);
            const auto input = inputs.col(i-1);
            states.col(i) = dynamics(prev_state, input(0), input(1), dt);
        }

        return states;
    }

    void draw_states(
        const Eigen::Matrix<double, 3, Eigen::Dynamic> & next_states,
        cv::Mat & image,
        cv::Scalar color = cv::Scalar{200, 200, 200}) const {
        for (int i = 0; i < horizon_length_; ++i) {
            const auto state = next_states.col(i);
            const auto center = cv::Point(cvRound(state(0)), cvRound(state(1)));
            cv::circle(image, center, 1, color, cv::FILLED, cv::LINE_AA);
        }
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

    MPPIController controller{50, 100};
    for (int i = 0; i < 1000; ++i)
    {
        std::cout << "-------------------" << std::endl;
        draw_background();
        const auto next_input = controller.control(img, robot.get_state());
        robot.move(next_input(0), next_input(1), 0.1);
        robot.draw(img);
        cv::imshow("Robot", img);
        const int key = cv::waitKey(0);
        if (key == ESC_KEY) {
            break;
        }
    }
#endif

    return EXIT_SUCCESS;
}