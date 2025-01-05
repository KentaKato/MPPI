#include <cstddef>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <array>
#include <Eigen/Dense>
#include <random>

#include "MPPI/obstacles.hpp"
#include "MPPI/cost_map.hpp"


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
    MPPIController(
        const size_t T,
        const size_t num_samples,
        const Eigen::Matrix<double, 2, 1> & goal,
        const CostMap & cost_map)
    : T_(T), num_samples_(num_samples), goal_(goal), cost_map_(cost_map)
    {
        init_members();
    }

    Eigen::MatrixXd control(cv::Mat & image, const State & current_state)
    {
        std::vector<double> costs(num_samples_);
        Eigen::MatrixXd all_sample_inputs = Eigen::MatrixXd::Zero(2 * num_samples_, T_);

        for (size_t k = 0; k < num_samples_; ++k) {
            Eigen::Matrix<double, 2, Eigen::Dynamic> inputs = sample_normal_dist_multiple<2>(last_inputs_, noise_cov_);
            clip_inputs(inputs);
            all_sample_inputs.block(2 * k, 0, 2, T_) = inputs;

            Eigen::Matrix<double, 3, Eigen::Dynamic> states(3, T_ + 1);
            states = simulate(current_state, inputs, dt_);

            const double stage_cost = compute_stage_cost(inputs, states);
            const double terminal_cost = compute_terminal_cost(states.col(T_));
            const double nominal_input_penalty = compute_nominal_input_penalty(inputs);

            costs[k] = stage_cost + terminal_cost + nominal_input_penalty;
            draw_states(states, image);
        }

        Eigen::Matrix<double, 1, Eigen::Dynamic> weights = compute_weights(costs);

        Eigen::MatrixXd inputs_opt = Eigen::MatrixXd::Zero(2, T_);
        for (size_t k = 0; k < num_samples_; ++k) {
            inputs_opt += weights(k) * all_sample_inputs.block(2 * k, 0, 2, T_);
        }
        last_inputs_ = inputs_opt;
        const auto next_states = simulate(current_state, inputs_opt, dt_);
        draw_states(next_states, image, cv::Scalar{0, 0, 200});
        return inputs_opt.col(0);

    }

private:

    // TODO: remove the task specific parameters
    // task specific parameters
    // const double ref_vel_ = 40.0;
    Eigen::Matrix<double, 2, 1> goal_;
    const double dt_ = 0.1;
    const double w_dist_from_goal_ = 1.0;
    const double w_collision_ = 3000.0;

    // constants
    const double lambda_ = 1.5; // temperature
    const double gamma_ = lambda_;

    const size_t T_;
    const size_t num_samples_;
    Eigen::Matrix2d noise_cov_, noise_cov_inv_;

    Eigen::Matrix<double, 2, Eigen::Dynamic> last_inputs_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> nominal_inputs_;
    Eigen::Vector2d input_lower_bound_, input_upper_bound_;

    CostMap cost_map_;

    void init_members() {
        noise_cov_ = Eigen::Matrix2d::Identity();
        noise_cov_(0, 0) = 0.25;
        noise_cov_(1, 1) = 0.25;
        noise_cov_inv_ = noise_cov_.inverse();

        const auto zero_inputs = Eigen::MatrixXd::Zero(2, T_);
        last_inputs_ = zero_inputs;
        nominal_inputs_ = zero_inputs;

        input_lower_bound_ << 0.0, -1.5; // v_min, omega_min
        input_upper_bound_ << 60.0, 1.5; // v_max, omega_max
    }

    void clip_inputs(Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs) {
        for (int t = 0; t < T_; ++t) {
            for (int i = 0; i < 2; ++i) {
                inputs(i, t) = std::clamp(inputs(i, t), input_lower_bound_(i), input_upper_bound_(i));
            }
        }
    }

    double compute_stage_cost(
        const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs,
        const Eigen::Matrix<double, 3, Eigen::Dynamic> & states) {

        // TODO: parameterize
        // static const double w_ref_vel = 5.0;
        // static const double w_smoothness_v = 50.0;
        // static const double w_smoothness_omega = 100.0;

        double cost = 0.0;
        // for (int i = 0; i < T_ - 1; ++i) {
        //     const auto input = inputs.col(i); // v, omega

            // difference from the reference velocity
            // cost += w_ref_vel * std::pow(ref_vel_ - input(0), 2);
            // std::cout << "w_ref_vel cost: " << w_ref_vel * std::pow(ref_vel_ - input(0), 2) << std::endl;

            // smoothness
            // cost += w_smoothness * (inputs.col(i + 1) - input).squaredNorm();
            // if (i != 0) {
            //     const auto & input_v = input(0);
            //     const auto & input_omega = input(1);
            //     const auto & prev_input_v = inputs.col(i - 1)(0);
            //     const auto & prev_input_omega = inputs.col(i - 1)(1);
            //     const double v_smooth_cost = w_smoothness_v * std::pow(input_v - prev_input_v, 2);
            //     const double omega_smooth_cost = w_smoothness_omega * std::pow(input_omega - prev_input_omega, 2);
            //     cost += v_smooth_cost + omega_smooth_cost;
            //     std::cout << "v_smooth_cost: " << v_smooth_cost << ", omega_smooth_cost: " << omega_smooth_cost << std::endl;
            // }
        // }
        for (size_t t = 0; t < T_; ++t) {
            const auto state = states.col(t);
            cost += w_dist_from_goal_ * (goal_ - state.head(2)).squaredNorm();
            cost += w_collision_ * cost_map_.get_cost(cvRound(state(0)), cvRound(state(1)));
        }
        return cost;
    }

    double compute_terminal_cost(const Eigen::Matrix<double, 3, 1> & terminal_state) {
        const double dist_from_goal = (goal_ - terminal_state.head(2)).squaredNorm();
        return 100 * w_dist_from_goal_ * dist_from_goal;
    }


    double compute_nominal_input_penalty(const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs) {
        double  penalty = 0.0;
        for (size_t t = 0; t < T_; ++t) {
            const auto last_input = last_inputs_.col(t);
            const auto nominal_input = nominal_inputs_.col(t);
            const auto input = inputs.col(t);
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

    Eigen::Matrix<double, 1, Eigen::Dynamic> compute_weights(const std::vector<double> & costs) {

        const auto rho= *std::min_element(costs.begin(), costs.end());
        Eigen::Matrix<double, 1, Eigen::Dynamic> weights = Eigen::MatrixXd::Zero(1, num_samples_);
        for (size_t k = 0; k < num_samples_; ++k) {
            // NOTE: To avoid numerical instability, we subtract the minimum cost from the cost
            weights(k) = std::exp(-1.0 / lambda_ * (costs[k] - rho));
        }

        // normalize
        weights /= weights.sum();

        return weights;
    }

    Eigen::Matrix<double, 3, Eigen::Dynamic> simulate(
        const Eigen::Matrix<double, 3, 1> & state_t0,
        const Eigen::Matrix<double, 2, Eigen::Dynamic> & inputs, const double dt) {
        if (inputs.cols() != T_) {
            throw std::runtime_error("The number of columns of inputs must be equal to the horizon length.");
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> states(3, T_ + 1); // x0, x1, ..., xT
        states.col(0) = state_t0;
        for (size_t i = 0; i < T_; ++i) {
            const auto state = states.col(i);
            const auto input = inputs.col(i);
            states.col(i + 1) = dynamics(state, input(0), input(1), dt);
        }
        return states;
    }

    void draw_states(
        const Eigen::Matrix<double, 3, Eigen::Dynamic> & states,
        cv::Mat & image,
        cv::Scalar color = cv::Scalar{200, 200, 200}) const {

    #if 0 // draw the state with transparency
        cv::Mat state_image = image.clone();
        for (int i = 0; i < T_; ++i) {
            const auto state = states.col(i);
            const auto center = cv::Point(cvRound(state(0)), cvRound(state(1)));
            cv::circle(state_image, center, 1, color, cv::FILLED, cv::LINE_AA);
        }
        double alpha = 0.3;
        double beta = 1.0 - alpha;
        cv::addWeighted(state_image, alpha, image, beta, 0.0, image);

    #else
        for (int i = 0; i < T_; i += 2) {
            const auto state = states.col(i);
            const auto center = cv::Point(cvRound(state(0)), cvRound(state(1)));
            cv::circle(image, center, 1, color, cv::FILLED, cv::LINE_AA);
        }
    }
    #endif
};


int main()
{
    const cv::Scalar white{255,255, 255};
    const int image_width = 900;
    const int image_height = 900;

    Eigen::Matrix<double ,2, 1> goal{700, 700};

    cv::Mat img = cv::Mat(image_height, image_width, CV_8UC3, white);

    std::vector<Rectangle> rectangles;
    rectangles.emplace_back(std::array<double, 2>{190., 400.}, 190., 90., 45.);
    rectangles.emplace_back(std::array<double, 2>{500., 500.}, 30., 100., -20.);
    rectangles.emplace_back(std::array<double, 2>{400., 300.}, 100., 200., 20.);
    rectangles.emplace_back(std::array<double, 2>{800., 200.}, 250., 150., -40.);

    std::vector<Circle> circles;
    circles.emplace_back(std::array<double, 2>{400., 200.}, 50.);
    circles.emplace_back(std::array<double, 2>{550., 700.}, 70.);
    circles.emplace_back(std::array<double, 2>{500., 600.}, 50.);
    circles.emplace_back(std::array<double, 2>{200., 750.}, 70.);

    auto draw_background = [&img, &white, &rectangles, &circles](){
        for (const auto& obstacle : rectangles) {
            obstacle.draw(img);
        }
        for (const auto& obstacle : circles) {
            obstacle.draw(img);
        }
    };

    const std::array<double, 2> init_robot_pos{100., 100.};
    const double init_robot_theta = 0.0;
    const double robot_radius = 10.0;
    DiffDriveRobot robot{init_robot_pos, init_robot_theta, robot_radius};
    robot.draw(img);


    const int ESC_KEY = 27;
    img.setTo(white);
    draw_background();
    cv::Mat cost_map_img = img.clone();
    const CostMap cost_map {cost_map_img, robot_radius, 0.02f};
    const cv::Mat cost_color_map = cost_map.get_cost_color_map().clone();
    MPPIController controller{50, 500, goal, cost_map};
    for (int i = 0; i < 1000; ++i)
    {
        img.setTo(white);
        img = cost_color_map.clone();
        draw_background();

        cv::circle(img, cv::Point(cvRound(goal(0)), cvRound(goal(1))), 5, cv::Scalar(0, 200, 0), cv::FILLED, cv::LINE_AA);
        const auto next_input = controller.control(img, robot.get_state());
        robot.move(next_input(0), next_input(1), 0.1);
        robot.draw(img);
        cv::imshow("Robot", img);
        const int key = cv::waitKey(10);
        if (key == ESC_KEY) {
            break;
        }
    }

    return EXIT_SUCCESS;
}