#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "MPPI/cost_map.hpp"
#include "MPPI/obstacles.hpp"


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