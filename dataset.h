#pragma once

#include <Eigen/Dense>
#include <string>

struct Dataset {
    Eigen::MatrixXd X;  // rows: samples, cols: features
    Eigen::VectorXd y;

    int input_dim() const { return static_cast<int>(X.cols()); }
    int samples() const { return static_cast<int>(X.rows()); }
};

Dataset read_csv(const std::string& path);
