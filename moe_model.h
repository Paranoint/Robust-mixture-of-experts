#pragma once

#include <Eigen/Dense>
#include <vector>

struct ExpertParams {
    Eigen::VectorXd weights;
    double bias = 0.0;
    double variance = 1.0;
};

struct MoEModel {
    std::vector<ExpertParams> experts;
    std::vector<Eigen::VectorXd> gating;  // softmax weights incl. bias term
};

struct TrainOptions {
    int n_experts = 3;
    int max_iters = 50;
    int gating_steps = 5;
    double gating_lr = 0.05;
    double min_variance = 1e-3;
    double robust_gamma = 0.5;
    unsigned int seed = 42;
};

struct TrainResult {
    MoEModel model;
    double rmse = 0.0;
    double robustness_ratio = 0.0;
    std::vector<double> predictions;
};
