#include "moe_training.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kEps = 1e-12;
constexpr double kPi = 3.14159265358979323846;

Eigen::VectorXd augment_with_bias(const Eigen::VectorXd& x) {
    Eigen::VectorXd aug(x.size() + 1);
    aug.head(x.size()) = x;
    aug.tail(1) << 1.0;
    return aug;
}

Eigen::VectorXd softmax(const Eigen::VectorXd& logits) {
    double max_coeff = logits.maxCoeff();
    Eigen::VectorXd exps = (logits.array() - max_coeff).exp();
    double sum = exps.sum();
    if (sum < kEps) {
        return Eigen::VectorXd::Constant(logits.size(), 1.0 / logits.size());
    }
    return exps / sum;
}

Eigen::VectorXd gating_probabilities(const MoEModel& model, const Eigen::VectorXd& x_aug) {
    Eigen::VectorXd logits(model.gating.size());
    for (size_t k = 0; k < model.gating.size(); ++k) {
        logits(static_cast<int>(k)) = model.gating[k].dot(x_aug);
    }
    return softmax(logits);
}

Eigen::MatrixXd compute_responsibilities(
    const Dataset& data,
    const MoEModel& model,
    double robust_gamma) {
    const int n = data.samples();
    const int k = static_cast<int>(model.experts.size());
    Eigen::MatrixXd resp(n, k);

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x = data.X.row(i).transpose();
        Eigen::VectorXd x_aug = augment_with_bias(x);
        Eigen::VectorXd gating = gating_probabilities(model, x_aug);
        Eigen::VectorXd numer(k);
        for (int j = 0; j < k; ++j) {
            const auto& expert = model.experts[j];
            double mean = expert.weights.dot(x) + expert.bias;
            double resid = data.y(i) - mean;
            double var = std::max(expert.variance, kEps);
            double gaussian = (1.0 / std::sqrt(2.0 * kPi * var)) * std::exp(-0.5 * resid * resid / var);
            double val = gating(j) * gaussian;
            if (robust_gamma > 0.0) {
                val = std::pow(val + kEps, 1.0 / (1.0 + robust_gamma));
            }
            numer(j) = val;
        }
        double denom = numer.sum();
        if (denom < kEps) denom = kEps;
        resp.row(i) = (numer / denom).transpose();
    }
    return resp;
}

void update_experts(
    const Dataset& data,
    MoEModel& model,
    const Eigen::MatrixXd& resp,
    double min_variance) {
    const int n = data.samples();
    const int d = data.input_dim();
    const int k = static_cast<int>(model.experts.size());

    Eigen::MatrixXd X_aug(n, d + 1);
    X_aug.leftCols(d) = data.X;
    X_aug.col(d) = Eigen::VectorXd::Ones(n);

    for (int expert_idx = 0; expert_idx < k; ++expert_idx) {
        Eigen::VectorXd weights = resp.col(expert_idx);
        double weight_sum = weights.sum();
        Eigen::MatrixXd XtWX = Eigen::MatrixXd::Zero(d + 1, d + 1);
        Eigen::VectorXd XtWy = Eigen::VectorXd::Zero(d + 1);

        for (int i = 0; i < n; ++i) {
            double w = weights(i);
            Eigen::VectorXd xi = X_aug.row(i).transpose();
            XtWX.noalias() += w * (xi * xi.transpose());
            XtWy.noalias() += w * xi * data.y(i);
        }
        XtWX += 1e-6 * Eigen::MatrixXd::Identity(d + 1, d + 1);
        Eigen::VectorXd solution = XtWX.ldlt().solve(XtWy);
        auto& expert = model.experts[expert_idx];
        expert.weights = solution.head(d);
        expert.bias = solution(d);

        double var_num = 0.0;
        for (int i = 0; i < n; ++i) {
            double pred = expert.weights.dot(data.X.row(i)) + expert.bias;
            double resid = data.y(i) - pred;
            var_num += weights(i) * resid * resid;
        }
        expert.variance = std::max(var_num / (weight_sum + kEps), min_variance);
    }
}

void update_gating(
    const Dataset& data,
    MoEModel& model,
    const Eigen::MatrixXd& resp,
    const TrainOptions& opts) {
    const int n = data.samples();
    const int d = data.input_dim();
    const int k = static_cast<int>(model.gating.size());

    std::vector<Eigen::VectorXd> x_aug(n);
    for (int i = 0; i < n; ++i) {
        x_aug[i] = augment_with_bias(data.X.row(i).transpose());
    }

    for (int step = 0; step < opts.gating_steps; ++step) {
        std::vector<Eigen::VectorXd> grads(k, Eigen::VectorXd::Zero(d + 1));
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd probs = gating_probabilities(model, x_aug[i]);
            for (int expert_idx = 0; expert_idx < k; ++expert_idx) {
                double diff = resp(i, expert_idx) - probs(expert_idx);
                grads[expert_idx].noalias() += diff * x_aug[i];
            }
        }
        for (int expert_idx = 0; expert_idx < k; ++expert_idx) {
            model.gating[expert_idx].noalias() += (opts.gating_lr / static_cast<double>(n)) * grads[expert_idx];
        }
    }
}

TrainResult evaluate_model(const Dataset& data, const MoEModel& model) {
    const int n = data.samples();
    TrainResult result;
    result.model = model;
    result.predictions.resize(n);
    Eigen::VectorXd residuals(n);

    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd x = data.X.row(i).transpose();
        Eigen::VectorXd probs = gating_probabilities(model, augment_with_bias(x));
        double pred = 0.0;
        for (size_t k = 0; k < model.experts.size(); ++k) {
            double mean = model.experts[k].weights.dot(x) + model.experts[k].bias;
            pred += probs(static_cast<int>(k)) * mean;
        }
        result.predictions[i] = pred;
        residuals(i) = data.y(i) - pred;
    }

    double mse = residuals.squaredNorm() / static_cast<double>(n);
    result.rmse = std::sqrt(mse);

    std::vector<double> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = residuals(i) * residuals(i);
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    int top = std::max(1, static_cast<int>(std::round(0.1 * n)));
    double top_mean = 0.0;
    for (int i = 0; i < top; ++i) top_mean += sorted[i];
    top_mean /= static_cast<double>(top);
    result.robustness_ratio = top_mean / (mse + kEps);
    return result;
}

MoEModel initialize_model(const Dataset& data, const TrainOptions& opts) {
    std::mt19937 gen(opts.seed);
    std::normal_distribution<double> norm(0.0, 0.5);
    MoEModel model;
    model.experts.resize(opts.n_experts);
    for (auto& expert : model.experts) {
        expert.weights = Eigen::VectorXd(data.input_dim());
        for (int i = 0; i < data.input_dim(); ++i) {
            expert.weights(i) = norm(gen);
        }
        expert.bias = norm(gen);
        expert.variance = 1.0;
    }
    model.gating.resize(opts.n_experts, Eigen::VectorXd(data.input_dim() + 1));
    for (auto& vec : model.gating) {
        for (int i = 0; i < vec.size(); ++i) vec(i) = 0.1 * norm(gen);
    }
    return model;
}

}  // namespace

TrainResult train_moe(const Dataset& data, const TrainOptions& opts, double robust_gamma) {
    MoEModel model = initialize_model(data, opts);
    for (int iter = 0; iter < opts.max_iters; ++iter) {
        Eigen::MatrixXd resp = compute_responsibilities(data, model, robust_gamma);
        update_experts(data, model, resp, opts.min_variance);
        update_gating(data, model, resp, opts);
    }
    return evaluate_model(data, model);
}

void print_summary(const std::string& title, const TrainResult& result) {
    std::cout << title << "\n";
    std::cout << "  RMSE: " << result.rmse << "\n";
    std::cout << "  Robustness ratio (top10% / overall MSE): " << result.robustness_ratio << "\n";
    for (size_t k = 0; k < result.model.experts.size(); ++k) {
        const auto& expert = result.model.experts[k];
        std::cout << "  Expert " << k << " weights:";
        for (int j = 0; j < expert.weights.size(); ++j) {
            std::cout << " " << expert.weights(j);
        }
        std::cout << " | bias=" << expert.bias << " | var=" << expert.variance << "\n";
    }
}
