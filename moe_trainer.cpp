#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dataset.h"
#include "moe_model.h"
#include "moe_training.h"

void usage(const char* prog) {
    std::cerr << "Usage: " << prog << " data.csv [options]\n"
              << "Options:\n"
              << "  --experts N        Number of experts (default 3)\n"
              << "  --iters N          EM iterations (default 50)\n"
              << "  --gating-lr V      Learning rate for gating update (default 0.05)\n"
              << "  --gating-steps N   Gradient steps per EM iteration (default 5)\n"
              << "  --gamma V          Robust gamma divergence parameter (default 0.5)\n"
              << "  --seed N           RNG seed (default 42)\n";
}

TrainOptions parse_args(int argc, char** argv, std::string& csv_path) {
    if (argc < 2) {
        usage(argv[0]);
        throw std::runtime_error("CSV path is required.");
    }
    csv_path = argv[1];
    TrainOptions opts;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
        };
        if (arg == "--experts") {
            need_value(arg);
            opts.n_experts = std::stoi(argv[++i]);
        } else if (arg == "--iters") {
            need_value(arg);
            opts.max_iters = std::stoi(argv[++i]);
        } else if (arg == "--gating-lr") {
            need_value(arg);
            opts.gating_lr = std::stod(argv[++i]);
        } else if (arg == "--gating-steps") {
            need_value(arg);
            opts.gating_steps = std::stoi(argv[++i]);
        } else if (arg == "--gamma") {
            need_value(arg);
            opts.robust_gamma = std::stod(argv[++i]);
        } else if (arg == "--seed") {
            need_value(arg);
            opts.seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (opts.n_experts < 1) throw std::runtime_error("Need at least one expert.");
    if (opts.max_iters < 1) throw std::runtime_error("Need at least one iteration.");
    if (opts.robust_gamma < 0.0) throw std::runtime_error("Gamma must be non-negative.");
    return opts;
}

int main(int argc, char** argv) {
    try {
        std::string csv_path;
        TrainOptions opts = parse_args(argc, argv, csv_path);
        Dataset data = read_csv(csv_path);

        TrainResult standard = train_moe(data, opts, 0.0);
        TrainResult robust = train_moe(data, opts, opts.robust_gamma);

        std::cout << std::fixed << std::setprecision(5);
        print_summary("Standard MoE (Gaussian EM)", standard);
        print_summary("Robust MoE (gamma-divergence)", robust);
        std::cout << "RMSE improvement (standard - robust): " << (standard.rmse - robust.rmse) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
