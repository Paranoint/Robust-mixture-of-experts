#pragma once

#include <string>

#include "dataset.h"
#include "moe_model.h"

TrainResult train_moe(const Dataset& data, const TrainOptions& opts, double robust_gamma);
void print_summary(const std::string& title, const TrainResult& result);
