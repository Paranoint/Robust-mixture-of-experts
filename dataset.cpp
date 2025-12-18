#include "dataset.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

Dataset read_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open CSV: " + path);
    }

    std::string header;
    if (!std::getline(in, header)) {
        throw std::runtime_error("CSV appears empty.");
    }

    std::vector<std::vector<double>> rows;
    std::vector<double> targets;
    std::string line;
    int dim = -1;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> values;
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) values.push_back(std::stod(cell));
        }
        if (values.empty()) continue;
        targets.push_back(values.back());
        values.pop_back();
        if (dim == -1) dim = static_cast<int>(values.size());
        if (static_cast<int>(values.size()) != dim) {
            throw std::runtime_error("Inconsistent feature dimension in CSV");
        }
        rows.push_back(std::move(values));
    }
    if (rows.empty()) {
        throw std::runtime_error("No data rows found in CSV.");
    }

    Dataset data;
    const int n = static_cast<int>(rows.size());
    const int d = dim;
    data.X.resize(n, d);
    data.y.resize(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) data.X(i, j) = rows[i][j];
        data.y(i) = targets[i];
    }
    return data;
}
