#pragma once
#include <Eigen/Core>

template<int dim>
std::vector<std::vector<int>> find_neighbors_compact(
    const Eigen::VectorXd & x,
    const double h
);