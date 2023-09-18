#pragma once

#include <Eigen/Core>

template <int dim>
std::vector<std::vector<int>> find_neighbors_brute_force(
    const Eigen::VectorXd & x,
    const double h)
{
    int n = x.size() / dim;
    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            const auto& xi = x.template segment<dim>(dim * i);
            const auto& xj = x.template segment<dim>(dim * j);
            if ((xj - xi).norm() < h && (xj - xi).norm() > 0){
                neighbors[i].push_back(j);
            }
        }
    }
    return neighbors;
}