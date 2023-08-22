#pragma once

#include <Eigen/Core>

template <int DIM>
double surface_tension_energy(
    Eigen::VectorXd & X_flat,
    std::vector<std::vector> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa
){
    int n = X_flat.size() / DIM;
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(X_flat.size() / DIM);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            auto& xi = x.segment<2>(2 * i);
            auto& xj = x.segment<2>(2 * j);
            double r = (xj - xi).norm()/h;
            densities(i) += cubic_bspline(r, m*fac);
        }
    }

    double energy = 0.0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            energy += kappa * (densities(i) - densities(neighbors[i][j])) * (densities(i) - densities(neighbors[i][j]));
        }
    }
    return energy;
};


template <int DIM>
Eigen::VectorXd surface_tension_gradient(
    Eigen::VectorXd & X_flat,
    std::vector<std::vector> neighbors,
    const double h,
    const double m,
    const double fac,
    const double kappa
){
    std::vector<Triplet<double>> B_triplets;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                RowVector2d diff = X_curr.row(j) - X_curr.row(i);
                double r = diff.norm() / h;
                double deriv = cubic_bspline_derivative(r, m*fac) / rho_0;

            if (deriv != 0.0) {
                // dci_dxj
                // Negating because constraint is c(J,x) = J - J(x) 
                RowVector2d dc_dx = -(deriv * diff / r / h);
                B_triplets.push_back(Triplet<double>(i, 2 * j, dc_dx(0)));
                B_triplets.push_back(Triplet<double>(i, 2 * j + 1, dc_dx(1)));
            }
        }
    }
    B.setFromTriplets(B_triplets.begin(), B_triplets.end());
};

template <int DIM>
Eigen::SparseMatrix<double> surface_tension_hessian();