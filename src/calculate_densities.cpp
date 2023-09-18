#include "calculate_densities.h"
#include "cubic_bspline.h"

using namespace Eigen;

template <int dim>
Matrix<double, dim*dim, 1> density_gradient(
    const Matrix<double, dim, 1>& xi,
    const Matrix<double, dim, 1>& xj,
    const double h,
    const double m,
    const double fac)
{
    Matrix<double, dim*dim, 1> grad;
    Vector<double, dim> xij = xj - xi;
    double r = xij.norm() / h;
    double deriv = cubic_bspline_derivative(r, m*fac);

    Vector2d dphi_dx = deriv * xij / r / h;
    grad.template segment<dim>(0) = -dphi_dx;
    grad.template segment<dim>(dim) = dphi_dx;
    return grad;
}

template <int dim>
Eigen::VectorXd calculate_densities(
    const Eigen::VectorXd & x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
)
{
    int n = x.size() / dim;
    assert(neighbors.size() == n);

    double initial_density = cubic_bspline(0.0, m*fac);

    // Eigen::VectorXd densities = Eigen::VectorXd::Constant(n, initial_density);
    Eigen::VectorXd densities = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < neighbors[i].size(); j++){
            const auto& xi = x.template segment<dim>(dim * i);
            const auto& xj = x.template segment<dim>(dim * neighbors[i][j]);
            double r = (xj - xi).norm()/h;
            densities(i) += cubic_bspline(r, m*fac);
        }
    }
    return densities;
};


// explicit instantiations at bottom of file
template Matrix<double, 2*2, 1> density_gradient<2>(
    const Matrix<double, 2, 1>& xi,
    const Matrix<double, 2, 1>& xj,
    const double h,
    const double m,
    const double fac
);


template Eigen::VectorXd calculate_densities<2>(
    const Eigen::VectorXd& x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);

template Eigen::VectorXd calculate_densities<3>(
    const Eigen::VectorXd& x,
    const std::vector<std::vector<int>>& neighbors,
    const double h,
    const double m,
    const double fac
);
