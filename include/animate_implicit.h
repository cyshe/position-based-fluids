#pragma once

#include <Eigen/Core>
// use implicit method to iterate

template <int DIM>
void animate_implicit(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::VectorXd & J, 
    Eigen::MatrixXi & N,
    const Eigen::Matrix<double, DIM, 1> & low_bound,
    const Eigen::Matrix<double, DIM, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
);

// template <typename Derived>
    // const Eigen::MatrixBase<Derived>& x, 

inline double cubic_bspline(double r, double fac)
{
    double ret = 0.0;
    if (r <= 1 && r >= 0){
        ret = (1 - 1.5 * r * r *(1 - 0.5 *r)) * fac;
    }
    else if (r > 1 && r <= 2){
        ret = (2-r)*(2-r)*(2-r) * fac /4;
    }
    return ret;
}

// template <typename Derived>
inline double cubic_bspline_derivative(double r, double fac)
{
    double ret = 0.0;

    if (r <= 1 && r > 0) {
        ret = -fac * (3*r - 9 * r * r/4);
    }
    else if (r > 1 && r <= 2){
        ret = -fac * 0.75 * (2 - r) * (2 - r);
    }
    return ret;
}
