#pragma once

#include <Eigen/Core>
// use implicit method to iterate

template <int DIM>
void animate_smoke(
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

// Computes derivative of norm of a vector
// d/dx (||x||) = x / r  where r = ||x||
template <int DIM>
Eigen::Matrix<double,DIM,1> norm_derivative(
    const Eigen::Matrix<double,DIM,1>& x, double r,
    double eps = 0.01) {
  return x / (r+eps);
}

// Computes hessian of norm of a vector
// d^2/dx^2 (||x||) = I / r - xx^T/r^3  
//  where r = ||x||
template <int DIM>
Eigen::Matrix<double,DIM,DIM> norm_hessian(
    const Eigen::Matrix<double,DIM,1>& x, double r, double eps=0.01) {
  Eigen::Matrix<double,DIM,DIM> I = Eigen::Matrix<double,DIM,DIM>::Identity();
  r = r + eps;
  return I / r - x * x.transpose() / (r * r * r);
}

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

    if (r <= 1 && r >= 0) {
        ret = -fac * (3*r - 9 * r * r/4);
    }
    else if (r >= 1 && r <= 2){
        ret = -fac * 0.75 * (2 - r) * (2 - r);
    }
    return ret;
}

inline double cubic_bspline_hessian(double r, double fac)
{
    double ret = 0.0;

    if (r <= 1 && r >= 0) {
        ret = -fac * (3 - 9 * r / 2);
    }
    else if (r >= 1 && r <= 2){
        ret = fac * 1.5 * (2 - r);
    }
    return ret;
    
    
}
