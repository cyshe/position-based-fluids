#include<Eigen/Core>

template<int DIM>
void calculate_lambda(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXi & N,
    Eigen::VectorXd & lambda,
    const double h,
    const double rho_0 = 1000    
);

template<int DIM>
double W(const Eigen::Matrix<double, DIM, 1> r, const double h);

template<int DIM>
double C(const Eigen::MatrixXd x, 
    const Eigen::MatrixXi N,
    const double rho_0,
    const int i,
    const double h);

template<int DIM>
double grad_C_squared(const Eigen::MatrixXd x, 
    const Eigen::MatrixXi N,
    const double rho_0,
    const int i,
    const double h);