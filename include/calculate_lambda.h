#include<Eigen/Core>

void calculate_lambda(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXd & N,
    Eigen::VectorXd & lambda,
    const double h,
    const double rho_0 = 1000    
);

double W(const Eigen::Vector3d r, const double h);

double C(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0,
    const int i,
    const double h);

double grad_C_squared(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0,
    const int i,
    const double h);