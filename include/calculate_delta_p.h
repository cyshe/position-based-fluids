#include<Eigen/Core>

void calculate_delta_p(
    Eigen::MatrixXd & delta_p,
    const Eigen::MatrixXd & X,
    const Eigen::MatrixXd & N,
    const Eigen::VectorXd & lambda,
    const double rho_0,
    const int numofparticles,
    const double h,
    const double k
);