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