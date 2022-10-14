#include <Eigen/Core>
// Run Jacobi loop, number of iterations = iter

template <int DIM>
void animate_sph(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::MatrixXi & N,
    const Eigen::Matrix<double, DIM, 1> & low_bound,
    const Eigen::Matrix<double, DIM, 1> & up_bound,
    const int numofparticles,
    const int iters, 
    const double dt
);