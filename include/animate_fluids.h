#include <Eigen/Core>
// Run Jacobi loop, number of iterations = iter

void animate_fluids(
    Eigen::MatrixXd & X, 
    Eigen::MatrixXd & V, 
    Eigen::MatrixXd & N,
    const Eigen::Vector3d & low_bound,
    const Eigen::Vector3d & up_bound,
    const int numofparticles,
    const double iters, 
    const double dt
);