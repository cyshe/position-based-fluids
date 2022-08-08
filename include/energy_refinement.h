#include <Eigen/Core>

void energy_refinement(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXi & N,
    Eigen::MatrixXd & v, 
    const int numofparticles, 
    const double h,
    const double dt);