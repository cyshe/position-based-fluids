#include <Eigen/Core>

void vorticity_confinement(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXd & v, 
    Eigen::MatrixXd & f,
    const int numofparticles, 
    const double h);