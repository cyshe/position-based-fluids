#include <Eigen/Core>
// predict position of each particle before applying all forces

// Inputs:
// x position of particle 
// v velocity of particle
// f_ext external forces of each particle
void predict_position(
    Eigen::MatrixXd &x, 
    Eigen::MatrixXd &v,
    const Eigen::MatrixXd f_ext,
    const double dt);