#include <Eigen/Core>

//Find neighboring particles of each particle for the current timestep 
// TODO: check which method used
// Input:
// x position;
// Output:
// N neighbors

void find_neighbor(
    const Eigen::MatrixXd x,
    Eigen::MatrixXd N
);