#include <Eigen/Core>

//Find neighboring particles of each particle for the current timestep 
// TODO: check which method used
// Input:
// x position;
// Output:
// N neighbors

void find_neighbor(
    const Eigen::MatrixXd & X, 
    const Eigen::Vector3d lower_bound,
    const Eigen::Vector3d upper_bound, 
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXi & N
);