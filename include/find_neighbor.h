#include <Eigen/Core>

//Find neighboring particles of each particle for the current timestep 
// TODO: check which method used
// Input:
// x position;
// Output:
// N neighbors

template<int DIM>
void find_neighbor(
    const Eigen::MatrixXd & X, 
    const Eigen::Matrix<double, DIM, 1> lower_bound,
    const Eigen::Matrix<double, DIM, 1> upper_bound, 
    const double cell_size,
    const int numofparticles,
    Eigen::MatrixXi & N
);