#include "energy_refinement.h"
#include "vorticity_confinement.h"
#include <Eigen/Core>

void energy_refinement(
    const Eigen::MatrixXd & x,
    Eigen::MatrixXd & v, 
    const int numofparticles, 
    const double h,
    const double dt
    ){
    Eigen::MatrixXd f;
    f.resize(numofparticles, 3);
    f.setZero();

    //vorticity_confinement(x, v, f, numofparticles, h);
    v += dt * f;
    return ;
}