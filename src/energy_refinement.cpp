#include "energy_refinement.h"
#include "vorticity_confinement.h"
#include <Eigen/Core>

#define _USE_MATH_DEFINES
#include <math.h>

void energy_refinement(
    const Eigen::MatrixXd & x,
    const Eigen::MatrixXi & N,
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

    //XSPH velocity
    for (int i = 0; i < numofparticles; i++) {
        /*
        for (int j = 0; j < numofparticles; j++){
            double l = (x.row(i) - x.row(j)).norm();
            double W = 0.0;
            if (l <= h){
                
            }
            v.row(i) += 0.01 * (v.row(j)  - v.row(i)) * W;
        }
        */
        for (int it = 0; it < N.cols(); it++){
            int j = N(i, it); 
            double W = 0.0;
            double l = (x.row(i) - x.row(j)).norm();
            if (l <= h && l > 0){
                W =  315 * pow(pow(h, 2) - pow(l,2), 3)/ (64 * M_PI * pow(h,9));
            }
            v.row(i) += 0.01 * (v.row(j)  - v.row(i)) * W;
        }    
    }
    
    return ;
}