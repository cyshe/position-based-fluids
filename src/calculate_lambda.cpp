#include "calculate_lambda.h"

void calculate_lambda(
    const Eigen::MatrixXd x,
    const Eigen::MatrixXd N,
    const double h;
    const double rho_0 = 1000;
    Eigen::VectorXd lambda){
    //C_i = rho_i/rho_0 -1
    //rho_i = Sigma m_j * W(p_i-p_j,h)
    for (int i = 0; i < x.rows(); i++){
        C(x, N, rho_0, i, h);
    }
    
}

double W(const Eigen::Vector3d r, const double h){
    double l = r.norm()
    if (l <= d){
        return 315 * pow(pow(h, 2) - pow(l,2), 3)/ (64 * PI * pow(h,2));
    }
    else{
        return 0.0;
    }
}

double C(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0;
    const int i,
    const double h){
    double rho_i = 0.0;
    for (int j = 0; j < x.rows(); j++){
        if (N(i, j) == 1){
            rho_i += W(x(i) - x(j), h);
        }
    }
    return rho_i/rho_0 - 1.0;
}