#include "calculate_lambda.h"

void calculate_lambda(
    const Eigen::MatrixXd x,
    const Eigen::MatrixXd N,
    const double h;
    const double rho_0 = 1000;
    Eigen::VectorXd lambda){
    //C_i = rho_i/rho_0 -1
    //rho_i = Sigma m_j * W(p_i-p_j,h)
    double c;
    double grad_c
    for (int i = 0; i < x.rows(); i++){
        c = C(x, N, rho_0, i, h);
        grad_c = grad_C_squared(x, N, rho_0, i, h);
        lamda(i) = -c/grad_c
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

double grad_C_squared(const Eigen::MatrixXd x, 
    const Eigen::MatrixXd N,
    const double rho_0;
    const int i,
    const double h){
    double sum_k = 0;
    for (int k = 0; k < x.rows(); k++){
        Eigen::Vector3d sum_j;
        if (k == i){
            for (int j = 0; j < x.rows(); j++){
                sum_j += 45 * pow(h - (x(i) -x(j).norm), 2) / (3.14 * pow(h, 6)) * x(k);
            }
            sum_j = sum_j/rho_0;
            sum_k += pow(sum_j.norm, 2);
        }
        else{
            sum_k += pow((-45 * pow(h - (x(i) -x(k).norm), 2) / (3.14 * rho_0 * pow(h, 6)) * x(k)).norm, 2);
        }
    }
    return sum_k;
}