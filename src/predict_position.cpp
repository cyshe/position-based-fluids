#include "predict_position.h"

void predict_position(
    Eigen::MatrixXd x, 
    Eigen::MatrixXd v, 
    const Eigen::MatrixXd f_ext,
    const double dt){
    v = v + dt * f_ext;
    x = x + dt * v;
}