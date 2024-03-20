#include <Eigen/Core>
#include "line_search.h"
#include <iostream>
#include <Eigen/Eigenvalues> 

using namespace Eigen;

template<int DIM>
float line_search(
    const MatrixXd & A,
    const int it,
    const double (*energy_func)(double) 
    ){

    float alpha = 1.0;
    double e_new = energy_func(alpha);
    double e0 = energy_func(0);
    std::cout << "e0: " << e0 << std::endl;

    while (e_new > e0 && alpha > 1e-10){ 
    //    //std::cout << "alpha: " << alpha << std::endl;
        alpha *= 0.5;
        e_new = energy_func(alpha);
    }
    std::cout << "!!!alpha: " << alpha << std::endl;
    if (alpha < 1e-10 && it == 0){
        std::cout << "line search failed" << std::endl;
        SelfAdjointEigenSolver<MatrixXd> es;
        es.compute(MatrixXd(A));
        std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose().head(10) << std::endl;
        //std::cout << delta_X << std::endl;
    }
    return alpha;
}
