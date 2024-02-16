#include <Eigen/Core>

using namespace Eigen;

template<int DIM>
float line_search(
    const MatrixXd & A,
    const int it,
    const double (*energy_func)(double) 
    );