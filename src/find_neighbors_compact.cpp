#pragma once
#include "find_neighbors_compact.h"
#include <iostream>
#include <CompactNSearch>
#include <Eigen/Core>

template <>
std::vector<std::vector<int>> find_neighbors_compact<2>(
    const Eigen::VectorXd & x,
    const double h){
    int DIM = 2;
    CompactNSearch::NeighborhoodSearch nsearch(h);
    int numofparticles = x.rows()/DIM;
    

    if (DIM == 2){
        //extend to 3d
        Eigen::MatrixXd x_3d(numofparticles, 2);
        x_3d.setZero();
        x_3d = Eigen::Map<Eigen::MatrixXd>(x.data(), 2, numofparticles).transpose();
        x_3d = vector<vector<int>>(2, vector<int>numofparticles);
    }
    else{
        //change to matrix
        Eigen::MatrixXd x_3d(numofparticles, 2);
        x_3d.setZero();
        x_3d = Eigen::Map<Eigen::MatrixXd>(x.data(), 3, numofparticles)transpose();
    }

    //Eigen::MatrixXd::Map(point_set[0].data(), numofparticles, 3) = x_3d;

    unsigned int point_set_id = nsearch.add_point_set(x_3d.front().data(), numofparticles);
    nsearch.find_neighbors();
    CompactNSearch::PointSet const& ps_1 = nsearch.point_set(point_set_id);
    std::vector<std::vector<int>> neighbors(numofparticles);
    for (int i = 0; i < ps_1.n_points(); ++i)
    {
    // Get point set 1 neighbors of point set 1.
        neighbors[i] = ps_1.neighbor_list(point_set_1_id, i);
        
    }
    return neighbors;

}



template <>
std::vector<std::vector<int>> find_neighbors_compact<3>(
    const Eigen::VectorXd & x,
    const double h
);