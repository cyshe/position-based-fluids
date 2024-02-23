#pragma once
#include "find_neighbors_compact.h"
#include <iostream>
#include <CompactNSearch>
#include <Eigen/Core>
#include <vector>

using namespace CompactNSearch;

template <>
std::vector<std::vector<int>> find_neighbors_compact<2>(
    const Eigen::VectorXd & x,
    const double h){
    int DIM = 2;

    NeighborhoodSearch nsearch(h);
    int numofparticles = x.rows()/DIM;
    Eigen::VectorXd x_new = x;
    Eigen::MatrixXd x_3d;
    if (DIM == 2){
        //extend to 3d
        x_3d.setZero();
        Eigen::MatrixXd x_2d = Eigen::Map<Eigen::MatrixXd>(x_new.data(), 2, numofparticles).transpose();

        x_3d.col(0) = x_2d.col(0);
        x_3d.col(1) = x_2d.col(1);
        x_3d.col(2).setZero();
    }
    else{
        //change to matrix
        x_3d = Eigen::Map<Eigen::MatrixXd>(x_new.data(), 3, numofparticles).transpose();
    }

    //Eigen::MatrixXd::Map(point_set[0].data(), numofparticles, 3) = x_3d;

    std::vector<std::array<Real, 3>> point_set_1;
    for (int i = 0; i < numofparticles; i++){
        std::array<Real, 3> point;
        point[0] = x_3d(i, 0);
        point[1] = x_3d(i, 1);
        point[2] = x_3d(i, 2);
        point_set_1.push_back(point);
    }

    unsigned int point_set_id = nsearch.add_point_set(  point_set_1.front().data(), numofparticles);
    nsearch.find_neighbors();
    std::vector<std::vector<int>> neighbors(numofparticles);
    CompactNSearch::PointSet const& ps_1 = nsearch.point_set(point_set_id);
    for (int i = 0; i < ps_1.n_points(); ++i)
    {
        // Get point set 1 neighbors of point set 1.
        for (size_t j = 0; j < ps_1.n_neighbors(point_set_id, i); ++j)
        {
            // Return the point id of the jth neighbor of the ith particle in the point_set_1.
            neighbors[i].push_back(ps_1.neighbor(point_set_id, i, j));
        }
    }
    return neighbors;

}



template <>
std::vector<std::vector<int>> find_neighbors_compact<3>(
    const Eigen::VectorXd & x,
    const double h
);