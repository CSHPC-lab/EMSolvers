#include "fem_simulation.hpp"

#include <openacc.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

// vectorのポインタ
double *electric_field_x_ptr_;
double *electric_field_y_ptr_;
double *electric_field_z_ptr_;
int *connectivity_x_ptr_;
int *connectivity_y_ptr_;
int *connectivity_z_ptr_;
int *source_position_x_ptr_;
int *source_position_y_ptr_;
int *source_position_z_ptr_;
std::array<int, 3> *source_dipole_position_x_ptr_;
std::array<int, 3> *source_dipole_position_y_ptr_;
std::array<int, 3> *source_dipole_position_z_ptr_;
std::array<double, 12> electric_field_0;
std::array<std::array<double, 4>, 4> Kmat00;
std::array<std::array<double, 4>, 4> Kmat01;
std::array<std::array<double, 4>, 4> Kmat02;

double stabilization_factor = 0.1; // 安定化係数
bool use_ofem = true;              // OFEMを使用するかどうか

#pragma acc routine seq
double source_function(double t, double permeability, double domain_size)
{
    const double f0 = 1.0e9;               // 中心周波数 [Hz]
    const double t0 = std::sqrt(2.0) / f0; // 中心時間 [s]
    double tau = t - t0;
    // return (2.0 * M_PI * M_PI * f0 * f0 * tau * (3.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
    //         std::exp(-M_PI * M_PI * f0 * f0 * tau * tau)) * // ricker wavelet
    return 4.0 * M_PI * M_PI * f0 * f0 * (1.0 - 4.0 * M_PI * M_PI * f0 * f0 * (t - 1.0 / f0) * (t - 1.0 / f0)) *
           std::exp(-2.0 * M_PI * M_PI * f0 * f0 * (t - 1.0 / f0) * (t - 1.0 / f0)) *
           permeability / domain_size / domain_size / domain_size * 0.001; // ヘルツダイポールのdl = 0.001
};

FemSimulation::FemSimulation(const std::array<int, 3> &grid_size, double domain_size,
                             double time_step, double permittivity, double permeability,
                             int time_frequency)
    : grid_size_x_(grid_size[0]),
      grid_size_y_(grid_size[1]),
      grid_size_z_(grid_size[2]),
      domain_size_(domain_size),
      time_step_(time_step),
      permittivity_(permittivity),
      permeability_(permeability),
      time_frequency_(time_frequency),
      current_time_(0.0),
      time_0_(0),
      time_1_(1),
      time_2_(2)
{
#pragma acc data create(electric_field_0[0 : 12])
    initializeMesh();
    calculateElementStiffnessMatrix();
}

void FemSimulation::initializeMesh()
{
    int idx;
    int idx_element;

    size_t total_size = 3ull * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_ + 1);
    electric_field_x_.resize(total_size, 0.0);
    total_size = 3ull * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1);
    electric_field_y_.resize(total_size, 0.0);
    total_size = 3ull * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_);
    electric_field_z_.resize(total_size, 0.0);
    total_size = static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_) * 4;
    connectivity_x_.resize(total_size, 0);
    connectivity_y_.resize(total_size, 0);
    connectivity_z_.resize(total_size, 0);

    electric_field_x_ptr_ = electric_field_x_.data();
    electric_field_y_ptr_ = electric_field_y_.data();
    electric_field_z_ptr_ = electric_field_z_.data();
    connectivity_x_ptr_ = connectivity_x_.data();
    connectivity_y_ptr_ = connectivity_y_.data();
    connectivity_z_ptr_ = connectivity_z_.data();
    int ENx = electric_field_x_.size();
    int ENy = electric_field_y_.size();
    int ENz = electric_field_z_.size();
    int CNx = connectivity_x_.size();
    int CNy = connectivity_y_.size();
    int CNz = connectivity_z_.size();
#pragma acc data copyin(electric_field_x_ptr_[0 : ENx])
#pragma acc data copyin(electric_field_y_ptr_[0 : ENy])
#pragma acc data copyin(electric_field_z_ptr_[0 : ENz])
#pragma acc data copyin(connectivity_x_ptr_[0 : CNx])
#pragma acc data copyin(connectivity_y_ptr_[0 : CNy])
#pragma acc data copyin(connectivity_z_ptr_[0 : CNz])
#pragma acc data copyin(stabilization_factor)
#pragma acc parallel loop collapse(3) private(idx, idx_element)
    for (int k = 0; k < grid_size_z_; ++k)
    {
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int i = 0; i < grid_size_x_; ++i)
            {
                idx = (i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_) * 4;
                idx_element =
                    i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                connectivity_x_ptr_[idx] = idx_element + k * grid_size_x_;
                connectivity_x_ptr_[idx + 1] = idx_element + (k + 1) * grid_size_x_;
                connectivity_x_ptr_[idx + 2] = idx_element + (k + grid_size_y_ + 1) * grid_size_x_;
                connectivity_x_ptr_[idx + 3] = idx_element + (k + grid_size_y_ + 2) * grid_size_x_;
                connectivity_y_ptr_[idx] = idx_element + j + k * grid_size_y_;
                connectivity_y_ptr_[idx + 1] = idx_element + j + (k + grid_size_x_ + 1) * grid_size_y_;
                connectivity_y_ptr_[idx + 2] = idx_element + j + k * grid_size_y_ + 1;
                connectivity_y_ptr_[idx + 3] = idx_element + j + (k + grid_size_x_ + 1) * grid_size_y_ + 1;
                connectivity_z_ptr_[idx] = idx_element + j + k * (grid_size_x_ + grid_size_y_ + 1);
                connectivity_z_ptr_[idx + 1] = idx_element + j + k * (grid_size_x_ + grid_size_y_ + 1) + 1;
                connectivity_z_ptr_[idx + 2] = idx_element + j + k * (grid_size_x_ + grid_size_y_ + 1) + grid_size_x_ + 1;
                connectivity_z_ptr_[idx + 3] = idx_element + j + k * (grid_size_x_ + grid_size_y_ + 1) + grid_size_x_ + 2;
            }
        }
    }
}

void FemSimulation::calculateElementStiffnessMatrix()
{
    Eigen::MatrixXd Amat_true(12, 12);
    Eigen::MatrixXd Bmat_true(12, 12);
    Eigen::MatrixXd Cmat_true(12, 12);
    if (!use_ofem)
    {
        // 要素剛性行列の計算(従来法)
        std::array<std::array<double, 12>, 12> Amat = {
            {{0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5},
             {0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5},
             {0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5},
             {0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5},
             {-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5},
             {-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5},
             {-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5},
             {-0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 0.5, -0.5},
             {0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
             {0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
             {0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
             {0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0}}};
        Eigen::MatrixXd Amat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Amat_Eigen(i, j) = Amat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Amat = {
            {{4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0},
             {-1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0},
             {-1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0},
             {-2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0},
             {-2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0},
             {-1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0},
             {2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0},
             {1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0},
             {-2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0},
             {2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0},
             {-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0},
             {1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0}}};
        Eigen::MatrixXd _Amat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Amat_Eigen(i, j) = _Amat[i][j];
            }
        }
        Amat_true = Amat_Eigen + stabilization_factor * _Amat_Eigen;
        std::array<std::array<double, 12>, 12> Bmat = {
            {{8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {4.0 / 9.0, 8.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {4.0 / 9.0, 2.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 4.0 / 9.0, 2.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0}}};
        Eigen::MatrixXd Bmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Bmat_Eigen(i, j) = Bmat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Bmat = {
            {{0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0},
             {13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0},
             {-13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0},
             {13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0},
             {-13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0},
             {-13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {-13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, 13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, -13.5 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0}}};
        Eigen::MatrixXd _Bmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Bmat_Eigen(i, j) = _Bmat[i][j];
            }
        }
        Bmat_true = Bmat_Eigen + stabilization_factor * _Bmat_Eigen;
        std::array<std::array<double, 12>, 12> Cmat = {
            {{0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5},
             {0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5},
             {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
             {0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5},
             {0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5},
             {-0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, -0.5, -0.5, -0.5, -0.5},
             {0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5},
             {-0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5},
             {-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
             {-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0},
             {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0},
             {0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0}}};
        Eigen::MatrixXd Cmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Cmat_Eigen(i, j) = Cmat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Cmat = {
            {{4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0},
             {-1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0},
             {-1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0},
             {-2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0},
             {-2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0},
             {-1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0},
             {2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0},
             {1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0},
             {-2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0},
             {2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0},
             {-1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -1.0 / 3.0},
             {1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 4.0 / 3.0}}};
        Eigen::MatrixXd _Cmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Cmat_Eigen(i, j) = _Cmat[i][j];
            }
        }
        Cmat_true = Cmat_Eigen + stabilization_factor * _Cmat_Eigen;
    }

    if (use_ofem)
    {
        // 要素剛性行列の計算(直交不連続基底)
        std::array<std::array<double, 12>, 12> Amat = {
            {{0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0}}};
        Eigen::MatrixXd Amat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Amat_Eigen(i, j) = Amat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Amat = {
            {{0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0},
             {6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0, -3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0},
             {6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0},
             {-3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0, -3.0 / 3.0, 3.0 / 3.0, 3.0 / 3.0, -3.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0}}};
        Eigen::MatrixXd _Amat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Amat_Eigen(i, j) = _Amat[i][j];
            }
        }
        Amat_true = Amat_Eigen + stabilization_factor * _Amat_Eigen;
        std::array<std::array<double, 12>, 12> Bmat = {
            {{72.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 8.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 72.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 8.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 72.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 24.0 / 9.0, 0.0 / 9.0},
             {0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 0.0 / 9.0, 8.0 / 9.0}}};
        Eigen::MatrixXd Bmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Bmat_Eigen(i, j) = Bmat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Bmat = {
            {{0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {-216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, -216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {216.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0},
             {0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0, 0.0 / 27.0}}};
        Eigen::MatrixXd _Bmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Bmat_Eigen(i, j) = _Bmat[i][j];
            }
        }
        Bmat_true = Bmat_Eigen + stabilization_factor * _Bmat_Eigen;
        std::array<std::array<double, 12>, 12> Cmat = {
            {{0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {-6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0},
             {6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0}}};
        Eigen::MatrixXd Cmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                Cmat_Eigen(i, j) = Cmat[i][j];
            }
        }
        std::array<std::array<double, 12>, 12> _Cmat = {
            // {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            //  {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}} // Dは集中質量化不要
            {{0.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0},
             {0.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0},
             {0.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 3.0 / 3.0},
             {0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 3.0 / 3.0},
             {0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, -3.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0, -6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, 3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 6.0 / 3.0, -6.0 / 3.0},
             {0.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, -6.0 / 3.0, 0.0 / 3.0, -3.0 / 3.0, 0.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0, 6.0 / 3.0}} // Dは集中質量化必要
        };
        Eigen::MatrixXd _Cmat_Eigen(12, 12);
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 12; ++j)
            {
                _Cmat_Eigen(i, j) = _Cmat[i][j];
            }
        }
        Cmat_true = Cmat_Eigen + stabilization_factor * _Cmat_Eigen;
    }

    Eigen::MatrixXd Kmat_Eigen(12, 12);
    Kmat_Eigen = 0.5 * time_step_ * time_step_ / permeability_ / permittivity_ / domain_size_ / domain_size_ * Cmat_true * Bmat_true.inverse() * Amat_true;
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            // 出力
            std::cerr << Kmat_Eigen(i, j) / 0.5 / time_step_ / time_step_ * permeability_ * permittivity_ * domain_size_ * domain_size_ * 8.0 << " ";
            element_stiffness_matrix_[i][j] = Kmat_Eigen(i, j);
        }
        std::cerr << std::endl;
    }
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            Kmat00[i][j] = element_stiffness_matrix_[i][j];
            Kmat01[i][j] = element_stiffness_matrix_[i][j + 4];
            Kmat02[i][j] = element_stiffness_matrix_[i][j + 8];
        }
    }
    std::cerr << "----------------------------------------" << std::endl;
    std::cerr << "Element stiffness matrix set." << std::endl;
#pragma acc data copyin(Kmat00, Kmat01, Kmat02, element_stiffness_matrix_)
}

void FemSimulation::setSource_x(const std::array<int, 3> &position)
{
    source_position_x_.push_back(position[0] + position[1] * grid_size_x_ +
                                 position[2] * grid_size_x_ * (grid_size_y_ + 1));
}

void FemSimulation::setSource_y(const std::array<int, 3> &position)
{
    source_position_y_.push_back(position[0] + position[1] * (grid_size_x_ + 1) +
                                 position[2] * (grid_size_x_ + 1) * grid_size_y_);
}

void FemSimulation::setSource_z(const std::array<int, 3> &position)
{
    source_position_z_.push_back(position[0] + position[1] * (grid_size_x_ + 1) +
                                 position[2] * (grid_size_x_ + 1) * (grid_size_y_ + 1));
}

void FemSimulation::setSourceDipole_x(const std::array<int, 3> &position)
{
    source_dipole_position_x_.push_back(position);
}

void FemSimulation::setObservationPoint(
    const std::array<std::vector<std::array<int, 3>>, 3> &position)
{
    observation_points_.push_back(position);
    saved_electric_field_.push_back(std::vector<std::array<double, 3>>());
}

void FemSimulation::updateTimeStep()
{
    ef_x_idx_0_ = time_0_ * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    ef_x_idx_1_ = time_1_ * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    ef_x_idx_2_ = time_2_ * grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1);
    ef_y_idx_0_ = time_0_ * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    ef_y_idx_1_ = time_1_ * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    ef_y_idx_2_ = time_2_ * (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1);
    ef_z_idx_0_ = time_0_ * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;
    ef_z_idx_1_ = time_1_ * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;
    ef_z_idx_2_ = time_2_ * (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_;

#pragma acc parallel loop
    for (int i = ef_x_idx_2_; i < ef_x_idx_2_ + grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1); ++i)
    {
        electric_field_x_ptr_[i] = 0.0;
    }
#pragma acc parallel loop
    for (int i = ef_y_idx_2_; i < ef_y_idx_2_ + (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1); ++i)
    {
        electric_field_y_ptr_[i] = 0.0;
    }
#pragma acc parallel loop
    for (int i = ef_z_idx_2_; i < ef_z_idx_2_ + (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_; ++i)
    {
        electric_field_z_ptr_[i] = 0.0;
    }
    int idx;
    double temp, temp_x, temp_y, temp_z;
    int l, m, n, n_x, n_y, n_z;
    int conn_0, conn_1, conn_2, conn_3;

#pragma acc parallel loop private(idx, electric_field_0, temp, temp_x, temp_y, temp_z, l, m, n, n_x, n_y, n_z) collapse(3)
    for (int k = 0; k < grid_size_z_; ++k)
    {
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int i = 0; i < grid_size_x_; ++i)
            {
                idx = (i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_) * 4;
                electric_field_0 = {
                    electric_field_x_ptr_[ef_x_idx_1_ + connectivity_x_ptr_[idx + 0]],
                    electric_field_x_ptr_[ef_x_idx_1_ + connectivity_x_ptr_[idx + 1]],
                    electric_field_x_ptr_[ef_x_idx_1_ + connectivity_x_ptr_[idx + 2]],
                    electric_field_x_ptr_[ef_x_idx_1_ + connectivity_x_ptr_[idx + 3]],
                    electric_field_y_ptr_[ef_y_idx_1_ + connectivity_y_ptr_[idx + 0]],
                    electric_field_y_ptr_[ef_y_idx_1_ + connectivity_y_ptr_[idx + 1]],
                    electric_field_y_ptr_[ef_y_idx_1_ + connectivity_y_ptr_[idx + 2]],
                    electric_field_y_ptr_[ef_y_idx_1_ + connectivity_y_ptr_[idx + 3]],
                    electric_field_z_ptr_[ef_z_idx_1_ + connectivity_z_ptr_[idx + 0]],
                    electric_field_z_ptr_[ef_z_idx_1_ + connectivity_z_ptr_[idx + 1]],
                    electric_field_z_ptr_[ef_z_idx_1_ + connectivity_z_ptr_[idx + 2]],
                    electric_field_z_ptr_[ef_z_idx_1_ + connectivity_z_ptr_[idx + 3]],
                };

#pragma acc loop seq
                for (l = 0; l < 4; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < 12; ++m)
                    {
                        temp += element_stiffness_matrix_[l][m] * electric_field_0[m];
                    }
                    n = ef_x_idx_2_ + connectivity_x_ptr_[idx + l];
#pragma acc atomic update
                    electric_field_x_ptr_[n] -= temp;
                }

#pragma acc loop seq
                for (l = 4; l < 8; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < 12; ++m)
                    {
                        temp += element_stiffness_matrix_[l][m] * electric_field_0[m];
                    }
                    n = ef_y_idx_2_ + connectivity_y_ptr_[idx + l - 4];
#pragma acc atomic update
                    electric_field_y_ptr_[n] -= temp;
                }

#pragma acc loop seq
                for (l = 8; l < 12; ++l)
                {
                    temp = 0;
#pragma acc loop seq
                    for (m = 0; m < 12; ++m)
                    {
                        temp += element_stiffness_matrix_[l][m] * electric_field_0[m];
                    }
                    n = ef_z_idx_2_ + connectivity_z_ptr_[idx + l - 8];
#pragma acc atomic update
                    electric_field_z_ptr_[n] -= temp;
                }
                // 高速化の可能性？ただし対称性必須
                // #pragma acc loop seq
                //                 for (l = 0; l < 4; ++l)
                //                 {
                //                     temp_x = 0;
                //                     temp_y = 0;
                //                     temp_z = 0;
                //                     n_x = ef_x_idx_2_ + connectivity_x_ptr_[idx + l];
                //                     n_y = ef_y_idx_2_ + connectivity_y_ptr_[idx + l];
                //                     n_z = ef_z_idx_2_ + connectivity_z_ptr_[idx + l];
                // #pragma acc loop seq
                //                     for (m = 0; m < 4; ++m)
                //                     {
                //                         temp_x += Kmat00[l][m] * electric_field_0[m] +
                //                                   Kmat01[l][m] * electric_field_0[m + 4] +
                //                                   Kmat02[l][m] * electric_field_0[m + 8];
                //                         temp_y += Kmat02[l][m] * electric_field_0[m] +
                //                                   Kmat00[l][m] * electric_field_0[m + 4] +
                //                                   Kmat01[l][m] * electric_field_0[m + 8];
                //                         temp_z += Kmat01[l][m] * electric_field_0[m] +
                //                                   Kmat02[l][m] * electric_field_0[m + 4] +
                //                                   Kmat00[l][m] * electric_field_0[m + 8];
                //                     }
                // #pragma acc atomic update
                //                     electric_field_x_ptr_[n_x] -= temp_x;
                // #pragma acc atomic update
                //                     electric_field_y_ptr_[n_y] -= temp_y;
                // #pragma acc atomic update
                //                     electric_field_z_ptr_[n_z] -= temp_z;
                //                 }
            }
        }
    }
#pragma acc parallel loop
    for (int i = 0; i < grid_size_x_ * (grid_size_y_ + 1) * (grid_size_z_ + 1); ++i)
    {
        electric_field_x_ptr_[ef_x_idx_2_ + i] +=
            2.0 * electric_field_x_ptr_[ef_x_idx_1_ + i] - electric_field_x_ptr_[ef_x_idx_0_ + i];
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * grid_size_y_ * (grid_size_z_ + 1); ++i)
    {
        electric_field_y_ptr_[ef_y_idx_2_ + i] +=
            2.0 * electric_field_y_ptr_[ef_y_idx_1_ + i] - electric_field_y_ptr_[ef_y_idx_0_ + i];
    }
#pragma acc parallel loop
    for (int i = 0; i < (grid_size_x_ + 1) * (grid_size_y_ + 1) * grid_size_z_; ++i)
    {
        electric_field_z_ptr_[ef_z_idx_2_ + i] +=
            2.0 * electric_field_z_ptr_[ef_z_idx_1_ + i] - electric_field_z_ptr_[ef_z_idx_0_ + i];
    }

    // 外力項の計算
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_x_.size(); ++i)
    {
        electric_field_x_ptr_[ef_x_idx_2_ + source_position_x_ptr_[i]] +=
            source_function(current_time_, permeability_, domain_size_) * time_step_ *
            time_step_ / permeability_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_y_.size(); ++i)
    {
        electric_field_y_ptr_[ef_y_idx_2_ + source_position_y_ptr_[i]] +=
            source_function(current_time_, permeability_, domain_size_) * time_step_ *
            time_step_ / permeability_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_z_.size(); ++i)
    {
        electric_field_z_ptr_[ef_z_idx_2_ + source_position_z_ptr_[i]] +=
            source_function(current_time_, permeability_, domain_size_) * time_step_ *
            time_step_ / permeability_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
    // 外力項の計算(ヘルツダイポール)
    if (!use_ofem)
    {
#pragma acc parallel loop private(conn_0, conn_1, conn_2, conn_3, temp)
        for (size_t i = 0; i < source_dipole_position_x_.size(); ++i)
        {
            conn_0 = (source_dipole_position_x_[i][0] +
                      source_dipole_position_x_[i][1] * grid_size_x_ +
                      source_dipole_position_x_[i][2] * grid_size_x_ * grid_size_y_) *
                     4;
            conn_1 = (source_dipole_position_x_[i][0] +
                      (source_dipole_position_x_[i][1] + 1) * grid_size_x_ +
                      source_dipole_position_x_[i][2] * grid_size_x_ * grid_size_y_) *
                     4;
            conn_2 = (source_dipole_position_x_[i][0] +
                      source_dipole_position_x_[i][1] * grid_size_x_ +
                      (source_dipole_position_x_[i][2] + 1) * grid_size_x_ * grid_size_y_) *
                     4;
            conn_3 = (source_dipole_position_x_[i][0] +
                      (source_dipole_position_x_[i][1] + 1) * grid_size_x_ +
                      (source_dipole_position_x_[i][2] + 1) * grid_size_x_ * grid_size_y_) *
                     4;
            temp = source_function(current_time_, permeability_, domain_size_) * time_step_ * time_step_ /
                   permeability_ / permittivity_ / 8.0;
            // 従来法
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_0]] +=
                1.0 / 8.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_0 + 1]] +=
                3.0 / 4.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_0 + 2]] +=
                3.0 / 4.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_0 + 3]] +=
                9.0 / 2.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_1 + 1]] +=
                1.0 / 8.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_1 + 3]] +=
                3.0 / 4.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_2 + 2]] +=
                1.0 / 8.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_2 + 3]] +=
                3.0 / 4.0 * temp;
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_3 + 3]] +=
                1.0 / 8.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_0]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_0 + 2]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_1]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_1 + 2]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_2 + 1]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_2 + 3]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_3 + 1]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_y_ptr_[ef_y_idx_2_ + connectivity_y_ptr_[conn_3 + 3]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_0]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_0 + 1]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_1 + 2]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_1 + 3]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_2]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_2 + 1]] -=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_3 + 2]] +=
                stabilization_factor / 2.0 * temp;
            electric_field_z_ptr_[ef_z_idx_2_ + connectivity_z_ptr_[conn_3 + 3]] +=
                stabilization_factor / 2.0 * temp;
        }
    }
    if (use_ofem)
    {
#pragma acc parallel loop private(conn_0, conn_1, conn_2, conn_3, temp)
        for (size_t i = 0; i < source_dipole_position_x_.size(); ++i)
        {
            conn_0 = (source_dipole_position_x_[i][0] +
                      source_dipole_position_x_[i][1] * grid_size_x_ +
                      source_dipole_position_x_[i][2] * grid_size_x_ * grid_size_y_) *
                     4;
            conn_1 = (source_dipole_position_x_[i][0] +
                      (source_dipole_position_x_[i][1] + 1) * grid_size_x_ +
                      source_dipole_position_x_[i][2] * grid_size_x_ * grid_size_y_) *
                     4;
            conn_2 = (source_dipole_position_x_[i][0] +
                      source_dipole_position_x_[i][1] * grid_size_x_ +
                      (source_dipole_position_x_[i][2] + 1) * grid_size_x_ * grid_size_y_) *
                     4;
            conn_3 = (source_dipole_position_x_[i][0] +
                      (source_dipole_position_x_[i][1] + 1) * grid_size_x_ +
                      (source_dipole_position_x_[i][2] + 1) * grid_size_x_ * grid_size_y_) *
                     4;
            temp = source_function(current_time_, permeability_, domain_size_) * time_step_ * time_step_ /
                   permeability_ / permittivity_ / 8.0;
            // 直交不連続基底
            electric_field_x_ptr_[ef_x_idx_2_ + connectivity_x_ptr_[conn_0 + 3]] +=
                8.0 * temp;
        }
    }
}

void FemSimulation::applyBoundaryConditions()
{
    // 完全導体境界条件（電場の接線成分を0に）
    int idx_0, idx_1;
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            idx_0 = i + j * grid_size_x_;
            idx_1 = i + j * grid_size_x_ + grid_size_x_ * (grid_size_y_ + 1) * grid_size_z_;
            electric_field_x_ptr_[ef_x_idx_2_ + idx_0] = 0.0;
            electric_field_x_ptr_[ef_x_idx_2_ + idx_1] = 0.0;
        }
    }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int i = 0; i < grid_size_x_; ++i)
    {
        for (int k = 0; k < grid_size_z_ + 1; ++k)
        {
            idx_0 = i + k * grid_size_x_ * (grid_size_y_ + 1);
            idx_1 = i + k * grid_size_x_ * (grid_size_y_ + 1) + grid_size_x_ * grid_size_y_;
            electric_field_x_ptr_[ef_x_idx_2_ + idx_0] = 0.0;
            electric_field_x_ptr_[ef_x_idx_2_ + idx_1] = 0.0;
        }
    }

#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int j = 0; j < grid_size_y_; ++j)
    {
        for (int k = 0; k < grid_size_z_ + 1; ++k)
        {
            idx_0 = j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_;
            idx_1 =
                j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * grid_size_y_ + grid_size_x_;
            electric_field_y_ptr_[ef_y_idx_2_ + idx_0] = 0.0;
            electric_field_y_ptr_[ef_y_idx_2_ + idx_1] = 0.0;
        }
    }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int j = 0; j < grid_size_y_; ++j)
    {
        for (int i = 0; i < grid_size_x_ + 1; ++i)
        {
            idx_0 = i + j * (grid_size_x_ + 1);
            idx_1 = i + j * (grid_size_x_ + 1) + (grid_size_x_ + 1) * grid_size_y_ * grid_size_z_;
            electric_field_y_ptr_[ef_y_idx_2_ + idx_0] = 0.0;
            electric_field_y_ptr_[ef_y_idx_2_ + idx_1] = 0.0;
        }
    }

#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int k = 0; k < grid_size_z_; ++k)
    {
        for (int i = 0; i < grid_size_x_ + 1; ++i)
        {
            idx_0 = i + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
            idx_1 = i + k * (grid_size_x_ + 1) * (grid_size_y_ + 1) + (grid_size_x_ + 1) * grid_size_y_;
            electric_field_z_ptr_[ef_z_idx_2_ + idx_0] = 0.0;
            electric_field_z_ptr_[ef_z_idx_2_ + idx_1] = 0.0;
        }
    }
#pragma acc parallel loop collapse(2) private(idx_0, idx_1)
    for (int k = 0; k < grid_size_z_; ++k)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            idx_0 = j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1);
            idx_1 =
                j * (grid_size_x_ + 1) + k * (grid_size_x_ + 1) * (grid_size_y_ + 1) + grid_size_x_;
            electric_field_z_ptr_[ef_z_idx_2_ + idx_0] = 0.0;
            electric_field_z_ptr_[ef_z_idx_2_ + idx_1] = 0.0;
        }
    }
    //     // 直接励起
    // #pragma acc parallel loop
    //     for (size_t i = 0; i < source_position_x_.size(); ++i)
    //     {
    //         electric_field_x_ptr_[ef_x_idx_2_ + source_position_x_ptr_[i]] = source_function(current_time_);
    //     }
    // #pragma acc parallel loop
    //     for (size_t i = 0; i < source_position_y_.size(); ++i)
    //     {
    //         electric_field_y_ptr_[ef_y_idx_2_ + source_position_y_ptr_[i]] = source_function(current_time_);
    //     }
    // #pragma acc parallel loop
    //     for (size_t i = 0; i < source_position_z_.size(); ++i)
    //     {
    //         electric_field_z_ptr_[ef_z_idx_2_ + source_position_z_ptr_[i]] = source_function(current_time_);
    //     }
}

void FemSimulation::run(int num_steps)
{
    saved_electric_field_.resize(observation_points_.size());
    for (auto &point : saved_electric_field_)
    {
        point.resize(static_cast<int>(std::round(num_steps / time_frequency_)) + 1);
    }
    source_position_x_ptr_ = source_position_x_.data();
    source_position_y_ptr_ = source_position_y_.data();
    source_position_z_ptr_ = source_position_z_.data();
    source_dipole_position_x_ptr_ = source_dipole_position_x_.data();
    source_dipole_position_y_ptr_ = source_dipole_position_y_.data();
    source_dipole_position_z_ptr_ = source_dipole_position_z_.data();
#pragma acc data copyin(source_position_x_ptr_[0 : source_position_x_.size()])
#pragma acc data copyin(source_position_y_ptr_[0 : source_position_y_.size()])
#pragma acc data copyin(source_position_z_ptr_[0 : source_position_z_.size()])
#pragma acc data copyin(source_dipole_position_x_ptr_[0 : source_dipole_position_x_.size()])
#pragma acc data copyin(source_dipole_position_y_ptr_[0 : source_dipole_position_y_.size()])
#pragma acc data copyin(source_dipole_position_z_ptr_[0 : source_dipole_position_z_.size()])

    for (int step = 1; step < num_steps + 1; ++step)
    {
        std::cerr << "step: " << step << std::endl;
        updateTimeStep();
        applyBoundaryConditions();

        // 観測点での値を保存
        if (step % time_frequency_ == 0)
        {
            for (size_t i = 0; i < observation_points_.size(); ++i)
            {
                for (size_t j = 0; j < observation_points_[i][0].size(); ++j)
                {
                    int idx_x =
                        observation_points_[i][0][j][0] +
                        observation_points_[i][0][j][1] * grid_size_x_ +
                        observation_points_[i][0][j][2] * grid_size_x_ * (grid_size_y_ + 1);
#pragma acc update host(electric_field_x_ptr_[ef_x_idx_2_ + idx_x : ef_x_idx_2_ + idx_x + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [0] += electric_field_x_ptr_[ef_x_idx_2_ + idx_x];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][0] /=
                    observation_points_[i][0].size();
                for (size_t j = 0; j < observation_points_[i][1].size(); ++j)
                {
                    int idx_y =
                        observation_points_[i][1][j][0] +
                        observation_points_[i][1][j][1] * (grid_size_x_ + 1) +
                        observation_points_[i][1][j][2] * (grid_size_x_ + 1) * grid_size_y_;
#pragma acc update host(electric_field_y_ptr_[ef_y_idx_2_ + idx_y : ef_y_idx_2_ + idx_y + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [1] += electric_field_y_ptr_[ef_y_idx_2_ + idx_y];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][1] /=
                    observation_points_[i][1].size();
                for (size_t j = 0; j < observation_points_[i][2].size(); ++j)
                {
                    int idx_z =
                        observation_points_[i][2][j][0] +
                        observation_points_[i][2][j][1] * (grid_size_x_ + 1) +
                        observation_points_[i][2][j][2] * (grid_size_x_ + 1) * (grid_size_y_ + 1);
#pragma acc update host(electric_field_z_ptr_[ef_z_idx_2_ + idx_z : ef_z_idx_2_ + idx_z + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [2] += electric_field_z_ptr_[ef_z_idx_2_ + idx_z];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][2] /=
                    observation_points_[i][2].size();
            }
        }
        current_time_ += time_step_;
        time_0_ = (time_0_ + 1) % 3;
        time_1_ = (time_1_ + 1) % 3;
        time_2_ = (time_2_ + 1) % 3;
    }
}

void FemSimulation::saveResults(int num_steps)
{
    for (size_t i = 0; i < saved_electric_field_.size(); ++i)
    {
        std::string filename_i = "results_" + std::to_string(i) + "_" + std::to_string(time_step_) + "_" + std::to_string(domain_size_) + "_" +
                                 std::to_string(num_steps) + ".csv";
        std::ofstream file(filename_i);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename_i << std::endl;
            return;
        }

        // 結果をCSV形式で保存
        file << "time,Ex,Ey,Ez" << std::endl;
        for (size_t j = 0; j < saved_electric_field_[i].size(); ++j)
        {
            file << j * time_frequency_ * time_step_ << "," << saved_electric_field_[i][j][0] << ","
                 << saved_electric_field_[i][j][1] << "," << saved_electric_field_[i][j][2]
                 << std::endl;
        }
        file.close();
    }
}

bool check_params(const std::array<double, 3> &domain_sizes, const std::array<int, 3> &grid_num, double duration,
                  double time_step, double domain_size, double c)
{
    if (domain_sizes[0] <= 0 || domain_sizes[1] <= 0 || domain_sizes[2] <= 0)
    {
        return false;
    }
    if (grid_num[0] <= 0 || grid_num[1] <= 0 || grid_num[2] <= 0)
    {
        return false;
    }
    if (duration <= 0)
    {
        return false;
    }
    if (time_step <= 0)
    {
        return false;
    }
    if (std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[1] / grid_num[1]) ||
        std::round(domain_sizes[0] / grid_num[0]) != std::round(domain_sizes[2] / grid_num[2]))
    {
        std::cerr << "Invalid grid size" << std::endl;
        return false;
    }
    if (c * time_step >= domain_size)
    {
        std::cerr << "Invalid time step" << std::endl;
        std::cerr << "c * time_step: " << c * time_step << std::endl;
        std::cerr << "domain_size: " << domain_size << std::endl;
        return false;
    }
    return true;
};