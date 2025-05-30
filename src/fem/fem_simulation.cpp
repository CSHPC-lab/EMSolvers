#include "fem_simulation.hpp"

#include <openacc.h>

#include <cmath>
#include <fstream>
#include <iostream>

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
std::array<double, 12> electric_field_0;

#pragma acc routine seq
double source_function(double t)
{
    const double f0 = 1.0; // 中心周波数 [Hz]
    const double t0 = 2.0; // 時間シフト
    double tau = t - t0;
    return (1.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
           std::exp(-M_PI * M_PI * f0 * f0 * tau *
                    tau);
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
    element_stiffness_matrix_ = std::array<std::array<double, 12>, 12>{
        {{2, 0, 0, -2, -1, -1, 1, 1, -1, 1, -1, 1},
         {0, 2, -2, 0, 1, 1, -1, -1, -1, 1, -1, 1},
         {0, -2, 2, 0, -1, -1, 1, 1, 1, -1, 1, -1},
         {-2, 0, 0, 2, 1, 1, -1, -1, 1, -1, 1, -1},
         {-1, 1, -1, 1, 2, 0, 0, -2, -1, -1, 1, 1},
         {-1, 1, -1, 1, 0, 2, -2, 0, 1, 1, -1, -1},
         {1, -1, 1, -1, 0, -2, 2, 0, -1, -1, 1, 1},
         {1, -1, 1, -1, -2, 0, 0, 2, 1, 1, -1, -1},
         {-1, -1, 1, 1, -1, 1, -1, 1, 2, 0, 0, -2},
         {1, 1, -1, -1, -1, 1, -1, 1, 0, 2, -2, 0},
         {-1, -1, 1, 1, 1, -1, 1, -1, 0, -2, 2, 0},
         {1, 1, -1, -1, 1, -1, 1, -1, -2, 0, 0, 2}}}; // 2をかけた値
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 12; j++)
        {
            element_stiffness_matrix_[i][j] = element_stiffness_matrix_[i][j] / 2.0 * time_step_ *
                                              time_step_ / 2.0 / permeability_ / permittivity_ / domain_size_ / domain_size_;
        }
    }
#pragma acc data copyin(element_stiffness_matrix_[0 : 12][0 : 12])
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
    double temp;
    int l, m, n;

#pragma acc parallel loop private(idx, electric_field_0, temp, l, m, n) collapse(3)
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

#pragma acc parallel loop
    for (size_t i = 0; i < source_position_x_.size(); ++i)
    {
        electric_field_x_ptr_[ef_x_idx_2_ + source_position_x_ptr_[i]] -=
            source_function(current_time_) * time_step_ *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_y_.size(); ++i)
    {
        electric_field_y_ptr_[ef_y_idx_2_ + source_position_y_ptr_[i]] -=
            source_function(current_time_) * time_step_ *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_z_.size(); ++i)
    {
        electric_field_z_ptr_[ef_z_idx_2_ + source_position_z_ptr_[i]] -=
            source_function(current_time_) * time_step_ *
            time_step_ / permittivity_ / domain_size_ / domain_size_ / domain_size_;
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
            idx_0 = j * grid_size_x_ + k * (grid_size_x_ + 1) * grid_size_y_;
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
#pragma acc data copyin(source_position_x_ptr_[0 : source_position_x_.size()])
#pragma acc data copyin(source_position_y_ptr_[0 : source_position_y_.size()])
#pragma acc data copyin(source_position_z_ptr_[0 : source_position_z_.size()])

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
            current_time_ += time_step_;
            time_0_ = (time_0_ + 1) % 3;
            time_1_ = (time_1_ + 1) % 3;
            time_2_ = (time_2_ + 1) % 3;
        }
    }
}

void FemSimulation::saveResults(double observe_r, std::function<double(double, double, double, double)> analysis_function,
                                int num_steps, double c)
{
    for (size_t i = 0; i < saved_electric_field_.size(); ++i)
    {
        std::string filename_i = "results_" + std::to_string(i) + std::to_string(time_step_) + std::to_string(domain_size_) +
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
    std::string filename_r = "solution" + std::to_string(time_step_) + std::to_string(domain_size_) +
                             std::to_string(num_steps) + ".csv";
    std::ofstream file_r(filename_r);
    if (!file_r.is_open())
    {
        std::cerr << "Error: Could not open file " << filename_r << std::endl;
        return;
    }
    file_r << "time,Ex,Ey,Ez" << std::endl;
    for (int i = 0; i < num_steps + 1; i += time_frequency_)
    {
        file_r << i * time_frequency_ * time_step_ << ","
               << analysis_function(i * time_frequency_ * time_step_, observe_r, c, permeability_)
               << "," << 0 << "," << 0 << std::endl;
    }
    file_r.close();
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