#include "fdtd_simulation.hpp"

#include <openacc.h>

#include <cmath>
#include <fstream>
#include <iostream>

// vectorのポインタ
double *electric_field_x_ptr_;
double *electric_field_y_ptr_;
double *electric_field_z_ptr_;
double *magnetic_field_x_ptr_;
double *magnetic_field_y_ptr_;
double *magnetic_field_z_ptr_;
int *connectivity_x_ptr_;
int *connectivity_y_ptr_;
int *connectivity_z_ptr_;
int *source_position_x_ptr_;
int *source_position_y_ptr_;
int *source_position_z_ptr_;
std::array<double, 12> electric_field_0;

#pragma acc routine seq
double source_function(double t, double domain_size)
{
    const double f0 = 1.0e9;               // 中心周波数 [Hz]
    const double t0 = std::sqrt(2.0) / f0; // 中心時間 [s]
    double tau = t - t0;
    // return (1.0 - 2.0 * M_PI * M_PI * f0 * f0 * tau * tau) *
    //        std::exp(-M_PI * M_PI * f0 * f0 * tau *
    //                 tau) / // ricker wavelet
    return -4.0 * M_PI * M_PI * f0 * f0 * (t - 1.0 / f0) *
           std::exp(-2.0 * M_PI * M_PI * f0 * f0 * (t - 1.0 / f0) * (t - 1.0 / f0)) /
           domain_size / domain_size / domain_size * 0.001; // ヘルツダイポールのdl = 0.001
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
      current_time_(0.0)
{
#pragma acc data create(electric_field_0[0 : 12])
    initializeMesh();
}

void FemSimulation::initializeMesh()
{
    size_t total_size = 1ull * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_ + 2) * static_cast<size_t>(grid_size_z_ + 2);
    electric_field_x_.resize(total_size, 0.0);
    total_size = 1ull * static_cast<size_t>(grid_size_x_ + 2) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_ + 2);
    electric_field_y_.resize(total_size, 0.0);
    total_size = 1ull * static_cast<size_t>(grid_size_x_ + 2) * static_cast<size_t>(grid_size_y_ + 2) * static_cast<size_t>(grid_size_z_ + 1);
    electric_field_z_.resize(total_size, 0.0);
    total_size = 1ull * static_cast<size_t>(grid_size_x_) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_ + 1);
    magnetic_field_x_.resize(total_size, 0.0);
    total_size = 1ull * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_) * static_cast<size_t>(grid_size_z_ + 1);
    magnetic_field_y_.resize(total_size, 0.0);
    total_size = 1ull * static_cast<size_t>(grid_size_x_ + 1) * static_cast<size_t>(grid_size_y_ + 1) * static_cast<size_t>(grid_size_z_);
    magnetic_field_z_.resize(total_size, 0.0);

    electric_field_x_ptr_ = electric_field_x_.data();
    electric_field_y_ptr_ = electric_field_y_.data();
    electric_field_z_ptr_ = electric_field_z_.data();
    magnetic_field_x_ptr_ = magnetic_field_x_.data();
    magnetic_field_y_ptr_ = magnetic_field_y_.data();
    magnetic_field_z_ptr_ = magnetic_field_z_.data();
    int ENx = electric_field_x_.size();
    int ENy = electric_field_y_.size();
    int ENz = electric_field_z_.size();
    int EMx = magnetic_field_x_.size();
    int EMy = magnetic_field_y_.size();
    int EMz = magnetic_field_z_.size();
#pragma acc data copyin(electric_field_x_ptr_[0 : ENx])
#pragma acc data copyin(electric_field_y_ptr_[0 : ENy])
#pragma acc data copyin(electric_field_z_ptr_[0 : ENz])
#pragma acc data copyin(magnetic_field_x_ptr_[0 : EMx])
#pragma acc data copyin(magnetic_field_y_ptr_[0 : EMy])
#pragma acc data copyin(magnetic_field_z_ptr_[0 : EMz])
}

void FemSimulation::setSource_x(const std::array<int, 3> &position)
{
    source_position_x_.push_back(position[0] + position[1] * (grid_size_x_ + 1) +
                                 position[2] * (grid_size_x_ + 1) * (grid_size_y_ + 2));
}

void FemSimulation::setSource_y(const std::array<int, 3> &position)
{
    source_position_y_.push_back(position[0] + position[1] * (grid_size_x_ + 2) +
                                 position[2] * (grid_size_x_ + 2) * (grid_size_y_ + 1));
}

void FemSimulation::setSource_z(const std::array<int, 3> &position)
{
    source_position_z_.push_back(position[0] + position[1] * (grid_size_x_ + 2) +
                                 position[2] * (grid_size_x_ + 2) * (grid_size_y_ + 2));
}

void FemSimulation::setObservationPoint(
    const std::array<std::vector<std::array<int, 3>>, 3> &position)
{
    observation_points_.push_back(position);
    saved_electric_field_.push_back(std::vector<std::array<double, 3>>());
}

void FemSimulation::updateTimeStep()
{
    int idx;

#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 0; k < grid_size_z_ + 1; ++k)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            for (int i = 0; i < grid_size_x_; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                magnetic_field_x_ptr_[idx + k * grid_size_x_] -=
                    time_step_ / permeability_ / domain_size_ *
                    (electric_field_z_ptr_[idx + 2 * j + k * (2 * (grid_size_x_ + grid_size_y_) + 4) + 1 + grid_size_x_ + 2] -
                     electric_field_z_ptr_[idx + 2 * j + k * (2 * (grid_size_x_ + grid_size_y_) + 4) + 1] -
                     electric_field_y_ptr_[idx + 2 * j + k * (grid_size_x_ + 2 * grid_size_y_ + 2) + 1 + (grid_size_x_ + 2) * (grid_size_y_ + 1)] +
                     electric_field_y_ptr_[idx + 2 * j + k * (grid_size_x_ + 2 * grid_size_y_ + 2) + 1]);
            }
        }
    }
#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 0; k < grid_size_z_ + 1; ++k)
    {
        for (int j = 0; j < grid_size_y_; ++j)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                magnetic_field_y_ptr_[idx + j + k * grid_size_x_] -=
                    time_step_ / permeability_ / domain_size_ *
                    (electric_field_x_ptr_[idx + j + k * (2 * grid_size_x_ + grid_size_y_ + 2) + grid_size_x_ + 1 + (grid_size_x_ + 1) * (grid_size_y_ + 2)] -
                     electric_field_x_ptr_[idx + j + k * (2 * grid_size_x_ + grid_size_y_ + 2) + grid_size_x_ + 1] -
                     electric_field_z_ptr_[idx + 2 * j + k * (2 * (grid_size_x_ + grid_size_y_) + 4) + 1 + grid_size_x_ + 2] +
                     electric_field_z_ptr_[idx + 2 * j + k * (2 * (grid_size_x_ + grid_size_y_) + 4) + grid_size_x_ + 2]);
            }
        }
    }
#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 0; k < grid_size_z_; ++k)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                magnetic_field_z_ptr_[idx + j + k * (grid_size_x_ + grid_size_y_ + 1)] -=
                    time_step_ / permeability_ / domain_size_ *
                    (electric_field_y_ptr_[idx + 2 * j + k * (grid_size_x_ + 2 * grid_size_y_ + 2) + 1 + (grid_size_x_ + 2) * (grid_size_y_ + 1)] -
                     electric_field_y_ptr_[idx + 2 * j + k * (grid_size_x_ + 2 * grid_size_y_ + 2) + (grid_size_x_ + 2) * (grid_size_y_ + 1)] -
                     electric_field_x_ptr_[idx + j + k * (2 * grid_size_x_ + grid_size_y_ + 2) + grid_size_x_ + 1 + (grid_size_x_ + 1) * (grid_size_y_ + 2)] +
                     electric_field_x_ptr_[idx + j + k * (2 * grid_size_x_ + grid_size_y_ + 2) + (grid_size_x_ + 1) * (grid_size_y_ + 2)]);
            }
        }
    }
#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 1; k < grid_size_z_ + 1; ++k)
    {
        for (int j = 1; j < grid_size_y_ + 1; ++j)
        {
            for (int i = 0; i < grid_size_x_ + 1; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                electric_field_x_ptr_[idx + j + k * (2 * grid_size_x_ + grid_size_y_ + 2)] +=
                    time_step_ / permittivity_ / domain_size_ *
                    (magnetic_field_z_ptr_[idx - grid_size_x_ * grid_size_y_ + j + (k - 1) * (grid_size_x_ + grid_size_y_ + 1)] -
                     magnetic_field_z_ptr_[idx - grid_size_x_ - grid_size_x_ * grid_size_y_ + j - 1 + (k - 1) * (grid_size_x_ + grid_size_y_ + 1)] -
                     magnetic_field_y_ptr_[idx - grid_size_x_ + j - 1 + k * grid_size_y_] +
                     magnetic_field_y_ptr_[idx - grid_size_x_ - grid_size_x_ * grid_size_y_ + j - 1 + (k - 1) * grid_size_y_]);
            }
        }
    }
#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 1; k < grid_size_z_ + 1; ++k)
    {
        for (int j = 0; j < grid_size_y_ + 1; ++j)
        {
            for (int i = 1; i < grid_size_x_ + 1; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                electric_field_y_ptr_[idx + 2 * j + k * (grid_size_x_ + 2 * grid_size_y_ + 2)] +=
                    time_step_ / permittivity_ / domain_size_ *
                    (magnetic_field_x_ptr_[idx - 1 + k * grid_size_x_] -
                     magnetic_field_x_ptr_[idx - 1 - grid_size_x_ * grid_size_y_ + (k - 1) * grid_size_x_] -
                     magnetic_field_z_ptr_[idx - grid_size_x_ * grid_size_y_ + j + (k - 1) * (grid_size_x_ + grid_size_y_ + 1)] +
                     magnetic_field_z_ptr_[idx - 1 - grid_size_x_ * grid_size_y_ + j + (k - 1) * (grid_size_x_ + grid_size_y_ + 1)]);
            }
        }
    }
#pragma acc parallel loop private(idx) collapse(3)
    for (int k = 0; k < grid_size_z_ + 1; ++k)
    {
        for (int j = 1; j < grid_size_y_ + 1; ++j)
        {
            for (int i = 1; i < grid_size_x_ + 1; ++i)
            {
                idx = i + j * grid_size_x_ + k * grid_size_x_ * grid_size_y_;
                electric_field_z_ptr_[idx + 2 * j + k * (2 * (grid_size_x_ + grid_size_y_) + 4)] +=
                    time_step_ / permittivity_ / domain_size_ *
                    (magnetic_field_y_ptr_[idx - grid_size_x_ + j - 1 + k * grid_size_y_] -
                     magnetic_field_y_ptr_[idx - 1 - grid_size_x_ + j - 1 + k * grid_size_y_] -
                     magnetic_field_x_ptr_[idx - 1 + k * grid_size_x_] +
                     magnetic_field_x_ptr_[idx - 1 - grid_size_x_ + k * grid_size_x_]);
            }
        }
    }

#pragma acc parallel loop
    for (size_t i = 0; i < source_position_x_.size(); ++i)
    {
        electric_field_x_ptr_[source_position_x_ptr_[i]] -=
            source_function(current_time_, domain_size_) * time_step_ / permittivity_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_y_.size(); ++i)
    {
        electric_field_y_ptr_[source_position_y_ptr_[i]] -=
            source_function(current_time_, domain_size_) * time_step_ / permittivity_;
    }
#pragma acc parallel loop
    for (size_t i = 0; i < source_position_z_.size(); ++i)
    {
        electric_field_z_ptr_[source_position_z_ptr_[i]] -=
            source_function(current_time_, domain_size_) * time_step_ / permittivity_;
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

        // 観測点での値を保存
        if (step % time_frequency_ == 0)
        {
            for (size_t i = 0; i < observation_points_.size(); ++i)
            {
                for (size_t j = 0; j < observation_points_[i][0].size(); ++j)
                {
                    int idx_x =
                        observation_points_[i][0][j][0] +
                        observation_points_[i][0][j][1] * (grid_size_x_ + 1) +
                        observation_points_[i][0][j][2] * (grid_size_x_ + 1) * (grid_size_y_ + 2);
#pragma acc update host(electric_field_x_ptr_[idx_x : idx_x + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [0] += electric_field_x_ptr_[idx_x];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][0] /=
                    observation_points_[i][0].size();
                for (size_t j = 0; j < observation_points_[i][1].size(); ++j)
                {
                    int idx_y =
                        observation_points_[i][1][j][0] +
                        observation_points_[i][1][j][1] * (grid_size_x_ + 2) +
                        observation_points_[i][1][j][2] * (grid_size_x_ + 2) * (grid_size_y_ + 1);
#pragma acc update host(electric_field_y_ptr_[idx_y : idx_y + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [1] += electric_field_y_ptr_[idx_y];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][1] /=
                    observation_points_[i][1].size();
                for (size_t j = 0; j < observation_points_[i][2].size(); ++j)
                {
                    int idx_z =
                        observation_points_[i][2][j][0] +
                        observation_points_[i][2][j][1] * (grid_size_x_ + 2) +
                        observation_points_[i][2][j][2] * (grid_size_x_ + 2) * (grid_size_y_ + 2);
#pragma acc update host(electric_field_z_ptr_[idx_z : idx_z + 1])
                    saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))]
                                         [2] += electric_field_z_ptr_[idx_z];
                }
                saved_electric_field_[i][static_cast<int>(std::round(step / time_frequency_))][2] /=
                    observation_points_[i][2].size();
            }
        }
        current_time_ += time_step_;
    }
}

void FemSimulation::saveResults(int num_steps, double c)
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