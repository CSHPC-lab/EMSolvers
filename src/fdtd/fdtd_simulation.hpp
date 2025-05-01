#pragma once
#include <vector>
#include <string>
#include <iostream>

constexpr double EPSILON_0 = 8.854187817e-12;
constexpr double MU_0 = 1.2566370614e-6;

struct Parameters {
    double dx, dy, dz;
    int nx, ny, nz;
    double dt;
    int nt;

    int src_i, src_j, src_k;
    int obs_i, obs_j, obs_k;

    double f_peak; // ピーク周波数（Hz）
    double t0;     // 中心時間（秒）

    double rel_eps; // 比誘電率
    double rel_mu;  // 比透磁率
};

class FDTD {
public:
    FDTD(const Parameters& params);
    void run();
private:
    Parameters p;
    std::vector<std::vector<std::vector<double>>> Ex, Ey, Ez;
    std::vector<std::vector<std::vector<double>>> Hx, Hy, Hz;
    std::vector<double> observation;

    void initialize_fields();
    void update_H();
    void update_E();
    void apply_source(int t);
    void record_observation();
    void output_observation_csv(const std::string& filename);
    void check_cfl_condition();
};