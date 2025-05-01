#include "fdtd_simulation.hpp"
#include <fstream>
#include <cmath>

FDTD::FDTD(const Parameters& params) : p(params) {
    check_cfl_condition();
    initialize_fields();
}

void FDTD::check_cfl_condition() {
    double c = 1.0 / std::sqrt(EPSILON_0 * MU_0);
    double cfl_limit = 1.0 / (c * std::sqrt(1.0 / (p.dx * p.dx) + 1.0 / (p.dy * p.dy) + 1.0 / (p.dz * p.dz)));
    if (p.dt > cfl_limit) {
        std::cerr << "[WARNING] Time step dt = " << p.dt << " exceeds CFL limit = " << cfl_limit << "\n";
        std::cerr << "Simulation may be unstable! Consider reducing dt." << std::endl;
    }
}

void FDTD::initialize_fields() {
    Ex = Ey = Ez = std::vector(p.nx+2, std::vector(p.ny+2, std::vector<double>(p.nz+2, 0.0)));
    Hx = Hy = Hz = std::vector(p.nx+2, std::vector(p.ny+2, std::vector<double>(p.nz+2, 0.0)));
    observation.reserve(p.nt);
}

void FDTD::update_H() {
    double mu = MU_0 * p.rel_mu;
    for (int i = 1; i < p.nx + 1; ++i) {
        for (int j = 1; j < p.ny + 1; ++j) {
            for (int k = 1; k < p.nz + 1; ++k) {
                Hx[i][j][k] -= (p.dt / mu) * (
                    (Ez[i][j+1][k] - Ez[i][j][k]) / p.dy -
                    (Ey[i][j][k+1] - Ey[i][j][k]) / p.dz);
                Hy[i][j][k] -= (p.dt / mu) * (
                    (Ex[i][j][k+1] - Ex[i][j][k]) / p.dz -
                    (Ez[i+1][j][k] - Ez[i][j][k]) / p.dx);
                Hz[i][j][k] -= (p.dt / mu) * (
                    (Ey[i+1][j][k] - Ey[i][j][k]) / p.dx -
                    (Ex[i][j+1][k] - Ex[i][j][k]) / p.dy);
            }
        }
    }
}

void FDTD::update_E() {
    double epsilon = EPSILON_0 * p.rel_eps;
    for (int i = 1; i < p.nx+1; ++i) {
        for (int j = 1; j < p.ny+1; ++j) {
            for (int k = 1; k < p.nz+1; ++k) {
                Ex[i][j][k] += (p.dt / epsilon) * (
                    (Hz[i][j][k] - Hz[i][j-1][k]) / p.dy -
                    (Hy[i][j][k] - Hy[i][j][k-1]) / p.dz);
                Ey[i][j][k] += (p.dt / epsilon) * (
                    (Hx[i][j][k] - Hx[i][j][k-1]) / p.dz -
                    (Hz[i][j][k] - Hz[i-1][j][k]) / p.dx);
                Ez[i][j][k] += (p.dt / epsilon) * (
                    (Hy[i][j][k] - Hy[i-1][j][k]) / p.dx -
                    (Hx[i][j][k] - Hx[i][j-1][k]) / p.dy);
            }
        }
    }
}

void FDTD::apply_source(int t) {
    double t_real = t * p.dt;
    double arg = M_PI * p.f_peak * (t_real - p.t0);
    double source = (1 - 2 * arg * arg) * std::exp(-arg * arg); // リッカー波形
    Ez[p.src_i+1][p.src_j+1][p.src_k+1] += source;
}

void FDTD::record_observation() {
    observation.push_back(Ez[p.obs_i+1][p.obs_j+1][p.obs_k+1]);
}

void FDTD::output_observation_csv(const std::string& filename) {
    std::ofstream file(filename);
    file << "t,Ez\n";
    for (int t = 0; t < observation.size(); ++t) {
        file << t * p.dt << "," << observation[t] << "\n";
    }
}

void FDTD::run() {
    for (int t = 0; t < p.nt; ++t) {
        update_H();
        update_E();
        apply_source(t);
        record_observation();
    }
    output_observation_csv("observation.csv");
}
