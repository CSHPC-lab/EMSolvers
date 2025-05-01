// main.cpp
#include "fdtd_simulation.hpp"
#include <iostream>

int main() {
    Parameters params;

    // シミュレーション領域設定
    double duration = 4e-9; // s
    double x_max = 1.5; // m
    double y_max = 1.5; // m
    double z_max = 1.5; // m

    // ソース、観測点の座標
    double x_input = 0.70; // m
    double y_input = 0.75; // m
    double z_input = 0.75; // m
    double x_output = 0.80; // m
    double y_output = 0.75; // m
    double z_output = 0.75; // m

    // 比誘電率、比透磁率
    params.rel_eps = 1.0; // 真空：1.0
    params.rel_mu = 1.0;  // 真空：1.0

    // 空間設定
    params.dx = 0.015; // m
    params.dy = 0.015; // m
    params.dz = 0.015; // m
    params.nx = static_cast<int>(x_max / params.dx) + 1; // x方向の格子数
    params.ny = static_cast<int>(y_max / params.dy) + 1; // y方向の格子数
    params.nz = static_cast<int>(z_max / params.dz) + 1; // z方向の格子数

    // 時間設定
    params.dt = 2e-11; // s
    params.nt = static_cast<int>(duration / params.dt); // 時間ステップ数

    // ソース（送信）位置
    params.src_i = static_cast<int>(x_input / params.dx) + 1;
    params.src_j = static_cast<int>(y_input / params.dy) + 1;
    params.src_k = static_cast<int>(z_input / params.dz) + 1;

    // 観測点
    params.obs_i = static_cast<int>(x_output / params.dx) + 1;
    params.obs_j = static_cast<int>(y_output / params.dy) + 1;
    params.obs_k = static_cast<int>(z_output / params.dz) + 1;

    // リッカー波形パラメータ
    params.f_peak = 1e9; // Hz
    params.t0 = 2e-10; // s

    std::cout << "Starting FDTD simulation...\n";
    FDTD simulation(params);
    simulation.run();
    std::cout << "Simulation complete. Results written to observation.csv\n";

    return 0;
}
