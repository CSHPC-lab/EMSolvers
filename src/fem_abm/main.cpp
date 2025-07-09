#include <cmath>
#include <iostream>
#include <omp.h>

#include "fem_simulation.hpp"

int main()
{
    std::cout << "OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;

    // シミュレーションパラメータの設定
    std::array<double, 3> domain_sizes = {5.0, 5.0, 5.0};            // 計算領域のサイズ [m]
    std::array<int, 3> grid_num = {500, 500, 500};                   // グリッド数
    double duration = 3.0e-9;                                        // シミュレーション時間 [s]
    int num_steps = 780 * 2;                                         // 時間ステップ数
    double relative_permittivity = 1.0;                              // 比誘電率
    double relative_permeability = 1.0;                              // 比透磁率
    std::array<double, 3> source_position = {0.49, 0.49, 0.49};      // 入力点の位置
    std::array<double, 3> observation_position = {0.51, 0.51, 0.51}; // 観測点の位置
    int time_frequency = 1;

    double domain_size = domain_sizes[0] / grid_num[0];
    double time_step = duration / num_steps; // 時間ステップ [s]
    double permittivity =
        8.854187817e-12 * relative_permittivity; // 真空の誘電率 [F/m] * relative_permittivity
    double permeability =
        1.25663706212e-6 * relative_permeability; // 真空の透磁率 [H/m] * relative_permeability
    std::array<int, 3> source_index_0 = {
        static_cast<int>(std::round(source_position[0] / domain_size)) - 1,
        static_cast<int>(std::round(source_position[1] / domain_size)) - 1,
        static_cast<int>(std::round(source_position[2] / domain_size)) - 1}; // ヘルツダイポールの座標
    std::array<std::vector<std::array<int, 3>>, 3> observation_index = {
        std::vector<std::array<int, 3>>{
            {{static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
              static_cast<int>(std::round(observation_position[1] / domain_size)),
              static_cast<int>(std::round(observation_position[2] / domain_size))}}},
        std::vector<std::array<int, 3>>{
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
        },
        std::vector<std::array<int, 3>>{
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size)) - 1},
            {static_cast<int>(std::round(observation_position[0] / domain_size)) - 1,
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size)) - 1},
            {static_cast<int>(std::round(observation_position[0] / domain_size)),
             static_cast<int>(std::round(observation_position[1] / domain_size)),
             static_cast<int>(std::round(observation_position[2] / domain_size))},
        },
    }; // 観測点座標が始点から数えて何番目の節点か
    double c = 1.0 / std::sqrt(permittivity * permeability);

    // パラメータのチェック
    if (!check_params(domain_sizes, grid_num, duration, time_step, domain_size, c))
    {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }

    double start_time = omp_get_wtime();
    // シミュレーションオブジェクトの作成
    FemSimulation simulation(grid_num, domain_size, time_step, permittivity, permeability,
                             time_frequency);
    double end_time = omp_get_wtime();
    std::cout << "Simulation object created in " << end_time - start_time << " seconds" << std::endl;

    // 外力の設定
    simulation.setSourceDipole_x(source_index_0);

    // 観測点の設定
    simulation.setObservationPoint(observation_index);

    // 計算の実行
    std::cout << "Starting simulation..." << std::endl;
    start_time = omp_get_wtime();
    simulation.run(num_steps);
    end_time = omp_get_wtime();
    std::cout << "Simulation completed in " << end_time - start_time << " seconds" << std::endl;

    // 結果の保存
    simulation.saveResults(num_steps);
    std::cout << "Simulation completed. Results saved to results.csv" << std::endl;

    return 0;
}