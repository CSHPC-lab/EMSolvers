#include <cmath>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <sstream>
#include <iomanip>
#include <openacc.h>

#include "fem_simulation.hpp"

int main(int argc, char *argv[])
{
    int size;
    int rank;
    int scale;
    int use_ofem;
    int order;
    double domain_size_x, domain_size_y, domain_size_z;
    int base_num;
    double duration;
    double source_position_x, source_position_y, source_position_z;
    double observation_position_x, observation_position_y, observation_position_z;
    int dim_x, dim_y, dim_z;

    std::string filename;
    std::array<double, 3> domain_sizes;
    std::array<int, 3> grid_nums;
    double relative_permittivity = 1.0;
    double relative_permeability = 1.0;
    std::array<int, 3> source_index, observation_index;
    int time_frequency;
    double domain_size;
    double permittivity = relative_permittivity * 8.8541878128e-12;
    double permeability = relative_permeability * 1.25663706212e-6;
    double c = 1.0 / std::sqrt(permittivity * permeability);
    double time_step;
    int num_time_step;
    double start_time, end_time;
    std::array<int, 3> dims, coords;
    MPI_Comm comm_x_plane_0, comm_x_plane_1, comm_y_plane_0, comm_y_plane_1, comm_z_plane_0, comm_z_plane_1;
    MPI_Comm comm_x_line_0, comm_x_line_1, comm_x_line_2, comm_x_line_3, comm_y_line_0, comm_y_line_1, comm_y_line_2, comm_y_line_3,
        comm_z_line_0, comm_z_line_1, comm_z_line_2, comm_z_line_3;
    int rank_x_plane_0, rank_x_plane_1, rank_y_plane_0, rank_y_plane_1, rank_z_plane_0, rank_z_plane_1;
    int rank_x_line_0, rank_x_line_1, rank_x_line_2, rank_x_line_3, rank_y_line_0, rank_y_line_1, rank_y_line_2, rank_y_line_3,
        rank_z_line_0, rank_z_line_1, rank_z_line_2, rank_z_line_3;
    int size_x_plane_0, size_x_plane_1, size_y_plane_0, size_y_plane_1, size_z_plane_0, size_z_plane_1;
    int size_x_line_0, size_x_line_1, size_x_line_2, size_x_line_3, size_y_line_0, size_y_line_1, size_y_line_2, size_y_line_3,
        size_z_line_0, size_z_line_1, size_z_line_2, size_z_line_3;
    int gpu_id, num_gpus;

    if (argc != 18)
    {
        std::cout << "Usage: mpirun -n <size> ./fem_simulation <scale> <use_ofem> <order> <domain_size_x> <domain_size_y> <domain_size_z> <base_num> <duration> <source_position_x> <source_position_y> <source_position_z> <observation_position_x> <observation_position_y> <observation_position_z> <dim_x> <dim_y> <dim_z>" << std::endl;
        std::cout << "  scale: Grid size multiplier (e.g., 1 for 100x100x100)" << std::endl;
        std::cout << "  use_ofem: Use OFEM (1) or standard FEM (0)" << std::endl;
        std::cout << "  order: element order" << std::endl;
        std::cout << "  domain_size_x, domain_size_y, domain_size_z: Size of the domain in each dimension" << std::endl;
        std::cout << "  base_num: Base grid number" << std::endl;
        std::cout << "  duration: Simulation duration" << std::endl;
        std::cout << "  source_position_x, source_position_y, source_position_z: Source position coordinates" << std::endl;
        std::cout << "  observation_position_x, observation_position_y, observation_position_z: Observation position coordinates" << std::endl;
        std::cout << "  dim_x, dim_y, dim_z: Dimensions for MPI Cartesian topology" << std::endl;

        return 1;
    }

    MPI_Init(&argc, &argv);

    // シミュレーションパラメータの設定
    scale = std::stoi(argv[1]);
    use_ofem = std::stoi(argv[2]);
    order = std::stoi(argv[3]);
    domain_size_x = std::stod(argv[4]);
    domain_size_y = std::stod(argv[5]);
    domain_size_z = std::stod(argv[6]);
    base_num = std::stoi(argv[7]);
    duration = std::stod(argv[8]);
    source_position_x = std::stod(argv[9]);
    source_position_y = std::stod(argv[10]);
    source_position_z = std::stod(argv[11]);
    observation_position_x = std::stod(argv[12]);
    observation_position_y = std::stod(argv[13]);
    observation_position_z = std::stod(argv[14]);
    dim_x = std::stoi(argv[15]);
    dim_y = std::stoi(argv[16]);
    dim_z = std::stoi(argv[17]);

    domain_sizes[0] = domain_size_x;
    domain_sizes[1] = domain_size_y;
    domain_sizes[2] = domain_size_z;
    grid_nums[0] = scale * base_num;
    grid_nums[1] = scale * base_num;
    grid_nums[2] = scale * base_num;
    domain_size = domain_sizes[0] / grid_nums[0];
    source_index[0] = static_cast<int>(std::round(source_position_x / domain_size));
    source_index[1] = static_cast<int>(std::round(source_position_y / domain_size));
    source_index[2] = static_cast<int>(std::round(source_position_z / domain_size));
    observation_index[0] = static_cast<int>(std::round(observation_position_x / domain_size));
    observation_index[1] = static_cast<int>(std::round(observation_position_y / domain_size));
    observation_index[2] = static_cast<int>(std::round(observation_position_z / domain_size));
    time_step = domain_size / (c * std::sqrt(3.0)) / 2.0;
    time_frequency = scale * 2;
    num_time_step = static_cast<int>(std::floor(duration / time_step) + 1);
    if (use_ofem)
    {
        filename = "symplectic4thofem" + std::to_string(order) + "_" + std::to_string(scale) + "_" + compress(domain_size_x) + "_" + compress(domain_size_y) + "_" + compress(domain_size_z) + "_" + compress(duration) +
                   "_" + compress(source_position_x) + "_" + compress(source_position_y) + "_" + compress(source_position_z) +
                   "_" + compress(observation_position_x) + "_" + compress(observation_position_y) + "_" + compress(observation_position_z) + ".csv";
    }
    else
    {
        filename = "symplectic4thfem" + std::to_string(order) + "_" + std::to_string(scale) + "_" + compress(domain_size_x) + "_" + compress(domain_size_y) + "_" + compress(domain_size_z) + "_" + compress(duration) +
                   "_" + compress(source_position_x) + "_" + compress(source_position_y) + "_" + compress(source_position_z) +
                   "_" + compress(observation_position_x) + "_" + compress(observation_position_y) + "_" + compress(observation_position_z) + ".csv";
    }
    // パラメータのチェック
    if (!check_params(domain_sizes, grid_nums, duration, time_step, domain_size, c))
    {
        std::cerr << "Invalid parameters" << std::endl;
        return 1;
    }
    dims[0] = dim_x;
    dims[1] = dim_y;
    dims[2] = dim_z;

    // MPIの初期化
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    coords = {rank % dims[0], (rank / dims[0]) % dims[1], rank / (dims[0] * dims[1])};
    std::cout << "Number of processes: " << size << " Process: " << rank << " coordinates: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")" << std::endl;
    MPI_Comm_split(MPI_COMM_WORLD, (coords[0] + 1) / 2 + coords[1] * dims[0] + coords[2] * dims[0] * dims[1], (coords[0] + 1) % 2, &comm_x_plane_0);
    MPI_Comm_rank(comm_x_plane_0, &rank_x_plane_0);
    MPI_Comm_size(comm_x_plane_0, &size_x_plane_0);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] / 2 + coords[1] * dims[0] + coords[2] * dims[0] * dims[1], coords[0] % 2, &comm_x_plane_1);
    MPI_Comm_rank(comm_x_plane_1, &rank_x_plane_1);
    MPI_Comm_size(comm_x_plane_1, &size_x_plane_1);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + (coords[1] + 1) / 2 * dims[0] + coords[2] * dims[0] * dims[1], (coords[1] + 1) % 2, &comm_y_plane_0);
    MPI_Comm_rank(comm_y_plane_0, &rank_y_plane_0);
    MPI_Comm_size(comm_y_plane_0, &size_y_plane_0);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + coords[1] / 2 * dims[0] + coords[2] * dims[0] * dims[1], coords[1] % 2, &comm_y_plane_1);
    MPI_Comm_rank(comm_y_plane_1, &rank_y_plane_1);
    MPI_Comm_size(comm_y_plane_1, &size_y_plane_1);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + coords[1] * dims[0] + (coords[2] + 1) / 2 * dims[0] * dims[1], (coords[2] + 1) % 2, &comm_z_plane_0);
    MPI_Comm_rank(comm_z_plane_0, &rank_z_plane_0);
    MPI_Comm_size(comm_z_plane_0, &size_z_plane_0);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + coords[1] * dims[0] + coords[2] / 2 * dims[0] * dims[1], coords[2] % 2, &comm_z_plane_1);
    MPI_Comm_rank(comm_z_plane_1, &rank_z_plane_1);
    MPI_Comm_size(comm_z_plane_1, &size_z_plane_1);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + (coords[1] + 1) / 2 * dims[0] + ((coords[2] + 1) / 2) * dims[0] * dims[1], (coords[1] + 1) % 2 + ((coords[2] + 1) % 2) * 2, &comm_x_line_0);
    MPI_Comm_rank(comm_x_line_0, &rank_x_line_0);
    MPI_Comm_size(comm_x_line_0, &size_x_line_0);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + coords[1] / 2 * dims[0] + ((coords[2] + 1) / 2) * dims[0] * dims[1], coords[1] % 2 + ((coords[2] + 1) % 2) * 2, &comm_x_line_1);
    MPI_Comm_rank(comm_x_line_1, &rank_x_line_1);
    MPI_Comm_size(comm_x_line_1, &size_x_line_1);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + (coords[1] + 1) / 2 * dims[0] + (coords[2] / 2) * dims[0] * dims[1], (coords[1] + 1) % 2 + (coords[2] % 2) * 2, &comm_x_line_2);
    MPI_Comm_rank(comm_x_line_2, &rank_x_line_2);
    MPI_Comm_size(comm_x_line_2, &size_x_line_2);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] + coords[1] / 2 * dims[0] + (coords[2] / 2) * dims[0] * dims[1], coords[1] % 2 + (coords[2] % 2) * 2, &comm_x_line_3);
    MPI_Comm_rank(comm_x_line_3, &rank_x_line_3);
    MPI_Comm_size(comm_x_line_3, &size_x_line_3);
    MPI_Comm_split(MPI_COMM_WORLD, ((coords[0] + 1) / 2) + coords[1] * dims[0] + ((coords[2] + 1) / 2) * dims[0] * dims[1], (coords[2] + 1) % 2 + ((coords[0] + 1) % 2) * 2, &comm_y_line_0);
    MPI_Comm_rank(comm_y_line_0, &rank_y_line_0);
    MPI_Comm_size(comm_y_line_0, &size_y_line_0);
    MPI_Comm_split(MPI_COMM_WORLD, ((coords[0] + 1) / 2) + coords[1] * dims[0] + (coords[2] / 2) * dims[0] * dims[1], (coords[2] + 1) % 2 + ((coords[0] + 1) % 2) * 2, &comm_y_line_1);
    MPI_Comm_rank(comm_y_line_1, &rank_y_line_1);
    MPI_Comm_size(comm_y_line_1, &size_y_line_1);
    MPI_Comm_split(MPI_COMM_WORLD, (coords[0] / 2) + coords[1] * dims[0] + ((coords[2] + 1) / 2) * dims[0] * dims[1], (coords[2] + 1) % 2 + ((coords[0] + 1) % 2) * 2, &comm_y_line_2);
    MPI_Comm_rank(comm_y_line_2, &rank_y_line_2);
    MPI_Comm_size(comm_y_line_2, &size_y_line_2);
    MPI_Comm_split(MPI_COMM_WORLD, (coords[0] / 2) + coords[1] * dims[0] + (coords[2] / 2) * dims[0] * dims[1], (coords[2] % 2) + (coords[0] % 2) * 2, &comm_y_line_3);
    MPI_Comm_rank(comm_y_line_3, &rank_y_line_3);
    MPI_Comm_size(comm_y_line_3, &size_y_line_3);
    MPI_Comm_split(MPI_COMM_WORLD, (coords[0] + 1) / 2 + ((coords[1] + 1) / 2) * dims[0] + coords[2] * dims[0] * dims[1], (coords[0] + 1) % 2 + ((coords[1] + 1) % 2) * 2, &comm_z_line_0);
    MPI_Comm_rank(comm_z_line_0, &rank_z_line_0);
    MPI_Comm_size(comm_z_line_0, &size_z_line_0);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] / 2 + ((coords[1] + 1) / 2) * dims[0] + coords[2] * dims[0] * dims[1], coords[0] % 2 + ((coords[1] + 1) % 2) * 2, &comm_z_line_1);
    MPI_Comm_rank(comm_z_line_1, &rank_z_line_1);
    MPI_Comm_size(comm_z_line_1, &size_z_line_1);
    MPI_Comm_split(MPI_COMM_WORLD, (coords[0] + 1) / 2 + (coords[1] / 2) * dims[0] + coords[2] * dims[0] * dims[1], (coords[0] + 1) % 2 + (coords[1] % 2) * 2, &comm_z_line_2);
    MPI_Comm_rank(comm_z_line_2, &rank_z_line_2);
    MPI_Comm_size(comm_z_line_2, &size_z_line_2);
    MPI_Comm_split(MPI_COMM_WORLD, coords[0] / 2 + (coords[1] / 2) * dims[0] + coords[2] * dims[0] * dims[1], coords[0] % 2 + (coords[1] % 2) * 2, &comm_z_line_3);
    MPI_Comm_rank(comm_z_line_3, &rank_z_line_3);
    MPI_Comm_size(comm_z_line_3, &size_z_line_3);
    std::cout << "rank: " << rank << " comm_x_plane_0 rank: " << rank_x_plane_0 << ", size: " << size_x_plane_0 << std::endl;
    std::cout << "rank: " << rank << " comm_x_plane_1 rank: " << rank_x_plane_1 << ", size: " << size_x_plane_1 << std::endl;
    std::cout << "rank: " << rank << " comm_y_plane_0 rank: " << rank_y_plane_0 << ", size: " << size_y_plane_0 << std::endl;
    std::cout << "rank: " << rank << " comm_y_plane_1 rank: " << rank_y_plane_1 << ", size: " << size_y_plane_1 << std::endl;
    std::cout << "rank: " << rank << " comm_z_plane_0 rank: " << rank_z_plane_0 << ", size: " << size_z_plane_0 << std::endl;
    std::cout << "rank: " << rank << " comm_z_plane_1 rank: " << rank_z_plane_1 << ", size: " << size_z_plane_1 << std::endl;
    std::cout << "rank: " << rank << " comm_x_line_0 rank: " << rank_x_line_0 << ", size: " << size_x_line_0 << std::endl;
    std::cout << "rank: " << rank << " comm_x_line_1 rank: " << rank_x_line_1 << ", size: " << size_x_line_1 << std::endl;
    std::cout << "rank: " << rank << " comm_x_line_2 rank: " << rank_x_line_2 << ", size: " << size_x_line_2 << std::endl;
    std::cout << "rank: " << rank << " comm_x_line_3 rank: " << rank_x_line_3 << ", size: " << size_x_line_3 << std::endl;
    std::cout << "rank: " << rank << " comm_y_line_0 rank: " << rank_y_line_0 << ", size: " << size_y_line_0 << std::endl;
    std::cout << "rank: " << rank << " comm_y_line_1 rank: " << rank_y_line_1 << ", size: " << size_y_line_1 << std::endl;
    std::cout << "rank: " << rank << " comm_y_line_2 rank: " << rank_y_line_2 << ", size: " << size_y_line_2 << std::endl;
    std::cout << "rank: " << rank << " comm_y_line_3 rank: " << rank_y_line_3 << ", size: " << size_y_line_3 << std::endl;
    std::cout << "rank: " << rank << " comm_z_line_0 rank: " << rank_z_line_0 << ", size: " << size_z_line_0 << std::endl;
    std::cout << "rank: " << rank << " comm_z_line_1 rank: " << rank_z_line_1 << ", size: " << size_z_line_1 << std::endl;
    std::cout << "rank: " << rank << " comm_z_line_2 rank: " << rank_z_line_2 << ", size: " << size_z_line_2 << std::endl;
    std::cout << "rank: " << rank << " comm_z_line_3 rank: " << rank_z_line_3 << ", size: " << size_z_line_3 << std::endl;

    std::cout << "rank: " << rank << " OMP_NUM_THREADS = " << omp_get_max_threads() << std::endl;

    // GPUデバイスの確認
    num_gpus = acc_get_num_devices(acc_device_nvidia);
    gpu_id = rank % num_gpus;
    acc_set_device_num(gpu_id, acc_device_nvidia);
    gpu_id = acc_get_device_num(acc_device_nvidia);
    std::cout << "rank: " << rank << " Using GPU device: " << gpu_id << " out of " << num_gpus << " available GPUs." << std::endl;

    // シミュレーション
    start_time = omp_get_wtime();
    FemSimulation simulation(order, grid_nums, domain_size, time_step, permittivity, permeability,
                             time_frequency, use_ofem, dims, coords, rank,
                             comm_x_plane_0, comm_x_plane_1, comm_y_plane_0, comm_y_plane_1, comm_z_plane_0, comm_z_plane_1,
                             comm_x_line_0, comm_x_line_1, comm_x_line_2, comm_x_line_3,
                             comm_y_line_0, comm_y_line_1, comm_y_line_2, comm_y_line_3,
                             comm_z_line_0, comm_z_line_1, comm_z_line_2, comm_z_line_3,
                             rank_x_plane_0, rank_x_plane_1, rank_y_plane_0, rank_y_plane_1, rank_z_plane_0, rank_z_plane_1,
                             rank_x_line_0, rank_x_line_1, rank_x_line_2, rank_x_line_3,
                             rank_y_line_0, rank_y_line_1, rank_y_line_2, rank_y_line_3,
                             rank_z_line_0, rank_z_line_1, rank_z_line_2, rank_z_line_3);
    end_time = omp_get_wtime();
    std::cout << "rank: " << rank << " Simulation object created in " << end_time - start_time << " seconds" << std::endl;
    // 外力の設定
    simulation.setSource_z(source_index);
    // 観測点の設定
    simulation.setObservationPoint(observation_index);
    // 計算の実行
    std::cout << "rank: " << rank << " Starting simulation..." << std::endl;
    start_time = omp_get_wtime();
    simulation.run(num_time_step);
    end_time = omp_get_wtime();
    std::cout << "rank: " << rank << " Simulation completed in " << end_time - start_time << " seconds" << std::endl;
    // 結果の保存
    simulation.saveResults(num_time_step, filename);
    std::cout << "rank: " << rank << " Simulation completed." << std::endl;

    // MPIの終了
    MPI_Comm_free(&comm_x_plane_0);
    MPI_Comm_free(&comm_x_plane_1);
    MPI_Comm_free(&comm_y_plane_0);
    MPI_Comm_free(&comm_y_plane_1);
    MPI_Comm_free(&comm_z_plane_0);
    MPI_Comm_free(&comm_z_plane_1);
    MPI_Comm_free(&comm_x_line_0);
    MPI_Comm_free(&comm_x_line_1);
    MPI_Comm_free(&comm_x_line_2);
    MPI_Comm_free(&comm_x_line_3);
    MPI_Comm_free(&comm_y_line_0);
    MPI_Comm_free(&comm_y_line_1);
    MPI_Comm_free(&comm_y_line_2);
    MPI_Comm_free(&comm_y_line_3);
    MPI_Comm_free(&comm_z_line_0);
    MPI_Comm_free(&comm_z_line_1);
    MPI_Comm_free(&comm_z_line_2);
    MPI_Comm_free(&comm_z_line_3);
    MPI_Finalize();

    return 0;
}
