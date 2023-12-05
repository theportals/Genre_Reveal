/*
    Created by Bridger on 12/2/2023
    Adapted from shared_gpu and dist_cpu.
    Module load intel-mpi, module load cuda/12.2, module load gcc/8.5.0
    Compilation: mpixx -o dist_gpu dist_gpu.cpp -lcudart
*/

// dist_gpu.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>
#include "point.h"

extern "C" {
    void launch_update_centroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n);
}

int main(int argc, char** argv) {
    int rank, threads;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &threads);

    if (argc != 6) {
        if (rank == 0) {
            std::cout << "Usage: mpiexec -n <number of processes> dist_gpu.exe <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    std::string filepath = argv[1];
    int k = strtol(argv[2], nullptr, 10);
    std::string xcol = argv[3];
    std::string ycol = argv[4];
    std::string zcol = argv[5];

    // Load data from CSV using readcsv function
    auto before = chrono::high_resolution_clock::now();
    cout << "Loading points from csv (this may take a while)..." << endl;
    vector<Point> points_data = readcsv(filepath, xcol, ycol, zcol);
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << points_data.size() << " points loaded in " << duration.count() << "ms." << endl;
    int n = points_data.size();

    // MPI requires we specify how Points are transferred
    int nitems = 5;
    int blocklengths[5] = {1,1,1,1,1};
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
    MPI_Datatype mpi_point_type;
    MPI_Aint offsets[5];
    offsets[0]=offsetof(Point, x);
    offsets[1]=offsetof(Point, y);
    offsets[2]=offsetof(Point, z);
    offsets[3]=offsetof(Point, cluster);
    offsets[4]=offsetof(Point, minDist);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_point_type);
    MPI_Type_commit(&mpi_point_type);

    // Allocate and initialize arrays on rank 0
    Point* points = nullptr;
    Point* centroids = nullptr;
    int* nPoints = nullptr;
    double* sumX = nullptr;
    double* sumY = nullptr;
    double* sumZ = nullptr;

    if (rank == 0) {
        points = new Point[n];
        centroids = new Point[k];
        nPoints = new int[k]();
        sumX = new double[k]();
        sumY = new double[k]();
        sumZ = new double[k]();

        // Initialize centroids randomly
        srand(123);
        for (int i = 0; i < k; ++i) {
            centroids[i] = points_data[rand() % n];
        }

        // Scatter data to all ranks
        MPI_Scatter(points_data.data(), n / threads, mpi_point_type, points, n / threads, mpi_point_type, 0, MPI_COMM_WORLD);
    }

    // Broadcast centroids to all ranks
    MPI_Bcast(centroids, k, mpi_point_type, 0, MPI_COMM_WORLD);

    // Call the CUDA function
    before = chrono::high_resolution_clock::now();
    launch_update_centroids(points, centroids, nPoints, sumX, sumY, sumZ, k, n);
    after = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << "Clustered in " << duration.count() << "ms." << endl;

    // Write to file
    writeCSV("dist_gpu.csv", points, n);

    // Cleanup memory
    if (rank == 0) {
        delete[] points;
        delete[] centroids;
        delete[] nPoints;
        delete[] sumX;
        delete[] sumY;
        delete[] sumZ;
    }

    MPI_Finalize();
    return 0;
}
