/*
    Created by Bridger on 12/2/2023
    Adapted from shared_gpu and dist_cpu.
    Module load intel-mpi, module load cuda/12.2, module load gcc/8.5.0
    Compilation: nvcc -o dist_gpu dist_gpu.cpp point.cpp -std=c++11 -Xcompiler -fopenmp -lcudart -lm
*/

#include <iostream>
#include <mpi.h>
#include "point.h"

using namespace std;

string xcol;
string ycol;
string zcol;

double converge_threshold = 1e-7;

__global__ void updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n);

void sum_points(void *inbuf, void *inoutbuf, int *len, MPI_Datatype *type);

int main(int argc, char* argv[]) {
    string filepath;

    // Initialize MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(nullptr, nullptr);
    int threads;
    MPI_Comm_size(comm, &threads);
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    if (argc != 7) {
        if (my_rank == 0) {
            cout << "Usage: mpiexec -n <number of processes> dist_gpu.exe <CUDA device id> <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    int device = strtol(argv[1], nullptr, 10);
    filepath = argv[2];
    int k = strtol(argv[3], nullptr, 10);
    xcol = argv[4];
    ycol = argv[5];
    zcol = argv[6];

    cudaSetDevice(device);
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);
    int blockSize = properties.maxThreadsPerBlock;

    // Read data on rank 0
    vector<Point> points;
    int dataSize;

    if (my_rank == 0) {
        auto before = chrono::high_resolution_clock::now();
        cout << "Loading points from csv (this may take a while)..." << endl;
        points = readcsv(filepath, xcol, ycol, zcol);
        dataSize = points.size();
        auto after = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
        cout << dataSize << " points loaded in " << duration.count() << "ms." << endl;
    }

    // Broadcast data size to all ranks
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, comm);

    // Scatter data among ranks
    int myDataCount = dataSize / threads;
    vector<Point> myData(myDataCount);
    MPI_Scatter(&points[0], myDataCount, mpi_point_type, &myData[0], myDataCount, mpi_point_type, 0, comm);

    // Initialize GPU variables
    vector<Point> centroids;
    if (my_rank == 0) {
        srand(123);
        for (int i = 0; i < k; i++) {
            centroids.push_back(points[rand() % dataSize]);
        }
    }

    // Broadcast initial centroids to all ranks
    MPI_Bcast(&centroids[0], k, mpi_point_type, 0, comm);

    // GPU memory allocation
    cudaError_t cudaErr = cudaSuccess;
    Point* points_d;
    Point* centroids_d;
    int* nPoints_d;
    double* sumX_d;
    double* sumY_d;
    double* sumZ_d;
    cudaErr = cudaMalloc((void **)&points_d, sizeof(Point) * myDataCount);
    cudaErr = cudaMalloc((void **)&centroids_d, sizeof(Point) * k);
    cudaErr = cudaMalloc((void **)&nPoints_d, sizeof(int) * k);
    cudaErr = cudaMalloc((void **)&sumX_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **)&sumY_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **)&sumZ_d, sizeof(double) * k);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)\n", cudaGetErrorString(cudaErr));
        MPI_Finalize();
        return -2;
    }

    // Copy data to device
    cudaErr = cudaMemcpy(points_d, &myData[0], sizeof(Point) * myDataCount, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(centroids_d, &centroids[0], sizeof(Point) * k, cudaMemcpyHostToDevice);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device (error code %s)\n", cudaGetErrorString(cudaErr));
        MPI_Finalize();
        return -3;
    }

    // GPU clustering
    int epochs = 0;
    bool hasConverged = false;
    auto before = chrono::high_resolution_clock::now();
    while (!hasConverged) {
        epochs++;

        // Invoke GPU kernel for clustering
        updateCentroids<<<ceil((double) myDataCount / blockSize), blockSize>>>(points_d, centroids_d, nPoints_d, sumX_d, sumY_d, sumZ_d, k, myDataCount);
        cudaDeviceSynchronize();

        if (cudaGetLastError() != cudaSuccess) {
            fprintf(stderr, "Failed to launch kernel\n");
            MPI_Finalize();
            return -4;
        }

        // MPI reduce to gather sums and counts from all ranks
        MPI_Reduce(sumX_d, sumX_d, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumY_d, sumY_d, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumZ_d, sumZ_d, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(nPoints_d, nPoints_d, k, MPI_INT, MPI_SUM, 0, comm);

        // Update centroids on rank 0
        if (my_rank == 0) {
            hasConverged = true;

            for (int j = 0; j < k; j++) {
                double oldx = centroids[j].x;
                double oldy = centroids[j].y;
                double oldz = centroids[j].z;

                centroids[j].x = sumX_d[j] / nPoints_d[j];
                centroids[j].y = sumY_d[j] / nPoints_d[j];
                centroids[j].z = sumZ_d[j] / nPoints_d[j];

                double distMoved = (centroids[j].x - oldx) * (centroids[j].x - oldx) +
                                   (centroids[j].y - oldy) * (centroids[j].y - oldy) +
                                   (centroids[j].z - oldz) * (centroids[j].z - oldz);

                if (distMoved > converge_threshold)
                    hasConverged = false;
            }
        }

        // Broadcast the convergence status
        MPI_Bcast(&hasConverged, 1, MPI_C_BOOL, 0, comm);

        // Broadcast updated centroids to all ranks
        MPI_Bcast(&centroids[0], k, mpi_point_type, 0, comm);
    }

    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);

    // Gather all clustered points on rank 0
    MPI_Gather(points_d, myDataCount, mpi_point_type, &myData[0], myDataCount, mpi_point_type, 0, comm);

    // Write to file on rank 0
    if (my_rank == 0) {
        ofstream myfile;
        myfile.open("output.csv");
        myfile << "x,y,z,c" << endl;
        for (auto &point : points) {
            myfile << point.x << "," << point.y << "," << point.z << "," << point.cluster << endl;
        }
        myfile.close();
        cout << "Written to output.csv" << endl;
    }

    // Clean up
    cudaFree(points_d);
    cudaFree(centroids_d);
    cudaFree(nPoints_d);
    cudaFree(sumX_d);
    cudaFree(sumY_d);
    cudaFree(sumZ_d);

    MPI_Finalize();

    return 0;
}

__global__ void updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        Point p = points[tid];

        for (int j = 0; j < k; j++) {
            Point c = centroids[j];
            double dist = (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) + (p.z - c.z) * (p.z - c.z);

            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = j;
            }
        }

        int cluster = p.cluster;
        atomicAdd(&nPoints[cluster], 1);
        atomicAdd(&sumX[cluster], p.x);
        atomicAdd(&sumY[cluster], p.y);
        atomicAdd(&sumZ[cluster], p.z);

        p.minDist = DBL_MAX;
        points[tid] = p;
    }
}

void sum_points(void *inbuf, void *inoutbuf, int *len, MPI_Datatype *type) {
    auto *invals = (Point *) inbuf;
    auto *inoutvals = (Point *) inoutbuf;

    for (int i = 0; i < *len; i++) {
        inoutvals[i].x += invals[i].x;
        inoutvals[i].y += invals[i].y;
        inoutvals[i].z += invals[i].z;
    }
}
