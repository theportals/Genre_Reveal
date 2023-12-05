/*
    Created by Bridger 12/4/2023
    Modeled after shared_gpu
    Compilation: nvcc -c updateCentroids.cu -o updateCentroids.o
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include "../point.h"

inline cudaError_t checkCuda(cudaError_t result, const string& errorMessage) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s\n", errorMessage.c_str());
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
    return result;
}

// Function to perform atomic add on a double
__device__ double atomicAddDouble(double* address, double val) {
    auto* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old_val, new_val;
    do {
        old_val = *address_as_ull;
        new_val = __double_as_longlong(val + __longlong_as_double(old_val));
    } while (atomicCAS(address_as_ull, old_val, new_val) != old_val);
    return __longlong_as_double(old_val);
}

__global__ void updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < n) {
        // Assign each point to its nearest centroid
        Point p = points[tid];

        for (int j = 0; j < k; j++) {
            Point c = centroids[j];
            double dist = (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) + (p.z - c.z) * (p.z - c.z);

            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = j;
            }
        }

        // Append data to centroids
        int cluster = p.cluster;
        atomicAdd(&nPoints[cluster], 1);
        atomicAddDouble(&sumX[cluster], p.x);
        atomicAddDouble(&sumY[cluster], p.y);
        atomicAddDouble(&sumZ[cluster], p.z);

        p.minDist = DBL_MAX; // reset distance
        points[tid] = p;
    }
}

extern "C" void launch_update_centroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n, int thread) {
    //Dimensions
    int blockSize = 128;
    int numBlocks = ceil((double) n / blockSize);

    // GPU memory allocation
    Point* points_d;
    Point* centroids_d;
    int* nPoints_d;
    double* sumX_d;
    double* sumY_d;
    double* sumZ_d;
    checkCuda(cudaMalloc((void **)&points_d, sizeof(Point) * n), "Could not allocate points.");
    checkCuda(cudaMalloc((void **)&centroids_d, sizeof(Point) * k), "Could not allocate centroids.");
    checkCuda(cudaMalloc((void **)&nPoints_d, sizeof(int) * k), "Could not allocate nPoints.");
    checkCuda(cudaMalloc((void **)&sumX_d, sizeof(double) * k), "Could not allocate sumX.");
    checkCuda(cudaMalloc((void **)&sumY_d, sizeof(double) * k), "Could not allocate sumY.");
    checkCuda(cudaMalloc((void **)&sumZ_d, sizeof(double) * k), "Could not allocate sumZ.");

    // Copy data from CPU to GPU
    checkCuda(cudaMemcpy(points_d, points, sizeof(Point) * n, cudaMemcpyHostToDevice), "Could not copy points.");
    checkCuda(cudaMemcpy(centroids_d, centroids, sizeof(Point) * k, cudaMemcpyHostToDevice), "Could not copy centroids.");
    checkCuda(cudaMemcpy(nPoints_d, nPoints, sizeof(int) * k, cudaMemcpyHostToDevice), "Could not copy nPoints.");
    checkCuda(cudaMemcpy(sumX_d, sumX, sizeof(double) * k, cudaMemcpyHostToDevice), "Could not copy sumX.");
    checkCuda(cudaMemcpy(sumY_d, sumY, sizeof(double) * k, cudaMemcpyHostToDevice), "Could not copy sumY.");
    checkCuda(cudaMemcpy(sumZ_d, sumZ, sizeof(double) * k, cudaMemcpyHostToDevice), "Could not copy sumZ.");

    // Run kernel
    updateCentroids<<<numBlocks, blockSize>>> (points_d, centroids_d, nPoints_d, sumX_d, sumY_d, sumZ_d, k, n);

    // Wait for the kernel to finish
    string message;
    message.append("Thread ").append(to_string(thread)).append(" could not run kernel.");
    checkCuda(cudaDeviceSynchronize(), message);

    // Copy the result back from GPU to CPU
    checkCuda(cudaMemcpy(points, points_d, sizeof(Point) * n, cudaMemcpyDeviceToHost), "Could not copy points from device.");
    checkCuda(cudaMemcpy(centroids, centroids_d, sizeof(Point) * k, cudaMemcpyDeviceToHost), "Could not copy centroids from device.");
    checkCuda(cudaMemcpy(nPoints, nPoints_d, sizeof(int) * k, cudaMemcpyDeviceToHost), "Could not copy nPoints from device.");
    checkCuda(cudaMemcpy(sumX, sumX_d, sizeof(double) * k, cudaMemcpyDeviceToHost), "Could not copy sumX from device.");
    checkCuda(cudaMemcpy(sumY, sumY_d, sizeof(double) * k, cudaMemcpyDeviceToHost), "Could not copy sumY from device.");
    checkCuda(cudaMemcpy(sumZ, sumZ_d, sizeof(double) * k, cudaMemcpyDeviceToHost), "Could not copy sumZ from device.");

    // Free GPU memory
    cudaFree(points_d);
    cudaFree(centroids_d);
    cudaFree(nPoints_d);
    cudaFree(sumX_d);
    cudaFree(sumY_d);
    cudaFree(sumZ_d);

}


