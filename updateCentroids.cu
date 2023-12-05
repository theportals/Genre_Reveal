/*
    Created by Bridger 12/4/2023
    Modeled after shared_gpu
    Compilation: nvcc -c multiply.cu -o multiply.exe
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include "point.h"

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

extern "C" void launch_updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n) {
    //Dimensions
    int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // GPU memory allocation
    cudaError_t cudaErr = cudaSuccess;
    Point* points_d;
    Point* centroids_d;
    int* nPoints_d;
    double* sumX_d;
    double* sumY_d;
    double* sumZ_d;
    cudaErr = cudaMalloc((void **)&points_d, sizeof(Point) * n);
    cudaErr = cudaMalloc((void **)&centroids_d, sizeof(Point) * k);
    cudaErr = cudaMalloc((void **)&nPoints_d, sizeof(int) * k);
    cudaErr = cudaMalloc((void **)&sumX_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **)&sumY_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **)&sumZ_d, sizeof(double) * k);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)\n", cudaGetErrorString(cudaErr));
    }

    // Copy data from CPU to GPU
    cudaErr = cudaMemcpy(points_d, points, sizeof(Point) * n, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(centroids_d, centroids, sizeof(Point) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(nPoints_d, nPoints, sizeof(int) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumX_d, sumX, sizeof(double) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumY_d, sumY, sizeof(double) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumZ_d, sumZ, sizeof(double) * k, cudaMemcpyHostToDevice);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to copy data host to device (error code %s)\n", cudaGetErrorString(cudaErr));
    }

    updateCentroids<<<numBlocks, blockSize>>> (points_d, centroids_d, nPoints_d, sumX_d, sumY_d, sumZ_d, k, n);

    // Wait for the kernel to finish
    cudaThreadSynchronize();

    // Copy the result back from GPU to CPU
    cudaMemcpy(points, points_d, sizeof(Point) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(centroids, centroids_d, sizeof(Point) * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(nPoints, nPoints_d, sizeof(int) * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(sumX, sumX_d, sizeof(double) * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(sumY, sumY_d, sizeof(double) * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(sumZ, sumZ_d, sizeof(double) * k, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(points_d);
    cudaFree(centroids_d);
    cudaFree(nPoints_d);
    cudaFree(sumX_d);
    cudaFree(sumY_d);
    cudaFree(sumZ_d);

}


