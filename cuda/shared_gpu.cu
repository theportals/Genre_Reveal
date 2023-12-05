//
// Created by Tom on 11/29/2023.
//
#include <iostream>
#include "../point.hpp"

string xcol;
string ycol;
string zcol;

double converge_threshold = 1e-7;

__global__ void updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n);

int main(int argc, char* argv[]) {
    string filepath;

    if (argc != 7) {
        cout << "Usage: shared_gpu.exe <CUDA device id> <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        return -1;
    }

    // Initialize arguments
    int device = strtol(argv[1], nullptr, 10);
    filepath = argv[2];
    int k = strtol(argv[3], nullptr, 10);
    xcol = argv[4];
    ycol = argv[5];
    zcol = argv[6];
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);
    int blockSize = properties.maxThreadsPerBlock;

    // Read from csv file
    auto before = chrono::high_resolution_clock::now();
    cout << "Loading points from csv (this may take a while)..." << endl;
    vector<Point> points = readcsv(filepath, xcol, ycol, zcol);
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << points.size() << " points loaded in " << duration.count() << "ms." << endl;

    // Pick k points at random to create centroids
    vector<Point> centroids;
    srand(123);
    for (int i = 0; i < k; i++) {
        centroids.push_back(points[rand() % points.size()]);
    }

    // Used for calculating averages of cluster locations
    auto nPoints_h = new int[k];
    fill(&nPoints_h[0], &nPoints_h[k], 0);
    auto nPoints_d = new int[k];
    auto sumX_h = new double[k];
    auto sumY_h = new double[k];
    auto sumZ_h = new double[k];
    fill(&sumX_h[0], &sumX_h[k], 0.0);
    fill(&sumY_h[0], &sumY_h[k], 0.0);
    fill(&sumZ_h[0], &sumZ_h[k], 0.0);
    auto sumX_d = new double[k];
    auto sumY_d = new double[k];
    auto sumZ_d = new double[k];

    // Allocate device memory
    cudaError_t cudaErr = cudaSuccess;
    // CUDA likes arrays more than vectors
    Point* points_h = points.data();
    Point* centroids_h = centroids.data();
    Point* points_d;
    Point* centroids_d;
    cudaErr = cudaMalloc((void **) &points_d, sizeof(Point) * points.size());
    cudaErr = cudaMalloc((void **) &centroids_d, sizeof(Point) * k);
    cudaErr = cudaMalloc((void **) &nPoints_d, sizeof(int) * k);
    cudaErr = cudaMalloc((void **) &sumX_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **) &sumY_d, sizeof(double) * k);
    cudaErr = cudaMalloc((void **) &sumZ_d, sizeof(double) * k);
    cudaDeviceSynchronize();

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)\n", cudaGetErrorString(cudaErr));
        exit(-2);
    }

    // Copy host variables to device
    cudaErr = cudaMemcpy(points_d, points_h, sizeof(Point) * points.size(), cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(centroids_d, centroids_h, sizeof(Point) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(nPoints_d, nPoints_h, sizeof(int) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumX_d, sumX_h, sizeof(double) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumY_d, sumY_h, sizeof(double) * k, cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(sumZ_d, sumZ_h, sizeof(double) * k, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to copy host variables to device (error code %s)\n", cudaGetErrorString(cudaErr));
        exit(-3);
    }

    // Update step
    int epochs = 0;
    bool hasConverged = false;
    before = chrono::high_resolution_clock::now();
    while (!hasConverged) {
        epochs++;

        // Assign each point to the nearest centroid
        updateCentroids<<<ceil((double) points.size() / blockSize), blockSize>>>(points_d, centroids_d, nPoints_d, sumX_d, sumY_d, sumZ_d, k, points.size());
        cudaErr = cudaDeviceSynchronize();

        if (cudaErr != cudaSuccess) {
            fprintf(stderr, "Failed to start kernel (error code %s)\n", cudaGetErrorString(cudaErr));
            exit(-4);
        }

        // Retrieve updated sums from device
        // TODO: Make a function so we don't have this code in like twelve different 4-line blocks
        cudaErr = cudaMemcpy(nPoints_h, nPoints_d, sizeof(int) * k, cudaMemcpyDeviceToHost);
        cudaErr = cudaMemcpy(sumX_h, sumX_d, sizeof(double) * k, cudaMemcpyDeviceToHost);
        cudaErr = cudaMemcpy(sumY_h, sumY_d, sizeof(double) * k, cudaMemcpyDeviceToHost);
        cudaErr = cudaMemcpy(sumZ_h, sumZ_d, sizeof(double) * k, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (cudaErr != cudaSuccess) {
            fprintf(stderr, "Failed to copy updated points from device (error code %s)\n", cudaGetErrorString(cudaErr));
            exit(-5);
        }


        // Compute the new centroids. Because we usually have a small k, this doesn't need to be parallel
        bool shouldEnd = true;
        for (int j = 0; j < k; j++) {
            double oldx = centroids_h[j].x;
            double oldy = centroids_h[j].y;
            double oldz = centroids_h[j].z;

            centroids_h[j].x = sumX_h[j] / nPoints_h[j];
            centroids_h[j].y = sumY_h[j] / nPoints_h[j];
            centroids_h[j].z = sumZ_h[j] / nPoints_h[j];

            double distMoved = (centroids[j].x - oldx) * (centroids[j].x - oldx) +
                               (centroids[j].y - oldy) * (centroids[j].y - oldy) +
                               (centroids[j].z - oldz) * (centroids[j].z - oldz);

            if (distMoved > converge_threshold)
                shouldEnd = false;
        }

        hasConverged = shouldEnd;
        if (!hasConverged) {
            // If we haven't converged, copy new centroids to device, reset sums and nPoints
            for (int j = 0; j < k; j++) {
                nPoints_h[j] = 0;
                sumX_h[j] = 0.0;
                sumY_h[j] = 0.0;
                sumZ_h[j] = 0.0;
            }
            cudaErr = cudaMemcpy(nPoints_d, nPoints_h, sizeof(int) * k, cudaMemcpyHostToDevice);
            cudaErr = cudaMemcpy(sumX_d, sumX_h, sizeof(double) * k, cudaMemcpyHostToDevice);
            cudaErr = cudaMemcpy(sumY_d, sumY_h, sizeof(double) * k, cudaMemcpyHostToDevice);
            cudaErr = cudaMemcpy(sumZ_d, sumZ_h, sizeof(double) * k, cudaMemcpyHostToDevice);
            cudaErr = cudaMemcpy(centroids_d, centroids_h, sizeof(Point) * k, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            if (cudaErr != cudaSuccess) {
                fprintf(stderr, "Failed to copy new centroids to device (error code %s)\n", cudaGetErrorString(cudaErr));
                exit(-6);
            }
        }
    }
    after = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << "Clustered with " << epochs << " epochs in " << duration.count() << "ms." << endl;

    // Copy converged points from device
    cudaErr = cudaMemcpy(points_h, points_d, sizeof(Point) * points.size(), cudaMemcpyDeviceToHost);

    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to copy converged points from device (error code %s)\n", cudaGetErrorString(cudaErr));
        exit(-7);
    }

    // Write to file
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << endl;
    for (int i = 0; i < points.size(); i++) {
        Point point = points_h[i];
        myfile << point.x << "," << point.y << "," << point.z << "," << point.cluster << endl;
    }
    myfile.close();
    cout << "Written to output.csv" << endl;
}

__global__ void updateCentroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n) {
    // Calculate global thread index
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bx = blockDim.x;

    int gid = bx*bid + tid;
    if (gid < n) {
        Point p = points[gid];
        for (int j = 0; j < k; j++) {
            Point c = centroids[j];
            double dist = (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) + (p.z - c.z) * (p.z - c.z);
            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = j;
            }
        }

        // Rather than making a second kernel function, append data to centroids here
        int cluster = p.cluster;
        atomicAdd(&nPoints[cluster], 1);
        atomicAdd_block(&sumX[cluster], p.x);
        atomicAdd_block(&sumY[cluster], p.y);
        atomicAdd_block(&sumZ[cluster], p.z);

        p.minDist = DBL_MAX; // reset distance
        points[gid] = p;
    }
}