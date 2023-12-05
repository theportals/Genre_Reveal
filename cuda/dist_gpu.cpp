/*
    Created by Bridger on 12/2/2023
    Adapted from shared_gpu and dist_cpu.
    Module load gcc/8.5.0, module load intel-mpi, module load cuda/12.2,
    Compilation:
        mpicxx -c dist_gpu.cpp -o dist_gpu.o
        nvcc -c updateCentroids.cu -o updateCentroids.o
        mpicxx dist_gpu.o updateCentroids.o -lcudart -o dist_gpu
*/

// dist_gpu.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <mpi.h>
#include "../point.h"

using namespace std;

double converge_threshold = 1e-7;

extern "C" void launch_update_centroids(Point* points, Point* centroids, int* nPoints, double* sumX, double* sumY, double* sumZ, int k, int n, int rank);

int main(int argc, char** argv) {
    int myRank, threads;

    // Initialize MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &threads);

    // Initialize arguments
    if (argc != 6) {
        if (myRank == 0) {
            cout << "Usage: mpiexec -n <number of processes> dist_gpu.exe <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    string filepath = argv[1];
    int k = strtol(argv[2], nullptr, 10);
    string xcol = argv[3];
    string ycol = argv[4];
    string zcol = argv[5];

    int n;
    vector<Point> points_data;

    // Rank 0 loads data from CSV using readcsv function
    if (myRank == 0) {
        auto before = chrono::high_resolution_clock::now();
        cout << "Loading points from csv (this may take a while)..." << endl;
        points_data = readcsv(filepath, xcol, ycol, zcol);
        auto after = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
        cout << points_data.size() << " points loaded in " << duration.count() << "ms." << endl;
        n = (int) points_data.size();
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);


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

    // Allocate and initialize arrays on myRank 0
    auto points = new Point[n];
    auto centroids = new Point[k];
    auto nPoints = new int[k];
    auto sumX = new double[k];
    auto sumY = new double[k];
    auto sumZ = new double[k];

    auto dataCounts = new int[threads];
    auto displacements = new int[threads];
    if (myRank == 0) {
        points = points_data.data();
        // Initialize centroids randomly
        srand(123);
        for (int i = 0; i < k; ++i) {
            centroids[i] = points_data[rand() % n];
        }

        // Figure out how many points each process should get
        int runningDisplacement = 0;
        for (int i = 0; i < threads; i++) {
            int c = n / threads;
            if (n % threads != 0 && i < n % threads) {
                // If data doesn't divide evenly, give the first few threads an extra point
                c += 1;
            }
            dataCounts[i] = c;
            displacements[i] = runningDisplacement;
            runningDisplacement += c;
        }
    }

    // Send data to all threads
    MPI_Bcast(dataCounts, threads, MPI_INT, 0, comm);
    int myDataCount = dataCounts[myRank];

    auto myData = new Point[myDataCount];
    MPI_Scatterv(&points[0], dataCounts, displacements, mpi_point_type, myData, myDataCount, mpi_point_type, 0, comm);

    MPI_Bcast(centroids, k, mpi_point_type, 0, MPI_COMM_WORLD);

    // Update step
    int epoch = 0;
    bool hasConverged = false;
    auto before = chrono::high_resolution_clock::now();
    if (myRank == 0) cout << "Beginning clustering..." << endl;
    while (!hasConverged) {
        epoch++;
        // MPI requires the send and receive buffers to be separate
        auto nPoints_r = new int[k];
        auto sumX_r = new double[k];
        auto sumY_r = new double[k];
        auto sumZ_r = new double[k];

        // Initialize/reset sum arrays with zeros
        for (int i = 0; i < k; i++) {
            nPoints[i] = 0;
            nPoints_r[i] = 0;
            sumX[i] = 0;
            sumY[i] = 0;
            sumZ[i] = 0;
            sumX_r[i] = 0;
            sumY_r[i] = 0;
            sumZ_r[i] = 0;
        }

        // Call the CUDA function
        // Function assigns each point to nearest centroid, updates counts
        launch_update_centroids(myData, centroids, nPoints, sumX, sumY, sumZ, k, myDataCount, myRank);
        MPI_Barrier(comm);

        MPI_Reduce(sumX, sumX_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumY, sumY_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(sumZ, sumZ_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(nPoints, nPoints_r, k, MPI_INT, MPI_SUM, 0, comm);

        // Compute new centroids on thread 0
        bool shouldEnd = true;
        if (myRank == 0) {
            for (int i = 0; i < k; i++) {
                Point c = centroids[i];
                double oldx = c.x;
                double oldy = c.y;
                double oldz = c.z;

                c.x = sumX_r[i] / nPoints_r[i];
                c.y = sumY_r[i] / nPoints_r[i];
                c.z = sumZ_r[i] / nPoints_r[i];

                double distMoved = (c.x - oldx) * (c.x - oldx) + (c.y - oldy) * (c.y - oldy) + (c.z - oldz) * (c.z - oldz);
                if (distMoved > converge_threshold) shouldEnd = false;
                centroids[i] = c;
            }
            hasConverged = shouldEnd;
        }
        MPI_Bcast(&hasConverged, 1, MPI_C_BOOL, 0, comm);
        if (!hasConverged) {
            // If we haven't converged, send centroids back out to other cores
            MPI_Bcast(centroids, k, mpi_point_type, 0, comm);
        }
    }
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    if (myRank == 0) cout << "Clustered after " << epoch << " epochs in " << duration.count() << "ms." << endl;

    // Retrieve point data from each thread
    MPI_Gatherv(&myData[0], myDataCount, mpi_point_type, &points[0], dataCounts, displacements, mpi_point_type, 0,
                comm);
    // Write to file
    if (myRank == 0) {
        string outpath;
        outpath.append("dist_gpu_").append(to_string(threads)).append(".csv");
        writeCSV(outpath, points, n);
    }

    MPI_Finalize();
    return 0;
}
