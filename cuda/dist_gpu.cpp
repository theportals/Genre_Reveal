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

    // Call the CUDA function
    auto before = chrono::high_resolution_clock::now();
    launch_update_centroids(myData, centroids, nPoints, sumX, sumY, sumZ, k, myDataCount, myRank);
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    MPI_Barrier(comm);
    if (myRank == 0) cout << "Clustered in " << duration.count() << "ms." << endl;

    // Retrieve point data from each thread
    MPI_Gatherv(&myData[0], myDataCount, mpi_point_type, &points[0], dataCounts, displacements, mpi_point_type, 0, comm);

    // Write to file
    if (myRank == 0) writeCSV("dist_gpu.csv", points, n);

    // Cleanup memory
    if (myRank == 0) {
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
