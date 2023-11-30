//
// Created by Tom on 11/16/2023.
//

#include <iostream>
#include <mpi.h>
#include "point.h"

using namespace std;

string xcol;
string ycol;
string zcol;

double converge_threshold = 1e-7;

void sum_points(void *inbuf, void *inoutbuf, int *len, MPI_Datatype *type) {
    // Adds two point's coordinates together. Used when we re-calculate centroid coordinates
    auto *invals = (Point*) inbuf;
    auto *inoutvals = (Point*) inoutbuf;

    for (int i = 0; i < *len; i++) {
        inoutvals[i].x += invals[i].x;
        inoutvals[i].y += invals[i].y;
        inoutvals[i].z += invals[i].z;
    }
}

int main(int argc, char* argv[]) {
    int threads;
    int my_rank;

    string filepath;
    int k;
    int dataSize;
    vector<Point> points;

    // Open MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(comm, &threads);
    MPI_Comm_rank(comm, &my_rank);

    // Initialize arguments
    if (argc != 6) {
        if (my_rank == 0) {
            cout << "Usage: mpiexec -n <number of threads> dist_cpu.exe <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>"<< endl;
        }
        MPI_Finalize();
        return -1;
    }

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

    MPI_Op mpi_sum_points_op;
    MPI_Op_create(&sum_points, 0, &mpi_sum_points_op);

    if (my_rank == 0) {
        filepath = argv[1];
        k = strtol(argv[2], nullptr, 10);
        xcol = argv[3];
        ycol = argv[4];
        zcol = argv[5];

        // Rank 0 reads from csv
        auto before = chrono::high_resolution_clock::now();
        cout << "Reading points from csv (this may take a while)..." << endl;
        points = readcsv(filepath, xcol, ycol, zcol);
        dataSize = points.size();
        auto after = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
        cout << dataSize << " points loaded in " << duration.count() << "ms." << endl;
    }

    MPI_Bcast(&dataSize, 1, MPI_INT, 0, comm);
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);

    // It's easier to MPI_Bcast arrays over than vectors
    Point centroids[k];
    if (my_rank == 0) {
        // Pick k points at random to create centroids
        srand(123);
        for (int i = 0; i < k; i++) {
            centroids[i] = points[rand() % dataSize];
            printf("Centroid %d: (%f, %f, %f)\n", i, centroids[i].x, centroids[i].y, centroids[i].z);
        }
    }

    MPI_Bcast(&centroids, k, mpi_point_type, 0, comm);

    int dataCounts[threads];
    int displacements[threads];
    if (my_rank == 0) {
        // Figure out how many numbers each process should get
        int runningDisplacements = 0;
        for (int i = 0; i < threads; i++) {
            int c = dataSize / threads;
            if (dataSize % threads != 0 && i < dataSize % threads) {
                // If data doesn't divide evenly, give the first few threads an extra count
                c += 1;
            }
            dataCounts[i] = c;
            displacements[i] = runningDisplacements;
            runningDisplacements += c;
        }
    }
    MPI_Bcast(&dataCounts, threads, MPI_INT, 0, comm);
    int myDataCount = dataCounts[my_rank];

    vector<Point> myData(myDataCount);

    // Send portions of data to other threads
    MPI_Scatterv(&points[0], dataCounts, displacements, mpi_point_type, &myData[0], myDataCount, mpi_point_type, 0, comm);

    chrono::time_point<chrono::high_resolution_clock> before;
    chrono::time_point<chrono::high_resolution_clock> after;
    if (my_rank == 0) {
        cout << "Beginning clustering..." << endl;
        before = chrono::high_resolution_clock::now();
    }

    // Do our update step
    int epoch = 0;
    bool hasConverged = false;
    while (!hasConverged) {
        epoch++;
        if (my_rank == 0) {
            printf("Epoch %d: \n", epoch);
            for (int i = 0; i < k; i++) {
                Point c = centroids[i];
                printf("    c %d: (%f, %f, %f)\n", i, c.x, c.y, c.z);
            }
            printf("\n");
        }
        // Assign each point to the nearest centroid
        for (auto &p : myData) {
            for (int i = 0; i < k; i++) {
                Point c = centroids[i];
                double dist = c.distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = i;
                }
            }
        }
        // MPI requires the send and receive buffers to be separate
        int nPoints[k];
        int nPoints_r[k];
        double sumX[k], sumY[k], sumZ[k];
        double sumX_r[k], sumY_r[k], sumZ_r[k];

        // Initialize sum arrays with zeros
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

        // Iterate over points to append data to centroids
        for (auto &p : myData) {
            int clusterId = p.cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += p.x;
            sumY[clusterId] += p.y;
            sumZ[clusterId] += p.z;

            p.minDist = DBL_MAX; // reset distance
        }
        MPI_Reduce(&sumX, &sumX_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&sumY, &sumY_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&sumZ, &sumZ_r, k, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&nPoints, &nPoints_r, k, MPI_INT, MPI_SUM, 0, comm);

        if (my_rank == 0) {
            printf("    sums:\n");
            for (int i = 0; i < k; i++) {
                printf("        %d: (%f, %f, %f), %d\n", i, sumX_r[i], sumY_r[i], sumZ_r[i], nPoints_r[i]);

            }
            printf("\n");
        }

        // Compute the new centroids on rank 0
        bool shouldEnd = true;
        if (my_rank == 0) {
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
        }

        // Send the centroids back out to the other cores
        MPI_Bcast(&centroids, k, mpi_point_type, 0, comm);
        if (my_rank == 0) hasConverged = shouldEnd;
        MPI_Bcast(&hasConverged, 1, MPI_C_BOOL, 0, comm);
    }

    if (my_rank == 0) {
        after = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
        cout << "Converged after " << epoch << " epochs in " << duration.count() << "ms." << endl;
    }

    // Send all point info back to thread 0 for writing to file
    MPI_Gatherv(&myData[0], myDataCount, mpi_point_type, &points[0], dataCounts, displacements, mpi_point_type, 0, comm);

    // Write to file
    if (my_rank == 0) {
        // TODO: Move this into point.h to avoid recycled code
        ofstream myfile;
        myfile.open("output.csv");
        myfile << "x,y,z,c" << endl;
        for (auto &point: points) {
            myfile << point.x << "," << point.y << "," << point.z << "," << point.cluster << endl;
        }
        myfile.close();
        cout << "Written to output.csv" << endl;
    }
    // Close MPI
    MPI_Op_free(&mpi_sum_points_op);
    MPI_Finalize();
    return 0;
}