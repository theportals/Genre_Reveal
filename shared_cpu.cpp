/*
    Created by Bridger 11/27/2023.
    Adapted from serialized code.
*/


#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include "point.h"

using namespace std;

string xcol;
string ycol;
string zcol;

double converge_threshold = 1e-7;

int threadCount;

int kMeansClustering(vector<Point>* points, int k);

int main(int argc, char* argv[]) {
    string filepath;

    if (argc != 7) {
        cout << "Usage: shared_cpu.exe <number of threads> <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        return -1;
    }

    // Initialize arguments
    threadCount = strtol(argv[1], nullptr, 10);
    filepath = argv[2];
    int k = strtol(argv[3], nullptr, 10);
    xcol = argv[4];
    ycol = argv[5];
    zcol = argv[6];


    // Read from csv file
    auto before = chrono::high_resolution_clock::now();
    cout << "Loading points from csv (this may take a while)..." << endl;
    vector<Point> points = readcsv(filepath, xcol, ycol, zcol);
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << points.size() << " points loaded in " << duration.count() << "ms." << endl;

    // Cluster centroids
    cout << "Beginning clustering..." << endl;
    before = chrono::high_resolution_clock::now();
    int epochsTaken = kMeansClustering(&points, k);
    after = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << "Clustered with " << epochsTaken << " epochs in " << duration.count() << "ms." << endl;

    // Write to file
    string base = "shared_cpu_";
    string extension = ".csv";
    string path = base.append(to_string(threadCount)).append(extension);
    writeCSV(path, points);
}

int kMeansClustering(vector<Point>* points, int k) {
    // Pick k points at random to create centroids
    vector<Point> centroids;
    srand(123);
    for (int i = 0; i < k; i++) {
        centroids.push_back((*points)[rand() % points->size()]);
    }

    // Used for calculating averages of cluster locations
    vector<int> nPoints(k, 0);
    vector<double> sumX(k, 0.0), sumY(k, 0.0), sumZ(k, 0.0);

    // Do our update step
    int epochs = 0;
    bool hasConverged = false;
    while (!hasConverged) {
        epochs++;

        // Assign each point to the nearest centroid in parallel
#pragma omp parallel for num_threads(threadCount) default(none) shared(points, centroids, k)
        for (int i = 0; i < points->size(); i++) {
            for (int j = 0; j < k; j++) {
                double dist = centroids[j].distance((*points)[i]);
                if (dist < (*points)[i].minDist) {
                    (*points)[i].minDist = dist;
                    (*points)[i].cluster = j;
                }
            }
        }

        // Initialize sum arrays with zeros
        fill(nPoints.begin(), nPoints.end(), 0);
        fill(sumX.begin(), sumX.end(), 0.0);
        fill(sumY.begin(), sumY.end(), 0.0);
        fill(sumZ.begin(), sumZ.end(), 0.0);

        // Iterate over points to append data to centroids
        for (auto &point : *points) {
            int clusterId = point.cluster;
#pragma omp atomic
            nPoints[clusterId]++;
#pragma omp atomic
            sumX[clusterId] += point.x;
#pragma omp atomic
            sumY[clusterId] += point.y;
#pragma omp atomic
            sumZ[clusterId] += point.z;
            point.minDist = DBL_MAX; // reset distance
        }

        // Compute the new centroids
        bool shouldEnd = true;
#pragma omp parallel for num_threads(threadCount) default(none) shared(k, centroids, sumX, sumY, sumZ, nPoints, converge_threshold, shouldEnd)
        for (int j = 0; j < k; j++) {
            double oldx = centroids[j].x;
            double oldy = centroids[j].y;
            double oldz = centroids[j].z;

            centroids[j].x = sumX[j] / nPoints[j];
            centroids[j].y = sumY[j] / nPoints[j];
            centroids[j].z = sumZ[j] / nPoints[j];

            double distMoved = (centroids[j].x - oldx) * (centroids[j].x - oldx) +
                               (centroids[j].y - oldy) * (centroids[j].y - oldy) +
                               (centroids[j].z - oldz) * (centroids[j].z - oldz);

            if (distMoved > converge_threshold)
                shouldEnd = false;
        }

        hasConverged = shouldEnd;
    }

    return epochs;
}
