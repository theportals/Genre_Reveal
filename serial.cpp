//
// Created by Tom on 11/7/2023.
// Taken from tutorial at http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
//

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include "point.h"

using namespace std;

string xcol;
string ycol;
string zcol;

double converge_threshold = 1e-7;

int kMeansClustering(vector<Point>* points, int k);

int main(int argc, char* argv[]) {
    string filepath;

    if (argc != 6) {
        cout << "Usage: serial.exe <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        return -1;
    }

    // Initialize arguments
    filepath = argv[1];
    int k = strtol(argv[2], nullptr, 10);
    xcol = argv[3];
    ycol = argv[4];
    zcol = argv[5];

    // Read from csv file
    auto before = chrono::high_resolution_clock::now();
    cout << "Loading points from csv (this may take a while)..." << endl;
    vector<Point> points = readcsv(filepath, xcol, ycol, zcol);
    auto after = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << points.size() << " points loaded in " << duration.count() << "ms." << endl;

    // Cluster centroids
    cout << "Beginning clustering (this will definitely take a while)..." << endl;
    before = chrono::high_resolution_clock::now();
    int epochsTaken = kMeansClustering(&points, k);
    after = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(after - before);
    cout << "Clustered with " << epochsTaken << " epochs in " << duration.count() << "ms." << endl;

    // Write to file
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << endl;
    for (auto & point : points) {
        myfile << point.x << "," << point.y << "," << point.z << "," << point.cluster << endl;
    }
    myfile.close();
    cout << "Written to output.csv" << endl;
}

int kMeansClustering(vector<Point>* points, int k) {
    // Pick k points at random to create centroids
    vector<Point> centroids;
    srand(123);
    for (int i = 0; i < k; i++) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    // Used for calculating averages of cluster locations
    vector<int> nPoints(k, 0);
    vector<double> sumX(k, 0.0), sumY(k, 0.0), sumZ(k, 0.0);

    // Do our update step
    int epochs = 0;
    bool hasConverged = false;
    while (!hasConverged) {
        epochs++;
        // Assign each point to the nearest centroid
        for (auto & point : *points) {
            for (auto c = begin(centroids); c != end(centroids); c++) {
                int clusterId = c - begin(centroids);
                Point p = point;
                double dist = c->distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                point = p;
            }
        }

        // Initialize sum arrays with zeros
        for (int j = 0; j < k; j++) {
            nPoints[j] = 0;
            sumX[j] = 0;
            sumY[j] = 0;
            sumZ[j] = 0;
        }

        // Iterate over points to append data to centroids
        for (auto & point : *points) {
            int clusterId = point.cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += point.x;
            sumY[clusterId] += point.y;
            sumZ[clusterId] += point.z;

            point.minDist = DBL_MAX; // reset distance
        }

        // Compute the new centroids
        bool shouldEnd = true;
        for (auto c = begin(centroids); c != end(centroids); c++) {
            int clusterId = c - begin(centroids);
            double oldx = c->x;
            double oldy = c->y;
            double oldz = c->z;

            c->x = sumX[clusterId] / nPoints[clusterId];
            c->y = sumY[clusterId] / nPoints[clusterId];
            c->z = sumZ[clusterId] / nPoints[clusterId];

            double distMoved = (c->x - oldx) * (c->x - oldx) + (c->y - oldy) * (c->y - oldy) + (c->z - oldz) * (c->z - oldz);
            if (distMoved > converge_threshold) shouldEnd = false;
        }
        hasConverged = shouldEnd;
    }
    return epochs;
}
