//
// Created by Tom on 11/7/2023.
// Taken from tutorial at http://reasonabledeviations.com/2019/10/02/k-means-in-cpp/
//

#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <time.h>
#include "csv-parser/csv.hpp"

using namespace std;
using namespace csv;

// Global values for column ids. Must be in ascending order (i.e. xcol < ycol < zcol)
// numeric values we care about begin at 9, end at 23
//int xcol = 9;
//int ycol = 10;
//int zcol = 12;

string xcol;
string ycol;
string zcol;

struct Point {
    double x, y, z;    // coordinates
    int cluster;    // which cluster the point belongs to
    double minDist; // distance to nearest cluster center

    Point() :
        x(0.0),
        y(0.0),
        z(0.0),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    Point(double x, double y, double z) :
        x(x),
        y(y),
        z(z),
        cluster(-1),
        minDist(__DBL_MAX__) {}

    double distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
    }
};

vector<Point> readcsv(string filepath);

void kMeansClustering(vector<Point>* points, int epochs, int k);

int main(int argc, char* argv[]) {
    string filepath;

    if (argc != 6) {
        cout << "Usage: serial.exe <filepath to csv> <number of clusters> <x column key> <y column key> <z column key>" << endl;
        return -1;
    }

    filepath = argv[1];
    int k = strtol(argv[2], nullptr, 10);
    xcol = argv[3];
    ycol = argv[4];
    zcol = argv[5];
    long before = time(NULL);
    cout << "Loading points from csv (this may take a while)..." << endl;
    vector<Point> points = readcsv(filepath);
    long after = time(NULL);
    cout << points.size() << " points loaded in " << after - before << " seconds." << endl;

    cout << "Beginning clustering (this will definitely take a while)..." << endl;
    before = time(NULL);
    int epochs = 1000;
    kMeansClustering(&points, epochs, k);
    after = time(NULL);
    cout << "Clustered with " << epochs << " epochs in " << after - before << " seconds." << endl;
}

vector<Point> readcsv(string filepath) {
    CSVReader reader(filepath);
    vector<Point> points;
    double x, y, z;

    int maxct = 10000;
    int ct = 0;
    for (auto& row: reader) {
        ct++;
        if (row[xcol].is_num()) {
            x = row[xcol].get<double>();
        } else {
            cerr << "Value \"" << row[xcol] << "\" is not numeric." << endl;
            exit(2);
        }
        if (row[ycol].is_num()) {
            y = row[ycol].get<double>();
        } else {
            cerr << "Value \"" << row[ycol] << "\" is not numeric." << endl;
            exit(3);
        }
        if (row[xcol].is_num()) {
            z = row[zcol].get<double>();
        } else {
            cerr << "Value \"" << row[zcol] << "\" is not numeric." << endl;
            exit(4);
        }
        points.emplace_back(x, y, z);
//        if (ct == maxct) break;
    }
    return points;
}

void kMeansClustering(vector<Point>* points, int epochs, int k) {
    // Pick k points at random to create centroids
    vector<Point> centroids;
    srand(123);
    for (int i = 0; i < k; i++) {
        centroids.push_back(points->at(rand() % points->size()));
    }

    // Used for calculating averages of cluster locations
    vector<int> nPoints;
    vector<double> sumX, sumY, sumZ;

    // Do our update step
    for (int i = 0; i < epochs; i++) {
        // TODO: Implement method that goes until position changes are minimal, rather than for n epochs

        // Assign each point to the nearest centroid
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++) {
            int clusterId = c - begin(centroids);
            for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
                Point p = *it;
                double dist = c->distance(p);
                if (dist < p.minDist) {
                    p.minDist = dist;
                    p.cluster = clusterId;
                }
                *it = p;
            }
        }

        // Initialize sum arrays with zeros
        for (int j = 0; j < k; j++) {
            nPoints.push_back(0);
            sumX.push_back(0.0);
            sumY.push_back(0.0);
            sumZ.push_back(0.0);
        }

        // Iterate over points to append data to centroids
        for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
            int clusterId = it->cluster;
            nPoints[clusterId] += 1;
            sumX[clusterId] += it->x;
            sumY[clusterId] += it->y;
            sumZ[clusterId] += it->z;

            it->minDist = __DBL_MAX__; // reset distance
        }

        // Compute the new centroids
        for (vector<Point>::iterator c = begin(centroids); c != end(centroids); c++) {
            int clusterId = c - begin(centroids);
            c->x = sumX[clusterId] / nPoints[clusterId];
            c->y = sumY[clusterId] / nPoints[clusterId];
            c->z = sumZ[clusterId] / nPoints[clusterId];
        }
    }

    // Write to file
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "x,y,z,c" << endl;

    for (vector<Point>::iterator it = points->begin(); it != points->end(); it++) {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster << endl;
    }
    myfile.close();
}
