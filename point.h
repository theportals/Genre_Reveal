//
// Created by Tom on 11/16/2023.
//

#ifndef GENRE_REVEAL_POINT_H
#define GENRE_REVEAL_POINT_H
#include <vector>
#include "csv-parser/csv.hpp"

using namespace std;
using namespace csv;

struct Point {
    double x, y, z;     // coordinates
    int cluster;        // which cluster the point belongs to
    double minDist;     // distance to nearest cluster center

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

    double distance(Point p) const {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
    }
};

/**
 * Reads a .csv file and returns a vector of points based off of all spotify songs in the dataset.
 * @param filepath path to file
 * @param xcol Feature title for the x-axis
 * @param ycol Feature title for the y-axis
 * @param zcol Feature title for the z-axis
 * @return points
 */
vector<Point> readcsv(const string& filepath, const string& xcol, const string& ycol, const string& zcol) {
    CSVReader reader(filepath);
    vector<Point> points;
    double x, y, z;

//    int maxct = 1000;
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

#endif //GENRE_REVEAL_POINT_H
