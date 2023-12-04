//
// Created by Tom on 11/16/2023.
// Edited by Bridger to use new csv.h on 12/4/2023.
//

#ifndef GENRE_REVEAL_POINT_H
#define GENRE_REVEAL_POINT_H
#include <vector>
#include "csv-parser/csv.h"
#include <cfloat>

using namespace std;

struct Point {
    double x, y, z;     // coordinates
    int cluster;        // which cluster the point belongs to
    double minDist;     // distance to nearest cluster center

    Point() :
            x(0.0),
            y(0.0),
            z(0.0),
            cluster(-1),
            minDist(DBL_MAX) {}

    Point(double x, double y, double z) :
            x(x),
            y(y),
            z(z),
            cluster(-1),
            minDist(DBL_MAX) {}

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
    CSVParser csvParser(filepath);
    if (!csvParser.parse()) {
        cerr << "Error parsing CSV file." << endl;
        exit(1);
    }

    const auto& data = csvParser.getData();

    vector<Point> points;
    double x, y, z;

    for (const auto& row : data) {
        double x, y, z;
        
        if (istringstream(row[xcol]) >> x && istringstream(row[ycol]) >> y && istringstream(row[zcol]) >> z) {
            points.emplace_back(x, y, z);
        } else {
            cerr << "Error converting values to numeric in row." << endl;
            exit(2);
        }
    }
    return points;
}

#endif //GENRE_REVEAL_POINT_H
