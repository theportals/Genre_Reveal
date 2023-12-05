/*
    Created by Bridger 12/4/2023
    To be used by point.h
*/


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVParser {
public:
    CSVParser(const std::string& filename, char delimiter = ',') : filename(filename), delimiter(delimiter) {}

    bool parse() {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        std::string line;
        if (std::getline(file, line)) {
            // Parse header
            parseLine(line, header);
        } else {
            std::cerr << "Empty file: " << filename << std::endl;
            return false;
        }

        while (std::getline(file, line)) {
            // Parse data rows
            std::vector<std::string> row;
            parseLine(line, row);
            data.push_back(row);
        }

        file.close();
        return true;
    }

    void printData() const {
        // Print header
        for (const auto& column : header) {
            std::cout << column << "\t";
        }
        std::cout << std::endl;

        // Print data
        for (const auto& row : data) {
            for (const auto& value : row) {
                std::cout << value << "\t";
            }
            std::cout << std::endl;
        }
    }

    //Get parsed data as 2D vector
    const std::vector<std::vector<std::string>>& getData() const {
        return data;
    }

    // Get header information
    const std::vector<std::string>& getHeader() const {
        return header;
    }

private:
    void parseLine(const std::string& line, std::vector<std::string>& elements) const {
        std::stringstream ss(line);
        std::string element;

        while (std::getline(ss, element, delimiter)) {
            elements.push_back(element);
        }
    }

    std::string filename;
    char delimiter;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> data;
};

int main() {
    CSVParser csvParser("example.csv");

    if (csvParser.parse()) {
        //Get 2d parsed data
        const std::vector<std::vector<std::string>>& parsedData = csvParser.getData();

        //Access data
        for (const auto& row : parsedData) {
            for (const auto& value : row) {
                std::cout << value << "\t";
            }
            std::cout << std::endl;
        }

        // Access header
        const std::vector<std::string>& header = csvParser.getHeader();
        std::cout << "Header: ";
        for (const auto& column : header) {
            std::cout << column << "\t";
        }
    }

    return 0;
}
