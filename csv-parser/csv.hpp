/*
    Created by Bridger 12/4/2023
    To be used by point.hpp
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
            header = parseLine(line);
        } else {
            std::cerr << "Empty file: " << filename << std::endl;
            return false;
        }

        while (std::getline(file, line)) {
            // Parse data rows
            std::vector<std::string> row = parseLine(line);
            parseLine(line);
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
    std::string filename;
    char delimiter;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> data;

    std::vector<std::string> parseLine(const std::string& line) const {
        std::vector<std::string> elements;
        std::stringstream ss(line);
        std::string element;

        while (std::getline(ss, element, delimiter)) {
            // Check if the element is quoted
            if (!element.empty() && element.front() == '"' && element.back() == '"') {
                // Remove quotes and add to the vector
                elements.push_back(element.substr(1, element.size() - 2));
            } else {
                elements.push_back(element);
            }
        }

        return elements;
    }
};
/*
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
*/