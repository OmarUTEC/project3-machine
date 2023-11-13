#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

class CSVReader {
 public:
  CSVReader() {}

  pair<vector<vector<double>>, vector<vector<double>>> get_XY(string filename) {
    ifstream fdata;
    vector<vector<double>> X;
    vector<vector<double>> Y;
    string line;

    try {
      fdata = open_file(filename);
    } catch (const std::exception& e) {
      throw;
    }

    while (getline(fdata, line)) {
      istringstream iss(line);
      string token;
      vector<double> row;
      /// patient id (not used)
      getline(iss, token, ',');

      /// patient label (M, B)
      getline(iss, token, ',');
      Y.push_back({(token == "M" ? 0.0 : 1.0)});

      while (getline(iss, token, ',')) row.push_back(stod(token));
      X.push_back(row);
    }
    return make_pair(X, Y);
  }

 private:
  ifstream open_file(string filename) {
    ifstream file(filename);
    if (file.fail()) throw runtime_error("Could not open file: " + filename);
    return file;
  }
};