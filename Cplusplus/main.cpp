#include "CSVReader.hpp"
#include "MLP.hpp"

int main(int argc, char* argv[]) {
  auto reader{CSVReader()};
  try {
    auto [X, Y] = reader.get_XY("../db/db3.csv");
    for (auto label : Y) cout << label << endl;
    // auto mlp = MLP();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    exit(EXIT_FAILURE);
  }
  return 0;
}
