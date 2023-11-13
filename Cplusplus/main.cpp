#include <algorithm>
#include <utility>
#include <vector>

#include "CSVReader.hpp"
#include "mlp.cpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {
  auto reader{CSVReader()};
  try {
    auto [X, Y] = reader.get_XY("../db/db3.csv");

    int n_train = X.size() * 7 / 10;

    matriz X_train = {X.begin(), X.begin() + n_train};
    matriz Y_train = {Y.begin(), Y.begin() + n_train};

    matriz X_test = {X.begin() + n_train, X.end()};
    matriz Y_test = {Y.begin() + n_train, Y.end()};

    MLP mlp(30, {10}, 2, 1, 0, 1);
    double tasa_aprendizaje = 0.1;
    mlp.entrenar(X_train, Y_train, 100, tasa_aprendizaje);

    cout << mlp.testing(X_test, Y_test) << endl;
    /*
    int n = 30,m = 1,N,t;
    cout << "Ingrese la cantidad de capas ocultas: ";
    cin >> N;
    vector<int> v(N);
    for (int i = 0; i < N; i++) cin >> v[i];
    cout << "Funciones de activacion disponibles: \n";
    cout << "1. Sigmoid\n";
    cout << "2. Tanh\n";
    cout << "3. Relu\n";
    cout << "0. Identidad\n";
    cout << "Ingrese el tipo de función: ";
    cin >> t;
    MLP mlp = MLP(n, v, m, t, 0, 1);
    cout << "Leyendo base de datos\n";
    auto [X, Y] = reader.get_XY("../db/db3.csv");
    auto [X_train, X_test, Y_train, Y_test] = train_test_split(X, Y);
    MLP mlp = MLP(n, v, m, t, 0, 1);
    cout << "Leyendo base de datos\n";
    auto [X, Y] = reader.get_XY("../db/db3.csv");
    auto [X_train, X_test, Y_train, Y_test] = train_test_split(X, Y);

    // double tasa_aprendizaje = 0.1;
    // mlp.entrenar(X, Y, 1000, tasa_aprendizaje);
    // double tasa_aprendizaje = 0.1;
    // mlp.entrenar(X, Y, 1000, tasa_aprendizaje);
      */
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    exit(EXIT_FAILURE);
  }
  return 0;
}
