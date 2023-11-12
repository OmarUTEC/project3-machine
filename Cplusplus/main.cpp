#include "CSVReader.hpp"
#include "mlp.cpp"

int main(int argc, char* argv[]) {
  auto reader{CSVReader()};
  try {

    int n = 30,m = 1,N,t;
    cout << "Ingrese la cantidad de capas ocultas: ";
    cin >> N;
    vector<int> v(N);
    for(int i = 0; i < N; i++) cin >> v[i];  
    cout << "Funciones de activacion disponibles: \n";  
    cout << "1. Sigmoid\n";
    cout << "2. Tanh\n";
    cout << "3. Relu\n";
    cout << "0. Identidad\n";
    cout << "Ingrese el tipo de función: ";
    cin >> t;  
    Mlp mlp = Mlp(n,v,m,t,0,1);
    cout << "Leyendo base de datos\n";
    auto [X, Y] = reader.get_XY("../db/db3.csv");

    double tasa_aprendizaje = 0.1;
    //mlp.entrenar(X,Y,1000,tasa_aprendizaje);

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    exit(EXIT_FAILURE);
  }
  return 0;
}
