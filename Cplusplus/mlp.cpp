#include <vector>
#include <cmath>
#include <iostream>
#include <random>

using namespace std;
typedef vector<vector<double>> matriz;


class MLP{

private:
    int fa,_n,_m;
    matriz capas; 
    vector<matriz> pesos;

public:
    MLP(int n,vector<int> v,int m,int tipo_funcion,double min_valor = 0.0, double max_valor = 0.0){
        _n = n;
        _m = m;

        capas.resize(v.size()+2);
        capas[0].resize(n,0);
        for(int i = 0; i < v.size(); i++)
            capas[i+1].resize(v[i],0);
        capas[v.size()+1].resize(m,0);

        uniform_real_distribution<double> unif(min_valor, max_valor);
        random_device r;
        default_random_engine eng{r()};
        
        pesos.resize(v.size()+1);
        pesos[0].resize(n,vector<double>(v[0]));
        
        for(int i = 1; i < v.size(); i++)
            pesos[i].resize(v[i-1],vector<double>(v[i]));
        pesos[v.size()].resize(v[v.size()-1],vector<double>(m));

        for(int i = 0; i < pesos.size(); i++)
            for(int j = 0; j < pesos[i].size(); j++)
                for(int k = 0;k < pesos[i][j].size(); k++)
                    pesos[i][j][k] = unif(eng);
        fa = tipo_funcion;
       
    }
    
    
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

   
    double tanh(double x) {
        return tanh(x);
    }


    double relu(double x) {
        return max(0.0, x);
    }

    double funcion_activacion(double x) {
        switch (fa) {
            case 1:  
                return sigmoid(x);
            case 2:  
                return tanh(x);
            case 3:  
                return relu(x);
            default:
                return x;
        }
    }
   double derivada_activacion(double x) {
        switch (fa) {
            case 1:
                return sigmoid(x) * (1.0 - sigmoid(x));
            case 2:  
                return 1.0 - x * x;
            case 3:  
                return (x > 0) ? 1.0 : 0.0;
            default:
                return 1.0; 
        }
    }

  
    void forward(vector<double> &entrada) {
        
        for (int i = 0; i < entrada.size(); i++) {
            capas[0][i] = entrada[i];
        }

        for (int i = 1; i < capas.size(); i++) {
            for (int j = 0; j < capas[i].size(); j++) {
                double suma_ponderada = 0;
                for (int k = 0; k < capas[i-1].size(); k++) {
                    suma_ponderada += capas[i-1][k] * pesos[i-1][k][j];
                }
                capas[i][j] = funcion_activacion(suma_ponderada);
            }
        }

    }
    void backpropagation(vector<double>& objetivo, double tasa_aprendizaje) {
   
       
        vector<double> error_salida(capas[capas.size() - 1].size(), 0);
        for (int i = 0; i < error_salida.size(); i++) {
            error_salida[i] = objetivo[i] - capas[capas.size() - 1][i];
        }

       
        for (int i = capas.size() - 2; i > 0; i--) {
            for (int j = 0; j < capas[i].size(); j++) {
                double error = 0;
                for (int k = 0; k < capas[i + 1].size(); k++) {
                    error += pesos[i][j][k] * error_salida[k];
                }

                double derivada = derivada_activacion(capas[i][j]);

                for (int k = 0; k < capas[i - 1].size(); k++) {
                    double delta = tasa_aprendizaje * error * derivada * capas[i - 1][k];
                    pesos[i - 1][k][j] += delta;
                }
            }
        }
        for(int i = 0; i < capas[1].size(); i++){
            for(int j = 0; j < _m; j++){
                cout << pesos[0][i][j] << " ";
            }
            cout << endl;
        }

    }

    double softmax() {
        int m = capas[capas.size() - 1].size();
        double perdida = 0.0;
        vector<double> resultado(m, 0.0);
        double suma_exp = 0.0;

        for (int i = 0; i < m; i++) {
            resultado[i] = exp(capas[capas.size() - 1][i]);
            suma_exp += resultado[i];
        }

        for (int i = 0; i < m; i++) {
            resultado[i] = resultado[i] / suma_exp;
        }

      
        for (int i = 0; i < m; i++) {
            perdida += -capas[capas.size() - 1][i] * log(resultado[i] + 1e-10);
        }

        return perdida;
    }   
    
     void entrenar(matriz& entrada, matriz& salida, int n_iteracion, double tasa_aprendizaje) {
        if(entrada.size() == 0){
            cout << "No hay datos de entrenamiento\n";
            return;
        }   
        if(entrada.size() != salida.size()){
            cout << "No concuerda la cantidad de datos de entrada con la salida\n";
            return;
        }       
        if(entrada[0].size() != _n || salida[0].size() != _m){
             cout << "Error de tamaño de entrada o salida\n";
            return;
        }
        
        for (int iteracion = 0; iteracion < n_iteracion; iteracion++) {
            double perdida_total = 0.0;

            for (int i = 0; i < entrada.size(); i++) {
                forward(entrada[i]);
                backpropagation(salida[i], tasa_aprendizaje);
                perdida_total += softmax();
            }

            double perdida_promedio = perdida_total / entrada.size();
           // cout << iteracion + 1 << ". Pérdida: " << perdida_promedio << endl;
        }
    }

    double testing(matriz& entrada, matriz& salida){
        if(entrada.size() == 0){
            cout << "No hay datos de testring\n";
            return -1;
        }   
        if(entrada.size() != salida.size()){
            cout << "No concuerda la cantidad de datos de entrada con la salida\n";
            return -1;
        }     
        int correctos = 0;
        for(int i = 0; i < entrada.size(); i++){
            forward(entrada[i]);
            int tipo = -1, maxi = -1e9;
            cout << i << ": ";
            for(int j = 0; j < _m; j++){
                cout << capas[capas.size()-1][j] << " ";
                if(maxi < capas[capas.size()-1][j])
                    maxi = capas[capas.size()-1][j], tipo = j;
            }
            cout << endl;
            if(salida[i][tipo] > 0.99) correctos++;
        }
        return (1.0 * correctos) / entrada.size() * 100;
    }

};
