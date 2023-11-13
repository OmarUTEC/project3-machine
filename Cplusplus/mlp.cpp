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
        pesos[0].resize(n,vector<double>(v[0],unif(eng)));
        for(int i = 1; i < v.size(); i++)
            pesos[i].resize(v[i-1],vector<double>(v[i],unif(eng)));
        pesos[v.size()].resize(v[v.size()-1],vector<double>(m,unif(eng)));

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

       
        for (int i = 0; i < pesos[pesos.size() - 1].size(); i++) {
            for (int j = 0; j < pesos[pesos.size() - 1][i].size(); j++) {
                pesos[pesos.size() - 1][i][j] += tasa_aprendizaje * error_salida[j] * capas[capas.size() - 2][i];
            }
        }

       
        for (int i = capas.size() - 2; i > 0; i--) {
            for (int j = 0; j < capas[i].size(); j++) {
                double error = 0;
                for (int k = 0; k < capas[i+1].size(); k++) {
                    error += pesos[i][j][k] * error_salida[k];
                }
               
                for (int k = 0; k < capas[i-1].size(); k++) {
                    pesos[i-1][k][j] += tasa_aprendizaje * error * capas[i-1][k];
                }
            }
        }
    }

    double softmax(vector<double>& objetivo) {
        
        int m = objetivo.size();
        double perdida = 0.0;
        vector<double> resultado(m, 0.0);
        double suma_exp = 0.0;
     
        for (int i = 0; i < m; i++) {
            resultado[i] = exp(objetivo[i]);
            suma_exp += resultado[i];
        }

        for (int i = 0; i < m; i++) {
            resultado[i] = resultado[i] / suma_exp;
        }

      
        for (int i = 0; i < m; i++) {
            perdida += -objetivo[i] * log(resultado[i] + 1e-10);  
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

               
                perdida_total += softmax(salida[i]);
            }

            double perdida_promedio = perdida_total / entrada.size();

           cout << iteracion + 1 << ", Pérdida: " << perdida_promedio << endl;
        }
    }



};
