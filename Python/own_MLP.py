import numpy as np

class NeuralNetwork:
    def __init__(self, params):
        self.input_size = params["input_size"]
        self.hidden_layers = params["hidden_layers"]
        self.output_size = params["output_size"]
        self.activation_function = params["activation_function"]

        # Inicializar pesos y sesgos
        self.weights, self.biases = self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        weights = []
        biases = []

        # Capa de entrada
        weights.append(np.random.randn(self.hidden_layers[0], self.input_size))
        biases.append(np.zeros((self.hidden_layers[0], 1)))

        # Capas ocultas
        for i in range(1, len(self.hidden_layers)):
            weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i-1]))
            biases.append(np.zeros((self.hidden_layers[i], 1)))

        # Capa de salida
        weights.append(np.random.randn(self.output_size, self.hidden_layers[-1]))
        biases.append(np.zeros((self.output_size, 1)))

        return weights, biases

    def activate(self, x, activation_function):
        if activation_function == 'relu':
            return np.maximum(0, x)
        elif activation_function == 'tanh':
            return np.tanh(x)
        elif activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation_function == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exp_x / np.sum(exp_x, axis=0, keepdims=True)
        else:
            raise ValueError("Función de activación no válida")

    def forward(self, x):
        activations = [x]
        for i in range(len(self.hidden_layers) + 1):
            x = np.dot(self.weights[i], x) + self.biases[i]
            x = self.activate(x, self.activation_function)
            activations.append(x)
        return activations

    def loss_function(self, y_true, y_pred):
        # Softmax Cross-Entropy Loss
        m = y_true.shape[1]
        print(y_true)
        print(50*"*")
        print(np.log(y_pred + 1e-8))
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss

    def backward(self, x, y_true, activations):
        m = x.shape[1]
        gradients = []

        # Calcular el gradiente de la función de pérdida respecto a la salida
        dA = activations[-1] - y_true

        for i in range(len(self.hidden_layers), 0, -1):
            # Calcular gradientes locales
            dZ = dA
            dW = np.dot(dZ, activations[i - 1].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Almacenar gradientes
            gradients.insert(0, (dW, db))

            # Calcular gradiente para la capa anterior
            dA = np.dot(self.weights[i - 1].T, dZ)

        # Calcular gradientes para la capa de entrada
        dZ = dA
        dW = np.dot(dZ, x.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        gradients.insert(0, (dW, db))

        return gradients

    def update_parameters(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]

    def train(self, X, y, epochs, learning_rate):
        # Crear un arreglo para almacenar el historial de perdida
        loss_history = []

        for epoch in range(epochs):
            # Propagación hacia adelante
            activations = self.forward(X)

            # Calcular la pérdida
            loss = self.loss_function(y, activations[-1])
            loss_history.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

            # Retropropagación
            gradients = self.backward(X, y, activations)

            # Actualizar pesos y sesgos
            self.update_parameters(gradients, learning_rate)

        return loss_history

# # Ejemplo de uso
# params = {
#     "input_size" : 4,
#     "hidden_layers" : [5, 3],
#     "output_size" : 3,
#     "activation_function" : 'relu',
# }


# # Crear la red neuronal
# nn = NeuralNetwork(params)

# # Datos de entrenamiento de prueba (clasificación multiclase)
# X_train = np.random.randn(params["input_size"], 100)
# y_train = np.random.randint(2, size=(params["output_size"], 100))

# # Convertir etiquetas a codificación one-hot para softmax
# y_train_onehot = np.eye(params["output_size"])[y_train.reshape(-1)].T

# # Convertir etiquetas a codificación one-hot para softmax
# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(sparse=False)
# #y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1)).T


# # Entrenar la red neuronal
# loss_history = nn.train(X_train, y_train_onehot, epochs=100, learning_rate=0.01)

# import matplotlib.pyplot as plt

# plt.plot(loss_history)
# plt.title('Curva de la función de pérdida')
# plt.xlabel('Épocas')
# plt.ylabel('Pérdida')
# plt.show()
