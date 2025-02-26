import numpy as np 
#Crear vectores, matrices, y realizar operaciones como la multiplicación de matrices, sumas, y cálculos de funciones como ReLU y Softmax.

from collections import defaultdict 
#Crear un vocabulario de palabras, donde cada palabra tiene un contador que se incrementa automáticamente cuando se encuentra en el texto.

import matplotlib.pyplot as plt 
#Graficar las curvas de aprendizaje (pérdida y precisión) y la matriz de confusión.

from sklearn.metrics import confusion_matrix 
#Calcular la matriz de confusión, que es una herramienta para evaluar la precisión de un modelo de clasificación.

import seaborn as sns
#Visualizar la matriz de confusión con un mapa de calor (heatmap), lo que facilita la interpretación de los resultados.

# Función para cargar los datos desde un archivo
def load_data(filename, limit=None):
    texts, labels = [], []
    with open(filename, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if limit and i >= limit:
                break
            text, label = line.strip().split(';')
            texts.append(text)
            labels.append(label)
    return texts, labels

# Cargar datos
train_texts, train_labels = load_data('train.txt')
val_texts, val_labels = load_data('val.txt')
test_texts, test_labels = load_data('test.txt')

# Crear vocabulario
vocab = defaultdict(int)
for text in train_texts:
    for word in text.split():
        vocab[word] += 1
vocab = list(vocab.keys())  # Reducir vocabulario para mayor velocidad

# Función para convertir texto a vector
def text_to_vector(text, vocab):
    vector = np.zeros(len(vocab))
    for word in text.split():
        if word in vocab:
            vector[vocab.index(word)] += 1
    return vector

# Convertir textos a vectores
train_vectors = np.array([text_to_vector(text, vocab) for text in train_texts])
val_vectors = np.array([text_to_vector(text, vocab) for text in val_texts])
test_vectors = np.array([text_to_vector(text, vocab) for text in test_texts])

# Convertir etiquetas a índices
label_set = list(set(train_labels))
label_to_index = {label: i for i, label in enumerate(label_set)}
index_to_label = {i: label for label, i in label_to_index.items()}
train_labels = np.array([label_to_index[label] for label in train_labels])
val_labels = np.array([label_to_index[label] for label in val_labels])
test_labels = np.array([label_to_index[label] for label in test_labels])

# Convertir etiquetas a one-hot encoding
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

num_classes = len(label_set)
train_labels_one_hot = one_hot_encode(train_labels, num_classes)
val_labels_one_hot = one_hot_encode(val_labels, num_classes)
test_labels_one_hot = one_hot_encode(test_labels, num_classes)

# Implementación del MLP con mini-lotes
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    # Función de activación ReLU
    def relu(self, x):
        return np.maximum(0, x)

    # Función Softmax para la salida (probabilidades de clases)
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Propagación hacia adelante
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    # Backpropagation: cálculo de gradientes y actualización de parámetros
    def backward(self, X, y, output):
        m = X.shape[0] # Número de ejemplos en el batch

        # Cálculo del gradiente de la función de pérdida en la capa de salida
        self.dz2 = output - y
        self.dW2 = np.dot(self.a1.T, self.dz2) / m # Gradiente de pesos de la capa de salida
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True) / m # Gradiente de sesgos de la capa de salida
        
        # Retropropagación hacia la capa oculta usando la derivada de ReLU
        self.dz1 = np.dot(self.dz2, self.W2.T) * (self.a1 > 0)
        self.dW1 = np.dot(X.T, self.dz1) / m # Gradiente de pesos de la capa oculta
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True) / m # Gradiente de sesgos de la capa oculta

    # Actualización de parámetros
    def update_parameters(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    # Entrenamiento del modelo
    def train(self, X, y, epochs, learning_rate, batch_size=64, X_val=None, y_val=None):
        train_losses, val_losses, val_accuracies = [], [], []
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X, y = X[indices], y[indices]
            for i in range(0, X.shape[0], batch_size):
                X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                self.update_parameters(learning_rate)

            # Calcular pérdida y precisión en el conjunto de entrenamiento y validación
            train_output = self.forward(X)
            train_loss = -np.sum(y * np.log(train_output)) / X.shape[0]
            train_losses.append(train_loss)

            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val)
                val_loss = -np.sum(y_val * np.log(val_output)) / X_val.shape[0]
                val_losses.append(val_loss)

                val_predictions = np.argmax(val_output, axis=1)
                val_accuracy = np.mean(val_predictions == np.argmax(y_val, axis=1))
                val_accuracies.append(val_accuracy)

            if epoch % 50 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

        return train_losses, val_losses, val_accuracies

    # Método predict
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Entrenar modelo con optimizaciones
input_size = train_vectors.shape[1]
hidden_size = 32
output_size = num_classes

mlp = MLP(input_size, hidden_size, output_size)
train_losses, val_losses, val_accuracies = mlp.train(
    train_vectors, train_labels_one_hot, epochs=200, learning_rate=0.05, batch_size=64,
    X_val=val_vectors, y_val=val_labels_one_hot
)

# Graficar curvas de aprendizaje
plt.figure(figsize=(12, 5))

# Curva de pérdida
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Curva de Pérdida')
plt.legend()

# Curva de precisión
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Curva de Precisión')
plt.legend()

plt.tight_layout()
plt.show()

# Matriz de confusión en el conjunto de prueba
test_predictions = mlp.predict(test_vectors)
test_labels_true = np.argmax(test_labels_one_hot, axis=1)

conf_matrix = confusion_matrix(test_labels_true, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_set, yticklabels=label_set)
plt.xlabel('Predicciones')
plt.ylabel('Etiquetas Verdaderas')
plt.title('Matriz de Confusión')
plt.show()

# Función para predecir una oración nueva
def predict_sentence(sentence, model, vocab):
    vector = text_to_vector(sentence, vocab).reshape(1, -1)
    pred_index = model.predict(vector)[0]
    return index_to_label[pred_index]

# Prueba con entrada del usuario
while True:
    sentence = input("Escribe una oración (o 'salir' para terminar): ")
    if sentence.lower() == 'salir':
        break
    print("Predicción:", predict_sentence(sentence, mlp, vocab))