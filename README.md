# 📌 Proyecto: Perceptrón Multicapa para Clasificación de Emociones en Texto  

Este repositorio contiene una implementación de un **Perceptrón Multicapa (MLP)** desde cero usando **Python y NumPy** para clasificar emociones en textos cortos (por ejemplo, tweets).  

El modelo usa **descenso de gradiente con mini-lotes** y el algoritmo de **Backpropagation** para el entrenamiento. También incluye funciones para validar y probar el modelo con un conjunto de datos etiquetado.  

---

## 🚀 Requisitos  

Antes de ejecutar el código, asegúrate de tener instalado lo siguiente:  

### 1️⃣ **Software y versiones necesarias**  
- **Python**: Versión 3.7 o superior  

### 2️⃣ **Librerías de Python necesarias**  
Ejecuta el siguiente comando para instalar las dependencias:  

```bash
pip install numpy matplotlib seaborn scikit-learn
```

### 3️⃣ **Conjunto de datos**  
Debes contar con tres archivos de texto:  
- `train.txt` → Datos de entrenamiento  
- `val.txt` → Datos de validación  
- `test.txt` → Datos de prueba  

📌 **Formato del dataset:**  
Cada línea debe contener un texto y su etiqueta, separados por un punto y coma (`;`).  

Ejemplo:  

```
Estoy muy feliz;alegría  
Esto es aterrador;miedo  
No me gusta esto;tristeza  
```

---

## 🎯 Funcionamiento  

### 🔹 **Entrenamiento del modelo**  
- Se cargan y preprocesan los datos.  
- Se crea un vocabulario y se convierten los textos en vectores numéricos.  
- Se entrena el MLP con los datos de entrenamiento.  
- Se valida el modelo con los datos de validación.  
- Se evalúa el modelo con los datos de prueba.  

📌 **Durante el entrenamiento**, el script mostrará la pérdida y precisión en cada iteración.  

### 🔹 **Predicción en tiempo real**  
Después del entrenamiento, puedes ingresar oraciones para predecir su emoción (en ingles).  

```bash
Escribe una oración (o 'salir' para terminar): i run to him when i feel threatened and insecure
Predicción: fear
```

Para salir, ingresa `salir`.  

---

## 📊 Salida del modelo  

### 📌 **Curvas de aprendizaje**  
El script generará gráficos de:  
- **Pérdida en entrenamiento y validación**  
- **Precisión en validación**  

### 📌 **Matriz de confusión**  
Se genera una **matriz de confusión** con los resultados en el conjunto de prueba para evaluar el rendimiento del modelo.  

---

## ⚙️ Personalización  

🔹 **Modificar el tamaño del vocabulario**  
Puedes cambiar el número de palabras utilizadas para vectorizar los textos modificando esta línea en el código:  

```python
vocab = list(vocab.keys())[:5000]  # Cambiar 5000 según sea necesario
```

🔹 **Ajustar hiperparámetros**  
Puedes modificar los siguientes valores en el código para mejorar el rendimiento del modelo:  

| Parámetro       | Descripción                          | Valor por defecto |
|----------------|----------------------------------|----------------|
| `hidden_size`  | Tamaño de la capa oculta          | `32`           |
| `learning_rate` | Tasa de aprendizaje              | `0.05`         |
| `epochs`       | Número de iteraciones de entrenamiento | `200`          |
| `batch_size`   | Tamaño de los mini-lotes         | `64`           |

---

## 🚧 Limitaciones y recomendaciones  

✅ **Idioma**: Actualmente, el modelo asume que los textos están en español.  
✅ **Precisión**: La calidad de la clasificación dependerá de la cantidad y diversidad del dataset.  
✅ **Optimización**: Para conjuntos de datos grandes, considera usar frameworks como **TensorFlow** o **PyTorch**.  

---
