import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Supongamos que tienes etiquetas reales y etiquetas predichas para cinco clases
y_true = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]  # Ejemplo de etiquetas reales
y_pred = [0, 1, 1, 1, 2, 2, 4, 3, 4, 0]  # Ejemplo de etiquetas predichas

# Calcula la matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

# Configura el tamaño del gráfico
plt.figure(figsize=(8, 6))

# Crea el gráfico de la matriz de confusión
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(5), ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3', 'Clase 4'], rotation=45)
plt.yticks(np.arange(5), ['Clase 0', 'Clase 1', 'Clase 2', 'Clase 3', 'Clase 4'])
plt.xlabel('Etiquetas Predichas')
plt.ylabel('Etiquetas Reales')
plt.title('Matriz de Confusión (5x5)')

# Muestra los valores en cada celda
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

plt.show()

