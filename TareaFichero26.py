from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from TareaFichero1 import X_test_2d, X_train_2d, y_train


# Supongamos que tienes etiquetas reales y etiquetas predichas para cinco clases
y_true = [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]  # Ejemplo de etiquetas reales
y_pred = [0, 1, 1, 1, 2, 2, 4, 3, 4, 0]  # Ejemplo de etiquetas predichas


# Calcula la matriz de confusión
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

# Muestra la matriz de confusión
print("Matriz de Confusión (5x5):")
print(cm)

# Calcula la precisión y la puntuación F1 macro
acc_svm = accuracy_score(y_true, y_pred)
f1_svm = f1_score(y_true, y_pred, average='macro')

print("Precisión:", acc_svm)
print("Puntuación F1 (macro):", f1_svm)

dt_model = DecisionTreeClassifier(max_depth=4)
dt_model.fit(X_train_2d, y_train)
y_test_pred_dt = dt_model.predict(X_test_2d)

print("Desicion=",y_test_pred_dt)

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_2d, y_train)
y_test_pred_knn = knn_model.predict(X_test_2d)

print("KNM=",y_test_pred_knn)
