import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Generar un conjunto de datos más complejo y no linealmente separable
X, y = make_circles(n_samples=300, noise=0.2, factor=0.3, random_state=42)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instanciar los modelos, incluyendo max_depth en Decision Tree
models = {
    'Logistic Regression': LogisticRegression(),
    'Linear SVM': LinearSVC(),
    'SVM with RBF Kernel': SVC(kernel='rbf', gamma=0.1),
    'Decision Tree (max_depth=10)': DecisionTreeClassifier(max_depth=10),
    #'Decision Tree (max_depth=5)': DecisionTreeClassifier(max_depth=5),
    #'Decision Tree (max_depth=10)': DecisionTreeClassifier(max_depth=10),
    #'Decision Tree (max_depth=None)': DecisionTreeClassifier()  # Sin límite de profundidad
}

# Entrenar y evaluar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluación del modelo
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    # Visualización de la separación
    plt.figure(figsize=(8, 4))
    plt.title(f"{name} Decision Boundary")
    
    # Plot de la superficie de decisión
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot de los puntos de datos
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', s=50, label='Train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', edgecolor='k', s=50, label='Test')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
