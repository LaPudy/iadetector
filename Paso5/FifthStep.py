import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import load_npz
from pathlib import Path
import joblib

# Cargar los datos
BASE_DIR = Path(__file__).resolve().parents[1]
npztrain= BASE_DIR / "Paso4" / "X_train_tfidf.npz"
npztest= BASE_DIR / "Paso4" / "X_test_tfidf.npz"
X_train = load_npz(npztrain)
X_test = load_npz(npztest)
# Cargar etiquetas y asignar un nombre a la columna
ytrain= BASE_DIR / "Paso4" / "y_train.csv"
ytest= BASE_DIR / "Paso4" / "y_test.csv"
y_train = pd.read_csv(ytrain, header=None, names=["Etiqueta"])["Etiqueta"]
y_test = pd.read_csv(ytest, header=None, names=["Etiqueta"])["Etiqueta"]

# Entrenar el modelo
print("Entrenando el modelo de regresión logística...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, "modelo_clasificacion.pkl")

# Evaluar el modelo
print("Evaluando el modelo...")
y_pred = model.predict(X_test)

# Métricas de evaluación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

