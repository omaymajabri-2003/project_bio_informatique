# src/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Calcule les métriques de performance et les retourne sous forme de dictionnaire.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "confusion": confusion_matrix(y_test, y_pred)
    }
    
    return metrics

    # src/evaluation.py


def plot_confusion_matrix(y_true, y_pred, title="Matrice de Confusion"):
    """
    Affiche une carte thermique (heatmap) de la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.show()