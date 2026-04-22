# src/models/knn.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_optimized_knn(X_train, y_train):
    """
    Cherche le meilleur nombre de voisins (k) entre 3 et 15 
    et entraîne le modèle final.
    """
    # Définition de la plage de recherche (nombres impairs pour éviter les égalités)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'metric': ['euclidean', 'manhattan'] # Teste deux types de distances
    }
    
    # Création du modèle de base
    knn = KNeighborsClassifier()
    
    # GridSearch : teste toutes les combinaisons avec une validation croisée (CV) de 5
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Lancement de la recherche
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres : {grid_search.best_params_}")
    
    # Retourne le meilleur modèle déjà entraîné
    return grid_search.best_estimator_, grid_search.best_params_