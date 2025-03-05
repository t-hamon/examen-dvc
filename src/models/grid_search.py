import pandas as pd
import os
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Définition des fichiers d'entrée et de sortie
input_X = "data/processed_data/X_train_scaled.csv"
input_y = "data/processed_data/y_train.csv"
output_file = "models/best_params.pkl"

# Chargement les données
X_train = pd.read_csv(input_X)
y_train = pd.read_csv(input_y)

# Définition du modèle et de la grille d'hyperparamètres
model = RandomForestRegressor()
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

# GridSearch
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres
os.makedirs("models", exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(grid_search.best_params_, f)

print(f"GridSearch terminé. Meilleurs paramètres : {grid_search.best_params_}")
