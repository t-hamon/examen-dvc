import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor

# Définition des fichiers d'entrée et de sortie
input_X = "data/processed_data/X_train_scaled.csv"
input_y = "data/processed_data/y_train.csv"
input_params = "models/best_params.pkl"
output_model = "models/trained_model.pkl"

# Chargement les données
X_train = pd.read_csv(input_X)
y_train = pd.read_csv(input_y).values.ravel()  # Conversion pour éviter le warning

# Chargement des meilleurs hyperparamètres
with open(input_params, "rb") as f:
    best_params = pickle.load(f)

# Initialisation et entraînement du modèle
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné
os.makedirs("models", exist_ok=True)
with open(output_model, "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé !")
