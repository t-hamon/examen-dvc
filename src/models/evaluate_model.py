import pandas as pd
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, r2_score

# Définition des fichiers d'entrée et de sortie
input_X = "data/processed_data/X_test_scaled.csv"
input_y = "data/processed_data/y_test.csv"
input_model = "models/trained_model.pkl"
output_predictions = "data/processed_data/predictions.csv"
output_metrics = "metrics/scores.json"

# Chargement des données
X_test = pd.read_csv(input_X)
y_test = pd.read_csv(input_y)

# Chargement du modèle entraîné
with open(input_model, "rb") as f:
    model = pickle.load(f)

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarde des prédictions
predictions_df = pd.DataFrame({"y_test": y_test.values.ravel(), "y_pred": y_pred})
predictions_df.to_csv(output_predictions, index=False)

# Calcul des métriques
metrics = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

# Sauvegarde des métriques
os.makedirs("metrics", exist_ok=True)
with open(output_metrics, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Évaluation terminée. MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
