import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Définition des fichiers d'entrée et de sortie
input_train = "data/processed_data/X_train.csv"
input_test = "data/processed_data/X_test.csv"
output_train = "data/processed_data/X_train_scaled.csv"
output_test = "data/processed_data/X_test_scaled.csv"

# Chargement des données
X_train = pd.read_csv(input_train)
X_test = pd.read_csv(input_test)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertion en DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Dossier de sortie
os.makedirs("data/processed_data", exist_ok=True)

# Sauvegarde des fichiers normalisés
X_train_scaled.to_csv(output_train, index=False)
X_test_scaled.to_csv(output_test, index=False)

print("Normalisation terminée et fichiers sauvegardés !")
