import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_file = "data/raw_data/raw.csv"
df = pd.read_csv(input_file)

# Suppression de la colonne date
df = df.drop(columns=["date"])

# Features (X) sont sur toutes les colonnes sauf la dernière qui est la cible (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output
output_dir = "data/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Sauvegardes des résultats
X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

print("Split terminé et fichiers sauvegardés")
