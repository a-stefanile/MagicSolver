import numpy as np
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_manhattan_pipeline():
    # Caricamento del Dataset Manhattan
    dataset_path = 'rubiks_dataset_manhattan_2M.npz'

    if not os.path.exists(dataset_path):
        print(f"[!] Errore: Il file {dataset_path} non esiste. Generalo prima!")
        return

    print(f"[*] Caricamento del dataset Manhattan (Pipeline 2)...")
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']

    print(f"[+] Dataset caricato: {X.shape[0]} campioni.")
    print(f"[*] Dimensioni input: {X.shape[1]} (Manhattan Distances)")

    # Split del dataset (67% Training, 33% Test)
    # random_state=42 assicura che il test set sia confrontabile
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Configurazione del Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        verbose=1
    )

    # Addestramento
    print(f"[*] Inizio addestramento Pipeline 2. Il PC userà molta CPU...")
    start_time = time.time()

    model.fit(X_train, y_train)

    duration = (time.time() - start_time) / 60
    print(f"[+] Addestramento completato in {duration:.2f} minuti.")

    # Valutazione performance
    print("[*] Valutazione delle performance sul Test Set...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 40)
    print("      REPORT PIPELINE 2: MANHATTAN")
    print("=" * 40)
    print(f"Errore Medio Assoluto (MAE): {mae:.4f} mosse")
    print(f"Precisione (R2 Score):       {r2 * 100:.2f}%")
    print("=" * 40 + "\n")

    # Salvataggio del modello
    model_filename = 'magic_solver_manhattan.joblib'
    print(f"[*] Salvataggio modello in {model_filename}...")
    joblib.dump(model, model_filename)
    print(f"[!] Pronto! Hai completato la seconda pipeline.")


if __name__ == "__main__":
    train_manhattan_pipeline()