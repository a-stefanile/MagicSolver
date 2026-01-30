import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model():
    # 1. Caricamento del dataset
    print("[*] Caricamento del dataset 'rubiks_dataset_2M.npz'...")
    data = np.load('rubiks_dataset_2M.npz')
    X = data['X']
    y = data['y']

    # 2. Split del dataset (67% training, 33% test)
    print("[*] Suddivisione dei dati (67% Training, 33% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print(f"    - Campioni di Training: {len(y_train)}")
    print(f"    - Campioni di Test: {len(y_test)}")

    # 3. Configurazione del Random Forest
    # n_estimators: numero di alberi (100 è un buon compromesso)
    # max_depth: profondità per evitare overfitting
    # n_jobs=-1: usa tutti i core della CPU
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        verbose=1
    )

    # 4. Addestramento
    print("[*] Inizio addestramento (questo richiederà tempo)...")
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = (time.time() - start_time) / 60
    print(f"[+] Addestramento completato in {duration:.2f} minuti.")

    # 5. Valutazione sul Test Set
    print("[*] Valutazione delle performance sul Test Set...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\n--- RISULTATI TEST ---")
    print(f"Errore Medio Assoluto (MAE): {mae:.3f} mosse")
    print(f"Coefficiente R2 (Accuratezza): {r2 * 100:.2f}%")
    print(f"----------------------\n")

    # 6. Salvataggio del modello
    model_filename = 'magic_solver_model.joblib'
    print(f"[*] Salvataggio del modello in {model_filename}...")
    joblib.dump(model, model_filename)
    print("[!] Pronto! Ora puoi usare questo modello")


if __name__ == "__main__":
    train_model()