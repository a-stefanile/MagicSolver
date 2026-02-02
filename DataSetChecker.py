import numpy as np


def check_data(filename='rubiks_dataset_2M.npz'):
    print(f"--- Analisi del file: {filename} ---")

    # Caricamento del file
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non esiste.")
        return

    # Controllo delle chiavi
    keys = list(data.keys())
    print(f"Chiavi trovate: {keys}")
    if not all(k in keys for k in ['X', 'y', 'moves']):
        print("Errore: Mancano una o più chiavi fondamentali (X, y, moves).")
        return

    X = data['X']
    y = data['y']
    moves = data['moves']

    # Controllo delle dimensioni (Shapes)
    print("\n--- Dimensioni dei dati ---")
    print(f"Shape di X (Stati): {X.shape} (Dovrebbe essere [N, 324])")
    print(f"Shape di y (Distanze): {y.shape} (Dovrebbe essere [N])")
    print(f"Shape di moves (Sequenze): {moves.shape} (Dovrebbe essere [N, 14])")

    # Analisi di un campione casuale
    idx = np.random.randint(0, len(y))
    print(f"\n--- Analisi del campione casuale #{idx} ---")

    # Verifica X (One-Hot Encoding)
    unique_vals_x = np.unique(X[idx])
    print(f"Valori unici in X[{idx}]: {unique_vals_x} (Dovrebbero essere [0 1])")
    print(f"Numero di '1' in X[{idx}]: {np.sum(X[idx])} (Dovrebbero essere 54, uno per ogni sticker)")

    # Verifica Y (Distanza)
    print(f"Distanza dichiarata (y): {y[idx]}")

    # Verifica moves (Sequenza)
    m_seq = moves[idx]
    actual_moves_count = np.sum(m_seq != -1)
    print(f"Mosse reali nella sequenza: {actual_moves_count}")

    if actual_moves_count == y[idx]:
        print("[OK] La distanza y corrisponde al numero di mosse salvate.")
    else:
        print("[ERRORE] Discrepanza tra y e la lunghezza della sequenza!")

    print(f"ID delle mosse: {m_seq[:actual_moves_count]}")


if __name__ == "__main__":
    check_data()