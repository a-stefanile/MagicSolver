import numpy as np
from RubiksCube import RubiksCube
import multiprocessing as mp
from tqdm import tqdm
import random

# Definiamo le mosse fuori dalla funzione per non ricrearle 2 milioni di volte
FACES = ['top', 'bottom', 'front', 'back', 'left', 'right']


def generate_single_sample(_):
    """
    Funzione eseguita da ogni worker.
    Genera un cubo, lo mischia e calcola le distanze Manhattan.
    """
    cube = RubiksCube()

    # Scegliamo un numero di mosse tra 1 e 20
    num_moves = random.randint(1, 20)

    # Eseguiamo lo scramble
    for _ in range(num_moves):
        move = random.choice(FACES)
        rev = random.choice([True, False])
        cube.rotate_face(move, reverse=rev)

    # Estraiamo le feature (vettore di 54 distanze)
    # Questa chiamata ora usa la tua nuova funzione "slice and flatten"
    features = cube.get_manhattan_features()

    return features, num_moves


def create_parallel_dataset():
    num_samples = 2000000
    # Usiamo 16 core o quelli disponibili
    num_cores = mp.cpu_count()

    print(f"[*] Inizio generazione di {num_samples} campioni su {num_cores} core...")

    # 'with' assicura che i processi vengano chiusi correttamente
    with mp.Pool(processes=num_cores) as pool:
        # imap con chunksize è il modo più veloce per processare grandi liste
        # tqdm mostra la barra di avanzamento in tempo reale
        raw_results = list(tqdm(
            pool.imap(generate_single_sample, range(num_samples), chunksize=500),
            total=num_samples,
            desc="Generazione cubi"
        ))

    print("\n[*] Conversione in matrici NumPy...")
    X = np.array([res[0] for res in raw_results], dtype=np.uint8)
    y = np.array([res[1] for res in raw_results], dtype=np.uint8)

    output_file = 'rubiks_dataset_manhattan_2M.npz'
    print(f"[*] Salvataggio compresso in {output_file}...")
    np.savez_compressed(output_file, X=X, y=y)

    print(f"[+] Completato! File generato con successo.")


if __name__ == "__main__":
    mp.freeze_support()
    create_parallel_dataset()