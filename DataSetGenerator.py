import numpy as np
import multiprocessing as mp
import time
from RubiksCube import RubiksCube


class DataSetGenerator:
    def __init__(self, total_samples=2000000, max_moves=20):
        self.total_samples = total_samples
        self.max_moves = max_moves
        self.num_workers = mp.cpu_count()

        # Mappatura numerica delle mosse
        self.move_list = ['top', 'bottom', 'front', 'back', 'left', 'right']
        self.move_to_id = {name: i for i, name in enumerate(self.move_list)}

    def _generate_chunk(self, num_samples_chunk):
        """
        Funzione interna per la generazione di un singolo blocco di dati.
        """
        # X: 324 colonne per One-Hot Encoding degli sticker
        # y: Distanza (numero di mosse)
        # S: Sequenza delle mosse (ID da 0 a 11, -1 per padding)
        X = np.zeros((num_samples_chunk, 324), dtype=np.int8)
        y = np.zeros(num_samples_chunk, dtype=np.int8)
        S = np.full((num_samples_chunk, self.max_moves), -1, dtype=np.int8)

        for i in range(num_samples_chunk):
            cube = RubiksCube()
            n_scrambles = np.random.randint(1, self.max_moves + 1)

            sequence = []

            last_move_info = (None, None)

            while len(sequence) < n_scrambles:
                move = np.random.choice(self.move_list)
                is_reverse = np.random.choice([True, False])

                # CONTROLLO ANTI-ANNULLAMENTO
                # Se la mossa attuale è uguale alla precedente ma con reverse opposto
                # (es: R seguito da R'), allora la scartiamo e riproviamo.
                if last_move_info[0] == move and last_move_info[1] != is_reverse:
                    continue  # Salta il giro, genera un'altra mossa

                cube.rotate_face(move, reverse=is_reverse)

                # Encode mossa: 0-5 normali, 6-11 reverse
                m_id = self.move_to_id[move] + (6 if is_reverse else 0)
                sequence.append(m_id)

            X[i] = cube.get_state()
            y[i] = n_scrambles
            S[i, :n_scrambles] = sequence

        return X, y, S

    def generate(self, filename='rubiks_dataset_2M.npz'):
        """
        Esegue la generazione in parallelo e salva il file.
        """
        samples_per_worker = self.total_samples // self.num_workers
        print(f"[*] Avvio generazione di {self.total_samples} campioni...")
        print(f"[*] Utilizzo di {self.num_workers} core (chunk size: {samples_per_worker})")

        start_time = time.time()

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(self._generate_chunk, [samples_per_worker] * self.num_workers)

        print("[*] Generazione completata. Aggregazione dati...")

        # Unione dei risultati dai processi paralleli
        X_final = np.vstack([r[0] for r in results])
        y_final = np.concatenate([r[1] for r in results])
        S_final = np.vstack([r[2] for r in results])

        print(f"[*] Salvataggio in corso in '{filename}'...")
        np.savez_compressed(filename, X=X_final, y=y_final, moves=S_final)

        end_time = time.time()
        print(f"[+] Successo! Tempo impiegato: {(end_time - start_time) / 60:.2f} minuti.")


if __name__ == "__main__":
    gen = DataSetGenerator(total_samples=2000000, max_moves=14)
    gen.generate()
