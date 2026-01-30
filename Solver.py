import joblib
import numpy as np
import copy
import os
import time  # ← AGGIUNTO per timeout

# --- OTTIMIZZAZIONE SISTEMA ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class RubiksSolver:
    def __init__(self, pipeline='OHE'):
        self.pipeline = pipeline
        self.moves = ['top', 'bottom', 'front', 'back', 'left', 'right']

        # === CACHE PREDIZIONI ML ===
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"[*] Inizializzazione Solver. Pipeline selezionata: {pipeline}")

        # Caricamento modello
        try:
            if pipeline == 'OHE':
                print("[*] Caricamento modello OHE...")
                self.model = joblib.load('magic_solver_model.joblib')
            else:
                print("[*] Caricamento modello Manhattan...")
                self.model = joblib.load('magic_solver_manhattan.joblib')
        except FileNotFoundError as e:
            print(f"[-] ERRORE: {e}")
            exit()

        # Configurazione
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
        if hasattr(self.model, 'verbose'):
            self.model.verbose = 0
        if hasattr(self.model, 'estimators_'):
            for tree in self.model.estimators_:
                tree.verbose = 0

    def get_heuristic_cached(self, cube):
        """Euristica CON CACHE."""
        if cube.is_solved():
            return 0

        # Hash dello stato
        if self.pipeline == 'OHE':
            state_bytes = cube.get_state().tobytes()
        else:
            state_bytes = cube.get_manhattan_features().tobytes()

        state_hash = hash(state_bytes)

        # Check cache
        if state_hash in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[state_hash]

        # Calcola se non in cache
        self.cache_misses += 1
        if self.pipeline == 'OHE':
            features = cube.get_state().reshape(1, -1)
        else:
            features = cube.get_manhattan_features().reshape(1, -1)

        heuristic = int(np.floor(self.model.predict(features)[0]))
        self.prediction_cache[state_hash] = heuristic
        return heuristic

    def get_heuristic(self, cube):
        """Alias per retrocompatibilità."""
        return self.get_heuristic_cached(cube)

    def solve_beam_ultra(self, start_cube, beam_width, max_depth,
                         restart_prob=0.15, timeout_seconds=None):
        """Beam Search OTTIMIZZATO con TIMEOUT ADATTIVO."""

        # === INIZIO TIMER ===
        start_time = time.time()

        if start_cube.is_solved():
            return [], 0

        candidates = [(self.get_heuristic_cached(start_cube), start_cube, [])]
        total_nodes = 0

        stagnation_counter = 0
        prev_best_heuristic = float('inf')

        # === PRUNING DINAMICO ===
        stima_iniziale = self.get_heuristic_cached(start_cube)
        MAX_HEURISTIC_THRESHOLD = 15 if stima_iniziale > 12 else 20

        # === EARLY STOPPING ===
        MAX_NODES = 200000

        for depth in range(max_depth):
            # === CHECK TIMEOUT ===
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print(f"TIMEOUT {timeout_seconds}s (depth={depth}, nodi={total_nodes})")
                    return None, total_nodes

            # === CHECK EARLY STOPPING ===
            if total_nodes > MAX_NODES:
                print(f"Troppi nodi ({total_nodes}), stop anticipato")
                return None, total_nodes

            all_child_cubes = []
            all_child_moves = []

            # Espansione
            for _, parent_cube, parent_moves in candidates:
                for move_name in self.moves:
                    for is_rev in [False, True]:
                        # No backtracking
                        if parent_moves and parent_moves[-1] == (move_name, not is_rev):
                            continue

                        child = copy.deepcopy(parent_cube)
                        child.rotate_face(move_name, reverse=is_rev)
                        new_moves = parent_moves + [(move_name, is_rev)]
                        total_nodes += 1

                        # CHECK VITTORIA
                        if child.is_solved():
                            elapsed = time.time() - start_time
                            hit_rate = 100 * self.cache_hits / (self.cache_hits + self.cache_misses) if (
                                                                                                                    self.cache_hits + self.cache_misses) > 0 else 0
                            print(f"Risolto in {elapsed:.2f}s | Cache: {hit_rate:.1f}%")
                            return new_moves, total_nodes

                        all_child_cubes.append(child)
                        all_child_moves.append(new_moves)

            if not all_child_cubes:
                return None, total_nodes

            # Batch prediction
            if self.pipeline == 'OHE':
                X_batch = np.array([c.get_state() for c in all_child_cubes])
            else:
                X_batch = np.array([c.get_manhattan_features() for c in all_child_cubes])

            heuristics = self.model.predict(X_batch)

            #SELEZIONE + PRUNING
            combined = []
            for i in range(len(all_child_cubes)):
                h = heuristics[i]

                #PRUNING
                if h > MAX_HEURISTIC_THRESHOLD:
                    continue

                combined.append((h, all_child_cubes[i], all_child_moves[i]))

            if not combined:
                return None, total_nodes

            combined.sort(key=lambda x: x[0])

            #DETECT STAGNATION
            current_best = combined[0][0]
            if current_best >= prev_best_heuristic:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_heuristic = current_best

            #RIAVVIO CASUALE
            if np.random.random() < restart_prob or stagnation_counter >= 3:
                print(f"RESTART d={depth + 1} (best={current_best:.1f})")

                elite_size = min(10, beam_width // 4)
                elite = combined[:elite_size]

                rest_size = beam_width - elite_size
                rest_candidates = combined[elite_size:min(len(combined), elite_size + rest_size * 3)]

                if len(rest_candidates) >= rest_size:
                    weights = np.exp(-np.array([x[0] for x in rest_candidates]) / 10.0)
                    weights /= weights.sum()
                    selected = np.random.choice(
                        len(rest_candidates), rest_size,
                        replace=False, p=weights
                    )
                    rest = [rest_candidates[int(i)] for i in selected]
                else:
                    rest = rest_candidates

                candidates = elite + rest
                stagnation_counter = 0
            else:
                candidates = combined[:beam_width]

        return None, total_nodes

    def solve_adaptive_ultra(self, cube):
        """
        Strategia adattiva con TIMEOUT AUMENTATI per aumentare la chance di successo.

        TIMEOUTS OTTIMIZZATI:
        - Livello 1: 20s (cubi facili)
        - Livello 2: 60s (cubi medi)
        - Multi-Restart: 90s per tentativo (cubi difficili)

        TEMPO MASSIMO TOTALE: 20 + 60 + (90×3) = 350s (~6 minuti)
        """

        stima_iniziale = self.get_heuristic(cube)
        print(f"\n{'=' * 60}")
        print(f"[*] Cubo con stima iniziale: {stima_iniziale} mosse")
        print(f"{'=' * 60}")

        #LIVELLO 1
        print("[*] LIVELLO 1...")
        cube_copy_1 = copy.deepcopy(cube)
        path, nodes_1 = self.solve_beam_ultra(
            cube_copy_1, 200, 25,
            restart_prob=0.10,
            timeout_seconds=20
        )

        if path is not None:
            print("Risolto!")
            return path, nodes_1

        #LIVELLO 2
        print("[*] LIVELLO 2...")
        cube_copy_2 = copy.deepcopy(cube)
        path, nodes_2 = self.solve_beam_ultra(
            cube_copy_2, 800, 35,
            restart_prob=0.20,
            timeout_seconds=60
        )

        if path is not None:
            print("Risolto Livello!")
            return path, nodes_1 + nodes_2

        #MULTI-RESTART
        print("[*] MULTI-RESTART: 3 tentativi...")
        best_path = None
        best_length = float('inf')
        total_restart_nodes = 0

        for attempt in range(3):
            print(f"\n    --- Tentativo {attempt + 1}/3 ---")
            cube_copy = copy.deepcopy(cube)
            path, nodes = self.solve_beam_ultra(
                cube_copy, 1200, 40,
                restart_prob=0.35,
                timeout_seconds=90  # ← AUMENTATO da 60s a 90s
            )

            total_restart_nodes += nodes

            if path is not None:
                if len(path) < best_length:
                    best_path = path
                    best_length = len(path)
                    print(f"Soluzione trovata")
            else:
                print(f"Tentativo {attempt + 1} timeout/fallito")

        if best_path is not None:
            print(f"Risolto ")
            return best_path, nodes_1 + nodes_2 + total_restart_nodes

        print("\nTutti i livelli falliti")
        return None, nodes_1 + nodes_2 + total_restart_nodes

    def solve_adaptive(self, cube):
        """Alias per benchmark vecchio."""
        return self.solve_adaptive_ultra(cube)
