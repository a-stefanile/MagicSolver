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
                         restart_prob=0.15, timeout_seconds=None, epsilon=1.0):
        """Beam Search OTTIMIZZATO con EPSILON ADATTIVA."""
        start_time = time.time()

        if start_cube.is_solved():
            return [], 0

        # Inizializziamo candidates con: (f_score, h_val, cubo, mosse)
        h_start = self.get_heuristic_cached(start_cube)
        candidates = [(h_start * epsilon, h_start, start_cube, [])]

        total_nodes = 0
        stagnation_counter = 0
        prev_best_h = float('inf')
        MAX_HEURISTIC_THRESHOLD = 22
        MAX_NODES = 2000000

        for depth in range(max_depth):
            # Check Timeout e Nodi
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                print(f"TIMEOUT {timeout_seconds}s (depth={depth}, nodi={total_nodes})")
                return None, total_nodes
            if total_nodes > MAX_NODES:
                print(f"Troppi nodi ({total_nodes}), stop anticipato")
                return None, total_nodes

            all_child_cubes = []
            all_child_moves = []

            # --- ESPANSIONE ---
            # Notare che ora spacchettiamo 4 valori: _, _, parent_cube, parent_moves
            for _, _, parent_cube, parent_moves in candidates:
                for move_name in self.moves:
                    for is_rev in [False, True]:
                        if parent_moves and parent_moves[-1] == (move_name, not is_rev):
                            continue

                        child = copy.deepcopy(parent_cube)
                        child.rotate_face(move_name, reverse=is_rev)
                        new_moves = parent_moves + [(move_name, is_rev)]
                        total_nodes += 1

                        if child.is_solved():
                            elapsed = time.time() - start_time
                            print(f"Risolto in {elapsed:.2f}s | Mosse: {len(new_moves)}")
                            return new_moves, total_nodes

                        all_child_cubes.append(child)
                        all_child_moves.append(new_moves)

            if not all_child_cubes:
                return None, total_nodes

            # --- PREDIZIONE BATCH ---
            X_batch = np.array([c.get_state() for c in all_child_cubes])
            heuristics = self.model.predict(X_batch)

            # --- SELEZIONE PESATA (EPSILON) ---
            combined = []
            for i in range(len(all_child_cubes)):
                h = heuristics[i]
                if h > MAX_HEURISTIC_THRESHOLD:
                    continue

                # Calcolo f = g + epsilon * h
                f_score = (depth + 1) + (h * epsilon)
                # Salviamo h separatamente per monitorare la stagnazione reale
                combined.append((f_score, h, all_child_cubes[i], all_child_moves[i]))

            if not combined:
                return None, total_nodes

            # Ordiniamo per f_score (primo elemento)
            combined.sort(key=lambda x: x[0])

            # --- MONITORAGGIO STAGNAZIONE (su h reale) ---
            current_best_h = combined[0][1]  # Prendiamo h reale dal secondo elemento
            if current_best_h >= prev_best_h:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_h = current_best_h

            # --- RESTART E LOG ---
            if np.random.random() < restart_prob or stagnation_counter >= 3:
                # Stampiamo h_best così capiamo se ci stiamo avvicinando alla soluzione
                print(f"RESTART d={depth + 1} (h_best={current_best_h:.1f})")

                elite_size = min(10, beam_width // 4)
                elite = combined[:elite_size]

                rest_size = beam_width - elite_size
                rest_candidates = combined[elite_size:min(len(combined), elite_size + rest_size * 3)]

                if len(rest_candidates) >= rest_size:
                    # Usiamo f_score (x[0]) per i pesi probabilistici
                    weights = np.exp(-np.array([x[0] for x in rest_candidates]) / 10.0)
                    weights /= weights.sum()
                    selected = np.random.choice(len(rest_candidates), rest_size, replace=False, p=weights)
                    rest = [rest_candidates[int(i)] for i in selected]
                else:
                    rest = rest_candidates

                candidates = elite + rest
                stagnation_counter = 0
            else:
                candidates = combined[:beam_width]

        return None, total_nodes

    def solve_adaptive(self, cube):
        """Alias per benchmark vecchio."""
        return self.solve_adaptive_ultra(cube)

    def solve_adaptive_ultra(self, cube):
        """
        Strategia adattiva con parametri potenziati per superare le 20 mosse.
        """
        stima_iniziale = self.get_heuristic(cube)
        print(f"\n{'=' * 60}")
        print(f"[*] Cubo con stima iniziale: {stima_iniziale} mosse")
        print(f"{'=' * 60}")

        # LIVELLO 1: Rapido (Cubi 1-12 mosse)
        print("[*] LIVELLO 1 (Veloce)...")
        path, nodes_1 = self.solve_beam_ultra(copy.deepcopy(cube), 200, 25, 0.10, 20,epsilon=1.0)
        if path:
            print("Risolto al Livello 1!")
            return path, nodes_1

        # LIVELLO 2: Medio (Cubi 13-17 mosse)
        print("[*] LIVELLO 2 (Medio-epsilon 1.3)...")
        path, nodes_2 = self.solve_beam_ultra(copy.deepcopy(cube), 1500, 40, 0.20, 120,epsilon=1.3)
        if path:
            print("Risolto al Livello 2!")
            return path, nodes_1 + nodes_2

        # LIVELLO 3: Intensivo (Cubi 18-20+ mosse)
        print("[*] LIVELLO 3 (Intensivo - 3 tentativi)...")
        total_restart_nodes = 0
        best_path = None

        for attempt in range(3):
            print(f"\n    --- Tentativo {attempt + 1}/3 ---")
            # Qui il beam_width è 3000 e MAX_NODES (nel metodo sopra) è 2 milioni
            path, nodes = self.solve_beam_ultra(copy.deepcopy(cube), 1500, 60, 0.40, 180,epsilon=1.6)
            total_restart_nodes += nodes

            if path:
                print(f"Risolto al tentativo {attempt + 1}!")
                return path, nodes_1 + nodes_2 + total_restart_nodes
            else:
                print(f"Tentativo {attempt + 1} fallito o timeout.")

        print("\n[!] Tutti i livelli falliti. Il cubo è troppo complesso per i parametri attuali.")
        return None, nodes_1 + nodes_2 + total_restart_nodes