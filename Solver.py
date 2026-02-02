import joblib
import numpy as np
import copy
import os
import time
import random

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


class RubiksSolver:
    def __init__(self, pipeline='OHE'):
        self.pipeline = pipeline
        self.moves = ['top', 'bottom', 'front', 'back', 'left', 'right']
        self.prediction_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"[*] Inizializzazione Solver. Pipeline selezionata: {pipeline}")

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

        # Disabilita i log di parallelismo
        if hasattr(self.model, 'verbose'):
            self.model.verbose = 0
        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = 1
        if hasattr(self.model, 'estimators_'):
            for est in self.model.estimators_:
                if hasattr(est, 'verbose'):
                    est.verbose = 0

    def get_heuristic_cached(self, cube):
        """Euristica con paracadute per evitare falsi positivi."""
        if cube.is_solved():
            return 0

        # Selezione features in base alla pipeline
        feat = cube.get_state() if self.pipeline == 'OHE' else cube.get_manhattan_features()
        state_hash = hash(feat.tobytes())

        if state_hash in self.prediction_cache:
            self.cache_hits += 1
            return self.prediction_cache[state_hash]

        self.cache_misses += 1
        raw_prediction = self.model.predict(feat.reshape(1, -1))[0]
        heuristic = int(np.round(raw_prediction))

        if heuristic <= 0:
            heuristic = 1

        self.prediction_cache[state_hash] = heuristic
        return heuristic

    def get_heuristic(self, cube):
        return self.get_heuristic_cached(cube)

    def solve_beam_ultra(self, start_cube, beam_width, max_depth,
                         restart_prob=0.15, timeout_seconds=None, epsilon=1.0):
        """Beam Search con Epsilon Adattiva."""
        start_time = time.time()
        if start_cube.is_solved(): return [], 0

        h_start = self.get_heuristic_cached(start_cube)
        candidates = [(h_start * epsilon, h_start, start_cube, [])]
        total_nodes = 0
        prev_best_h = float('inf')
        stagnation_counter = 0

        for depth in range(max_depth):
            if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                return None, total_nodes

            all_child_data = []
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
                            print(f"   >> Risolto d={depth + 1} | Nodi={total_nodes}")
                            return new_moves, total_nodes

                        all_child_data.append((child, new_moves))

            if not all_child_data: return None, total_nodes

            if self.pipeline == 'OHE':
                X_batch = np.array([c[0].get_state() for c in all_child_data])
            else:
                X_batch = np.array([c[0].get_manhattan_features() for c in all_child_data])

            heuristics = self.model.predict(X_batch)

            combined = []
            for i, h in enumerate(heuristics):
                f_score = (depth + 1) + (h * epsilon)
                combined.append((f_score, h, all_child_data[i][0], all_child_data[i][1]))

            combined.sort(key=lambda x: x[0])

            current_best_h = combined[0][1]
            if current_best_h >= prev_best_h:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            prev_best_h = current_best_h

            if np.random.random() < restart_prob or stagnation_counter >= 3:
                elite = combined[:min(10, beam_width // 4)]
                candidates = elite + random.sample(combined[len(elite):],
                                                   min(len(combined) - len(elite), beam_width - len(elite)))
                stagnation_counter = 0
            else:
                candidates = combined[:beam_width]

        return None, total_nodes

    def solve_adaptive_ultra(self, cube):
        self.prediction_cache = {}
        n1 = n2 = n3 = 0

        if cube.is_solved(): return [], 0

        # --- 1. CHECK-MATE PREVENTIVO ---
        for move_name in self.moves:
            for is_rev in [False, True]:
                test_cube = copy.deepcopy(cube)
                test_cube.rotate_face(move_name, reverse=is_rev)
                if test_cube.is_solved():
                    return [(move_name, is_rev)], 1

        # --- 2. LIVELLO 1: Rapido (Cerca la via più breve) ---
        # Timeout aggressivo
        path, n1 = self.solve_beam_ultra(copy.deepcopy(cube), 250, 25, 0.1, timeout_seconds=10, epsilon=1.0)
        if path: return path, n1

        # --- 3. LIVELLO 2: Espansivo (Bilanciato) ---
        # Aumentiamo epsilon a 1.4: diamo più peso all'IA per superare l'incertezza.
        path, n2 = self.solve_beam_ultra(copy.deepcopy(cube), 1200, 55, 0.3, timeout_seconds=45, epsilon=1.2)
        if path: return path, n1 + n2

        # --- 4. LIVELLO 3: Esplorazione profonda ---
        for attempt in range(2):
            path, n = self.solve_beam_ultra(copy.deepcopy(cube), 800, 80, 0.7, timeout_seconds=120, epsilon=1.5)
            n3 += n
            if path: return path, n1 + n2 + n3

        return None, n1 + n2 + n3