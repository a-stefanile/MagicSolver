import numpy as np
import time
import csv
import random
from RubiksCube import RubiksCube
from Solver import RubiksSolver


def run_benchmark(num_cubes=100, filename="benchmark_1_20_completo.csv"):
    # Carichiamo i risolutori una sola volta
    solvers = {
        'OHE': RubiksSolver(pipeline='OHE'),
        'MANHATTAN': RubiksSolver(pipeline='MANHATTAN')
    }

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header dettagliato per analisi statistiche avanzate
        writer.writerow([
            'ID_Test', 'Pipeline', 'Mosse_Reali', 'Stima_IA_Iniziale',
            'Risolto', 'Tempo_Secondi', 'Nodi_Esplorati', 'Lunghezza_Soluzione'
        ])

        print(f"[*] Avvio Benchmark Professionale (1-20 mosse casuali)")
        print(f"[*] Risultati in: {filename}")
        print("-" * 70)

        for i in range(1, num_cubes + 1):
            # Generiamo una profondità casuale per questo test
            current_depth = random.randint(1, 14)

            # Prepariamo il cubo e lo scramble
            base_cube = RubiksCube()
            scramble_sequence = base_cube.scramble(current_depth)

            print(f"[{i}/{num_cubes}] Testando profondità {current_depth}...")

            for p_name in ['OHE', 'MANHATTAN']:
                # Reset cubo per la pipeline corrente
                test_cube = RubiksCube()
                for m, r in scramble_sequence:
                    test_cube.rotate_face(m, reverse=r)

                # Catturiamo la prima stima dell'IA (Heuristic iniziale)
                stima_iniziale = solvers[p_name].get_heuristic(test_cube)

                # Risoluzione
                start_t = time.time()
                solution, nodes = solvers[p_name].solve_adaptive(test_cube)
                end_t = time.time()

                tempo = end_t - start_t
                risolto = 1 if solution is not None else 0
                lunghezza = len(solution) if solution is not None else 0

                # Scrittura nel file
                writer.writerow([
                    i, p_name, current_depth, stima_iniziale,
                    risolto, f"{tempo:.4f}", nodes, lunghezza
                ])

    print("\n" + "=" * 70)
    print(f"[!] BENCHMARK COMPLETATO CON SUCCESSO!")
    print(f"[!] Analizza {filename} su Excel per i grafici.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(num_cubes=100)
