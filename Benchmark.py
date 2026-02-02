import numpy as np
import time
import csv
import random
import os
from RubiksCube import RubiksCube
from Solver import RubiksSolver


def run_benchmark_ohe(num_cubes=50, filename="benchmark_ohe_20_mosse.csv"):
    # Inizializziamo solo il risolutore OHE
    solver = RubiksSolver(pipeline='OHE')

    file_exists = os.path.isfile(filename)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Test_ID', 'Mosse_Scramble', 'Stima_IA_Iniziale',
            'Risolto', 'Tempo_Secondi', 'Nodi_Esplorati',
            'Lunghezza_Soluzione', 'Efficienza_Nodi_Sec'
        ])

        print(f"[*] Avvio Benchmark OHE (1-20 mosse)")
        print(f"[*] Salvataggio dati in: {filename}")
        print("-" * 70)

        for i in range(1, num_cubes + 1):
            # Scegliamo una profondità tra 1 e 20
            current_depth = random.randint(1, 20)

            # Creazione del cubo e scramble
            cube = RubiksCube()
            scramble_sequence = cube.scramble(current_depth)

            # Catturiamo la stima iniziale prima di risolvere
            stima_iniziale = solver.get_heuristic(cube)

            print(f"[{i}/{num_cubes}] Profondità: {current_depth} | Stima IA: {stima_iniziale}...", end=" ", flush=True)

            # Esecuzione della risoluzione (usando solve_adaptive_ultra)
            start_t = time.time()
            solution, nodes = solver.solve_adaptive_ultra(cube)
            end_t = time.time()

            tempo = end_t - start_t
            risolto = 1 if solution is not None else 0
            lunghezza = len(solution) if solution is not None else 0

            nodi_sec = int(nodes / tempo) if tempo > 0 else 0

            # Salvataggio riga
            writer.writerow([
                i, current_depth, stima_iniziale,
                risolto, f"{tempo:.4f}", nodes, lunghezza, nodi_sec
            ])

            status = "✅ RISOLTO" if risolto else "❌ FALLITO"
            print(f"{status} in {tempo:.2f}s")

    print("\n" + "=" * 70)
    print(f"[!] BENCHMARK OHE COMPLETATO!")
    print(f"[!] File generato: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark_ohe(num_cubes=50)