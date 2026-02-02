import time
import csv
import random
import os
from RubiksCube import RubiksCube
from Solver import RubiksSolver


def run_hard_benchmark(num_cubes=20, filename="benchmark_ohe_HARD.csv"):
    # Inizializziamo il risolutore OHE
    solver = RubiksSolver(pipeline='OHE')

    successi = 0

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Test_ID', 'Mosse_Scramble', 'Stima_IA_Iniziale',
            'Esito', 'Risolto_Bool', 'Tempo_Secondi',
            'Nodi_Esplorati', 'Lunghezza_Soluzione', 'Delta_Ottimalita'
        ])

        print(f"[*] AVVIO BENCHMARK HARD (Range 18-20 mosse)")
        print(f"[*] Configurazione: Epsilon dinamico fino a 1.6/1.7")
        print("-" * 70)

        for i in range(1, num_cubes + 1):
            current_depth = random.randint(17, 20)
            cube = RubiksCube()
            cube.scramble(current_depth)
            stima_iniziale = solver.get_heuristic(cube)

            print(f"[{i}/{num_cubes}] Scramble: {current_depth} | Stima IA: {stima_iniziale}")

            start_t = time.time()
            solution, nodes = solver.solve_adaptive_ultra(cube)
            end_t = time.time()

            tempo = end_t - start_t
            risolto = solution is not None
            esito_str = "RISOLTO" if risolto else "FALLITO"
            risolto_bool = 1 if risolto else 0

            if risolto:
                successi += 1
                lunghezza = len(solution)
                delta = lunghezza - current_depth
            else:
                lunghezza = 0
                delta = 0

            # Scrittura riga nel CSV
            writer.writerow([
                i, current_depth, stima_iniziale,
                esito_str, risolto_bool, f"{tempo:.4f}",
                nodes, lunghezza, delta
            ])

            status_icon = "✅" if risolto else "❌"
            print(f"    --> {status_icon} {esito_str} | Tempo: {tempo:.2f}s | Mosse: {lunghezza} (Delta: {delta})")
            print("-" * 30)

    percentuale = (successi / num_cubes) * 100
    print("\n" + "=" * 70)
    print(f" BENCHMARK COMPLETATO ")
    print(f" Totale cubi testati: {num_cubes}")
    print(f" Risolti con successo: {successi} ({percentuale:.1f}%)")
    print(f" I risultati dettagliati sono in: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    run_hard_benchmark(num_cubes=20)