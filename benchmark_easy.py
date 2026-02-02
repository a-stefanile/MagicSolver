import time
import csv
import os
from RubiksCube import RubiksCube
from Solver import RubiksSolver


def run_easy_benchmark(num_tests=20, filename='results_ml_easy.csv'):
    # Inizializziamo il solver
    solver = RubiksSolver(pipeline='OHE')

    print(f"\n[*] AVVIO BENCHMARK EASY (1-10 mosse)")
    print(f"[*] Ogni livello (1-10) verrà testato {num_tests} volte")
    print("-" * 60)

    # Scrittura Header
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Mosse_Scramble', 'Risolto', 'Tempo_Secondi',
            'Nodi_Esplorati', 'Lunghezza_Soluzione', 'Success_Rate'
        ])

        for m in range(1, 11):
            successi_livello = 0

            for i in range(num_tests):
                cube = RubiksCube()
                cube.scramble(m)

                start_t = time.time()
                solution, nodes = solver.solve_adaptive_ultra(cube)
                end_t = time.time()

                tempo = end_t - start_t

                if solution is not None:
                    solved = 1
                    successi_livello += 1
                    sol_length = len(solution)
                else:
                    solved = 0
                    sol_length = 0

                # Salvataggio nel CSV
                writer.writerow([m, solved, f"{tempo:.4f}", nodes, sol_length])

                status = "✅" if solved else "❌"
                print(f"   [Mossa {m}] Test {i + 1}/{num_tests}: {status} in {tempo:.2f}s")

            # Calcolo e stampa della percentuale di successo per questo livello
            rate = (successi_livello / num_tests) * 100
            print(f"--- LIVELLO {m} COMPLETATO: Successo {rate}% ---\n")

    print("=" * 60)
    print(f"[!] BENCHMARK COMPLETATO. Risultati salvati in: {filename}")


if __name__ == "__main__":
    run_easy_benchmark(num_tests=20)