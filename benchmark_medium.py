import time
import csv
from RubiksCube import RubiksCube
from Solver import RubiksSolver


def run_medium_benchmark(num_tests=10, filename='results_ml_medium.csv'):
    solver = RubiksSolver(pipeline='OHE')

    print(f"\n{'=' * 60}")
    print(f"[*] AVVIO BENCHMARK MEDIUM (11-15 mosse)")
    print(f"[*] Test per livello: {num_tests} | Totale cubi: {num_tests * 5}")
    print(f"{'=' * 60}\n")

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Mosse_Scramble', 'Risolto', 'Tempo_Secondi', 'Nodi_Esplorati', 'Lunghezza_Soluzione'])

        for m in range(11, 16):
            successi_livello = 0
            tempi_livello = []

            print(f"--- LIVELLO {m} ---")
            for i in range(num_tests):
                cube = RubiksCube()
                cube.scramble(m)

                start_t = time.time()
                # Il solver gestirà internamente i passaggi tra i 3 livelli di intensità
                solution, nodes = solver.solve_adaptive_ultra(cube)
                end_t = time.time()

                tempo = end_t - start_t
                solved = 1 if solution is not None else 0
                sol_len = len(solution) if solved else 0

                if solved:
                    successi_livello += 1
                    tempi_livello.append(tempo)

                writer.writerow([m, solved, f"{tempo:.4f}", nodes, sol_len])

                status = "✅" if solved else "❌"
                print(f"   Test {i + 1:2}/{num_tests} | {status} | Tempo: {tempo:6.2f}s | Nodi: {nodes:8}")

            rate = (successi_livello / num_tests) * 100
            avg_t = sum(tempi_livello) / len(tempi_livello) if tempi_livello else 0
            print(f"\n[STIME LIVELLO {m}] Successo: {rate}% | Tempo Medio: {avg_t:.2f}s\n")

    print("=" * 60)
    print(f"[!] BENCHMARK MEDIUM COMPLETATO. File: {filename}")


if __name__ == "__main__":
    run_medium_benchmark()