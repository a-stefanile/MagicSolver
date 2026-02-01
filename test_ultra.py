from RubiksCube import RubiksCube
from Solver import RubiksSolver
import time

# 1. Inizializzazione
c = RubiksCube()
solver = RubiksSolver(pipeline='OHE') # Cambia in 'MANHATTAN' se preferisci quel modello

# 2. Scramble di 20 mosse reali
# Usiamo un seed per poter replicare il test se necessario
print("[*] Generazione scramble di 20 mosse...")
scramble_path = c.scramble(20)
print(f"[*] Scramble effettuato: {scramble_path}")

# 3. Risoluzione Adattiva
print("\n[*] Avvio risoluzione adattiva...")
start_time = time.time()
solution, total_nodes = solver.solve_adaptive_ultra(c)
end_time = time.time()

# 4. Report Finale
print(f"\n{'-'*40}")
if solution is not None:
    print(f"✅ SUCCESSO!")
    print(f"Lunghezza soluzione trovata: {len(solution)} mosse")
    print(f"Mosse: {solution}")
    print(f"Tempo totale: {end_time - start_time:.2f} secondi")
    print(f"Nodi totali esplorati: {total_nodes}")
    print(f"Efficienza: {int(total_nodes / (end_time - start_time))} nodi/secondo")
else:
    print("❌ FALLIMENTO: Il cubo non è stato risolto entro i timeout previsti.")
print(f"{'-'*40}")