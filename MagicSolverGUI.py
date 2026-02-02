import tkinter as tk
from tkinter import messagebox
import os
import datetime
from PIL import ImageGrab
from RubiksCube import RubiksCube
from Solver import RubiksSolver


class RubiksAI:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicSolver")
        self.root.geometry("850x800")
        self.root.configure(bg="#2c3e50")

        # Inizializzazione
        self.init_cube_data()
        try:
            self.ai_solver = RubiksSolver(pipeline='OHE')
        except Exception as e:
            print(f"Avviso: Modello IA non trovato. Esegui prima il training! {e}")
            self.ai_solver = None

        self.selected_color = "white"
        self.solution_moves = []
        self.snapshot_count = 0
        self.base_output_dir = "cubo_snapshots"

        self.color_map_logic_to_gui = {
            'w': 'white', 'y': 'yellow', 'g': 'green',
            'b': 'blue', 'r': 'red', 'o': 'orange', '': 'gray'
        }
        self.color_map_gui_to_logic = {v: k for k, v in self.color_map_logic_to_gui.items() if k != ''}

        self.setup_ui()

    def init_cube_data(self):
        """Inizializza l'oggetto logico RubiksCube del progetto."""
        self.cube_logic = RubiksCube()

    def setup_ui(self):
        # --- TOP: PALETTE ---
        top_frame = tk.LabelFrame(self.root, text=" 1. Seleziona Colore ", bg="#34495e", fg="white", padx=10, pady=5)
        top_frame.pack(pady=10)
        for c in ["white", "orange", "green", "red", "blue", "yellow"]:
            tk.Button(top_frame, bg=c, width=4, height=2, relief=tk.RAISED,
                      command=lambda col=c: self.set_color(col)).pack(side=tk.LEFT, padx=3)

        # --- CENTER: CANVAS ---
        self.canvas = tk.Canvas(self.root, width=650, height=480, bg="#ecf0f1", highlightthickness=0)
        self.canvas.pack(pady=5)

        # --- MIDDLE: MOSSE MANUALI ---
        manual_frame = tk.LabelFrame(self.root, text=" 2. Rotazioni Facce ", bg="#34495e", fg="white", padx=10, pady=5)
        manual_frame.pack(pady=5)
        row1 = tk.Frame(manual_frame, bg="#34495e")
        row1.pack()

        # Mosse GUI standard
        self.gui_moves = ['U', 'D', 'L', 'R', 'F', 'B']
        for m in self.gui_moves:
            tk.Button(row1, text=m, width=5, command=lambda x=m: self.apply_move_gui(x)).pack(side=tk.LEFT, padx=2)
            tk.Button(row1, text=f"{m}'", width=5, bg="#bdc3c7", command=lambda x=f"{m}'": self.apply_move_gui(x)).pack(
                side=tk.LEFT, padx=2)

        # --- BOTTOM: CONTROLLI IA ---
        bot_frame = tk.Frame(self.root, bg="#2c3e50")
        bot_frame.pack(pady=15)

        btn_style = {"font": ("Arial", 10, "bold"), "padx": 15, "pady": 8}
        tk.Button(bot_frame, text="MESCOLA", bg="#f1c40f", **btn_style, command=self.scramble_cube).pack(side=tk.LEFT,
                                                                                                         padx=5)
        tk.Button(bot_frame, text="VERIFICA", bg="#3498db", fg="white", **btn_style,
                  command=self.solve_with_ai).pack(side=tk.LEFT, padx=5)
        self.play_btn = tk.Button(bot_frame, text="ESEGUI SOLUZIONE", bg="#2ecc71", fg="white", state=tk.DISABLED,
                                  **btn_style, command=self.start_solving_process)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(bot_frame, text="RESET", bg="#e74c3c", fg="white", **btn_style, command=self.reset_ui).pack(
            side=tk.LEFT, padx=5)

        self.draw_cube()

    def set_color(self, color):
        self.selected_color = color

    def draw_cube(self, current_move="Stato Attuale"):
        self.canvas.delete("all")
        self.canvas.create_text(325, 450, text=f"ULTIMA AZIONE: {current_move}", font=("Arial", 12, "bold"),
                                fill="#2c3e50")

        size = 35
        offsets = {
            'U': (size * 3 + 20, 40),
            'L': (20, size * 3 + 40),
            'F': (size * 3 + 20, size * 3 + 40),
            'R': (size * 6 + 20, size * 3 + 40),
            'B': (size * 9 + 20, size * 3 + 40),
            'D': (size * 3 + 19, size * 6 + 40)
        }

        face_data = {
            'U': self.cube_logic.cube[0, 1:4, 1:4],
            'D': self.cube_logic.cube[4, 1:4, 1:4],
            'F': self.cube_logic.cube[1:4, 0, 1:4],
            'B': self.cube_logic.cube[1:4, 4, 1:4],
            'L': self.cube_logic.cube[1:4, 1:4, 0],
            'R': self.cube_logic.cube[1:4, 1:4, 4]
        }

        for face, (ox, oy) in offsets.items():
            data = face_data[face]
            for r in range(3):
                for c in range(3):
                    logic_color = data[r, c]
                    gui_color = self.color_map_logic_to_gui.get(logic_color, 'gray')
                    x1, y1 = ox + c * size, oy + r * size
                    rect = self.canvas.create_rectangle(x1, y1, x1 + (size - 2), y1 + (size - 2),
                                                        fill=gui_color, outline="#2c3e50")
                    self.canvas.tag_bind(rect, "<Button-1>",
                                         lambda e, f=face, row=r, col=c: self.paint_sticker(f, row, col))

    def paint_sticker(self, face, r, c):
        """Modifica un singolo sticker nel tensore logico."""
        new_val = self.color_map_gui_to_logic[self.selected_color]
        if face == 'U':
            self.cube_logic.cube[0, 1 + r, 1 + c] = new_val
        elif face == 'D':
            self.cube_logic.cube[4, 1 + r, 1 + c] = new_val
        elif face == 'F':
            self.cube_logic.cube[1 + r, 0, 1 + c] = new_val
        elif face == 'B':
            self.cube_logic.cube[1 + r, 4, 1 + c] = new_val
        elif face == 'L':
            self.cube_logic.cube[1 + r, 1 + c, 0] = new_val
        elif face == 'R':
            self.cube_logic.cube[1 + r, 1 + c, 4] = new_val
        self.draw_cube()

    def apply_move_gui(self, move):
        """Traduce le mosse GUI (U, D...) in mosse RubiksCube (top, bottom...)."""
        mapping = {'U': 'top', 'D': 'bottom', 'L': 'left', 'R': 'right', 'F': 'front', 'B': 'back'}
        face_logic = mapping[move[0]]
        is_rev = "'" in move
        self.cube_logic.rotate_face(face_logic, reverse=is_rev)
        self.draw_cube(move)

    def validate_cube_reality(self):
        """Verifica se la configurazione attuale del cubo è fisicamente possibile."""
        all_stickers = []
        face_data = {
            'U': self.cube_logic.cube[0, 1:4, 1:4], 'D': self.cube_logic.cube[4, 1:4, 1:4],
            'F': self.cube_logic.cube[1:4, 0, 1:4], 'B': self.cube_logic.cube[1:4, 4, 1:4],
            'L': self.cube_logic.cube[1:4, 1:4, 0], 'R': self.cube_logic.cube[1:4, 1:4, 4]
        }
        for data in face_data.values():
            all_stickers.extend(data.flatten().tolist())

        # 2. Controllo Conteggio (9 per colore)
        for color_code in ['w', 'y', 'g', 'b', 'r', 'o']:
            count = all_stickers.count(color_code)
            if count != 9:
                color_name = self.color_map_logic_to_gui[color_code]
                messagebox.showerror("Cubo non valido!",
                                     f"Il colore {color_name} appare {count} volte. Deve apparire esattamente 9 volte.")
                return False

        # 3. Controllo Centri (Devono essere fissi)
        centers = {
            "Top (Bianco)": self.cube_logic.cube[0, 2, 2], "Bottom (Giallo)": self.cube_logic.cube[4, 2, 2],
            "Front (Verde)": self.cube_logic.cube[2, 0, 2], "Back (Blu)": self.cube_logic.cube[2, 4, 2],
            "Left (Rosso)": self.cube_logic.cube[2, 2, 0], "Right (Arancio)": self.cube_logic.cube[2, 2, 4]
        }
        expected_centers = {'Top (Bianco)': 'w', 'Bottom (Giallo)': 'y', 'Front (Verde)': 'g',
                            'Back (Blu)': 'b', 'Left (Rosso)': 'r', 'Right (Arancio)': 'o'}

        for name, color in centers.items():
            if color != expected_centers[name]:
                messagebox.showerror("Cubo non valido!",
                                     f"Il centro della faccia {name} è errato. I centri non possono cambiare posizione!")
                return False

        opposites = [('w', 'y'), ('g', 'b'), ('r', 'o')]

        for c1, c2 in opposites:
            pass

        return True

    def solve_with_ai(self):
        """Utilizza il RubiksSolver per trovare la soluzione."""
        if not self.ai_solver:
            messagebox.showerror("Errore", "Modello IA non caricato. Controlla i file .joblib")
            return

        if not self.validate_cube_reality():
            return

        if self.cube_logic.is_solved():
            messagebox.showinfo("Info", "Il cubo è già risolto!")
            return

        # L'IA risolve il cubo attuale
        solution, nodes = self.ai_solver.solve_adaptive_ultra(self.cube_logic)

        if solution:
            self.solution_moves = solution  # Formato: [('top', False), ...]
            self.play_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Successo",
                                f"IA ha trovato una soluzione in {len(solution)} mosse!\nNodi esplorati: {nodes}")
        else:
            messagebox.showerror("IA Fallita", "L'IA non è riuscita a risolvere questa configurazione.")

    def start_solving_process(self):
        # Crea cartella snapshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = os.path.join(self.base_output_dir, f"ai_session_{timestamp}")
        if not os.path.exists(self.current_session_dir): os.makedirs(self.current_session_dir)

        self.snapshot_count = 0
        self.play_solution(0)

    def play_solution(self, index):
        if index < len(self.solution_moves):
            move_tuple = self.solution_moves[index]  # ('face', reverse)
            self.cube_logic.rotate_face(move_tuple[0], reverse=move_tuple[1])

            # Label per la mossa
            move_label = f"{move_tuple[0]} {' (REV)' if move_tuple[1] else ''}"
            self.draw_cube(move_label)

            # Screenshot
            self.take_snapshot(f"step_{index}")

            self.root.after(800, lambda: self.play_solution(index + 1))
        else:
            self.draw_cube("RISOLTO DALL'IA")
            self.play_btn.config(state=tk.DISABLED)

    def take_snapshot(self, filename):
        self.root.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab(bbox=(x, y, x1, y1)).save(os.path.join(self.current_session_dir, f"{filename}.png"))

    def scramble_cube(self):
        moves = self.cube_logic.scramble(n=11) # Numero di scramble
        self.draw_cube("MESCOLATO")
        self.play_btn.config(state=tk.DISABLED)

    def reset_ui(self):
        self.init_cube_data()
        self.play_btn.config(state=tk.DISABLED)
        self.draw_cube("RESET")


if __name__ == "__main__":
    root = tk.Tk()
    app = RubiksAI(root)
    root.mainloop()