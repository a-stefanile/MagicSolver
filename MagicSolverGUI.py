import tkinter as tk
from tkinter import messagebox
import kociemba
import random
import os
import datetime
from PIL import Image, ImageGrab
import time


class RubiksAI:
    def __init__(self, root):
        self.root = root
        self.root.title("MagicSolver")
        self.root.geometry("850x750")
        self.root.configure(bg="#2c3e50")

        self.init_cube_data()
        self.selected_color = "white"
        self.solution_moves = []
        self.snapshot_count = 0
        self.base_output_dir = "cubo_snapshots"
        self.current_session_dir = ""

        self.setup_ui()

    def init_cube_data(self):
        self.faces = {
            'U': [['white'] * 3 for _ in range(3)], 'D': [['yellow'] * 3 for _ in range(3)],
            'L': [['orange'] * 3 for _ in range(3)], 'R': [['red'] * 3 for _ in range(3)],
            'F': [['green'] * 3 for _ in range(3)], 'B': [['blue'] * 3 for _ in range(3)]
        }

    def setup_ui(self):
        # --- TOP: PALETTE ---
        top_frame = tk.LabelFrame(self.root, text=" 1. Colore ", bg="#34495e", fg="white", padx=10, pady=5)
        top_frame.pack(pady=10)
        for c in ["white", "orange", "green", "red", "blue", "yellow"]:
            tk.Button(top_frame, bg=c, width=4, height=2, relief=tk.RAISED,
                      command=lambda col=c: self.set_color(col)).pack(side=tk.LEFT, padx=3)

        # --- CENTER: CANVAS (Layout a croce) ---
        self.canvas = tk.Canvas(self.root, width=600, height=450, bg="#ecf0f1", highlightthickness=0)
        self.canvas.pack(pady=5)

        # --- MIDDLE: MOSSE MANUALI ---
        manual_frame = tk.LabelFrame(self.root, text=" 2. Mosse Manuali ", bg="#34495e", fg="white", padx=10, pady=5)
        manual_frame.pack(pady=5)

        row1 = tk.Frame(manual_frame, bg="#34495e")
        row1.pack()
        # Mosse orarie e antiorarie
        for m in ['U', 'D', 'L', 'R', 'F', 'B']:
            tk.Button(row1, text=m, width=4, command=lambda x=m: self.apply_move(x)).pack(side=tk.LEFT, padx=2)
            tk.Button(row1, text=f"{m}'", width=4, bg="#bdc3c7", command=lambda x=f"{m}'": self.apply_move(x)).pack(
                side=tk.LEFT, padx=2)

        # --- BOTTOM: CONTROLLI IA ---
        bot_frame = tk.Frame(self.root, bg="#2c3e50")
        bot_frame.pack(pady=15)

        btn_style = {"font": ("Arial", 10, "bold"), "padx": 15, "pady": 8}
        tk.Button(bot_frame, text="MESCOLA", bg="#f1c40f", **btn_style, command=self.scramble_cube).pack(side=tk.LEFT,
                                                                                                         padx=5)
        tk.Button(bot_frame, text="VERIFICA", bg="#3498db", fg="white", **btn_style, command=self.verify_cube).pack(
            side=tk.LEFT, padx=5)
        self.play_btn = tk.Button(bot_frame, text="RISOLVI & SNAPSHOT", bg="#2ecc71", fg="white",
                                  state=tk.DISABLED, **btn_style, command=self.start_solving_process)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(bot_frame, text="RESET", bg="#e74c3c", fg="white", **btn_style, command=self.reset_ui).pack(
            side=tk.LEFT, padx=5)

        self.draw_cube()

    def set_color(self, color):
        self.selected_color = color

    def draw_cube(self, current_move="Stato Attuale"):
        self.canvas.delete("all")

        # Testo posizionato sul fondo del canvas con margine
        self.canvas.create_text(300, 430, text=f"MOSSA: {current_move}", font=("Arial", 14, "bold"), fill="#2c3e50")

        # COORDINATE A CROCE (Pezzi adiacenti)
        size = 40  # Dimensione quadratino + bordo
        # Offset (X, Y) per ogni faccia
        offsets = {
            'U': (size * 3 + 10, 20), # BIANCO
            'L': (10, size * 3 + 20), # ARANCIO
            'F': (size * 3 + 10, size * 3 + 20), # VERDE
            'R': (size * 6 + 10, size * 3 + 20), # ROSSO
            'B': (size * 9 + 10, size * 3 + 20), # BLU
            'D': (size * 3 + 10, size * 6 + 20) # GIALLO
        }

        for face, (ox, oy) in offsets.items():
            for r in range(3):
                for c in range(3):
                    color = self.faces[face][r][c]
                    x1, y1 = ox + c * size, oy + r * size
                    rect = self.canvas.create_rectangle(x1, y1, x1 + (size - 2), y1 + (size - 2), fill=color,
                                                        outline="#2c3e50", width=1)
                    self.canvas.tag_bind(rect, "<Button-1>", lambda e, f=face, row=r, col=c: self.paint(f, row, col))

    def paint(self, f, r, c):
        self.faces[f][r][c] = self.selected_color
        self.draw_cube()

    def take_snapshot(self, filename):
        """Cattura il canvas e lo salva nella sottocartella della sessione attuale"""
        self.root.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab(bbox=(x, y, x1, y1))
        img.save(os.path.join(self.current_session_dir, f"{self.snapshot_count:02d}_{filename}.png"))
        self.snapshot_count += 1

    # --- LOGICA ROTAZIONI (Standard) ---
    def rotate_face_data(self, face):
        f = self.faces[face]
        self.faces[face] = [[f[2][0], f[1][0], f[0][0]], [f[2][1], f[1][1], f[0][1]], [f[2][2], f[1][2], f[0][2]]]

    def move_U(self):
        self.rotate_face_data('U')
        f, r, b, l = self.faces['F'][0], self.faces['R'][0], self.faces['B'][0], self.faces['L'][0]
        self.faces['F'][0], self.faces['R'][0], self.faces['B'][0], self.faces['L'][0] = r, b, l, f

    def move_D(self):
        self.rotate_face_data('D')
        f, r, b, l = self.faces['F'][2], self.faces['R'][2], self.faces['B'][2], self.faces['L'][2]
        self.faces['F'][2], self.faces['R'][2], self.faces['B'][2], self.faces['L'][2] = l, f, r, b

    def move_R(self):
        self.rotate_face_data('R')
        for i in range(3):
            t = self.faces['F'][i][2];
            self.faces['F'][i][2] = self.faces['D'][i][2]
            self.faces['D'][i][2] = self.faces['B'][2 - i][0];
            self.faces['B'][2 - i][0] = self.faces['U'][i][2]
            self.faces['U'][i][2] = t

    def move_L(self):
        self.rotate_face_data('L')
        for i in range(3):
            t = self.faces['F'][i][0];
            self.faces['F'][i][0] = self.faces['U'][i][0]
            self.faces['U'][i][0] = self.faces['B'][2 - i][2];
            self.faces['B'][2 - i][2] = self.faces['D'][i][0]
            self.faces['D'][i][0] = t

    def move_F(self):
        self.rotate_face_data('F')
        for i in range(3):
            t = self.faces['U'][2][i];
            self.faces['U'][2][i] = self.faces['L'][2 - i][2]
            self.faces['L'][2 - i][2] = self.faces['D'][0][2 - i];
            self.faces['D'][0][2 - i] = self.faces['R'][i][0]
            self.faces['R'][i][0] = t

    def move_B(self):
        self.rotate_face_data('B')
        for i in range(3):
            t = self.faces['U'][0][i];
            self.faces['U'][0][i] = self.faces['R'][i][2]
            self.faces['R'][i][2] = self.faces['D'][2][2 - i];
            self.faces['D'][2][2 - i] = self.faces['L'][2 - i][0]
            self.faces['L'][2 - i][0] = t

    def apply_move(self, move):
        m = move[0]
        times = 3 if "'" in move else (2 if "2" in move else 1)
        for _ in range(times): getattr(self, f"move_{m}")()
        self.draw_cube(move)

    # --- RISOLUZIONE E SNAPSHOT ---
    def start_solving_process(self):
        # 1. Crea sottocartella specifica per questa sessione
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session_dir = os.path.join(self.base_output_dir, f"session_{timestamp}")
        if not os.path.exists(self.current_session_dir):
            os.makedirs(self.current_session_dir)

        self.snapshot_count = 0
        self.draw_cube("Inizio")
        self.take_snapshot("00_stato_iniziale")
        self.play_solution(0)

    def play_solution(self, index):
        if index < len(self.solution_moves):
            move = self.solution_moves[index]
            self.apply_move(move)

            # Pulizia nome file
            clean_name = move.replace("'", "primo")
            self.take_snapshot(f"mossa_{clean_name}")

            self.root.after(800, lambda: self.play_solution(index + 1))
        else:
            self.draw_cube("Risolto")
            self.take_snapshot("99_stato_finale")
            messagebox.showinfo("Completato", f"Foto salvate in:\n{self.current_session_dir}")

    def verify_cube(self):
        try:
            centers = {self.faces[f][1][1]: f for f in ['U', 'R', 'F', 'D', 'L', 'B']}
            s = "".join(centers[self.faces[f][r][c]] for f in ['U', 'R', 'F', 'D', 'L', 'B'] for r in range(3) for c in
                        range(3))
            self.solution_moves = kociemba.solve(s).split()
            self.play_btn.config(state=tk.NORMAL)
            messagebox.showinfo("Successo", f"Cubo pronto! {len(self.solution_moves)} mosse.")
        except:
            messagebox.showerror("Errore", "Configurazione non valida.")

    def scramble_cube(self):
        m_list = ['U', 'D', 'L', 'R', 'F', 'B']
        for _ in range(15): self.apply_move(random.choice(m_list))
        self.draw_cube("Mescolato")

    def reset_ui(self):
        self.init_cube_data()
        self.play_btn.config(state=tk.DISABLED)
        self.draw_cube()


if __name__ == "__main__":
    root = tk.Tk()
    app = RubiksAI(root)
    root.mainloop()