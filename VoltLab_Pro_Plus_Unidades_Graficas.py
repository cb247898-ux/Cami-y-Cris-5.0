# -*- coding: utf-8 -*-
"""
VoltLab Pro — simulador educativo de circuitos AC
Ejecutar en Spyder con F5.

Instalar dependencias en la consola de Spyder:
    %pip install numpy matplotlib

Funciones:
- Circuitos R, RL, RC y RLC en serie o paralelo.
- Fuente sinusoidal configurable: Vp, frecuencia y fase.
- Fasores rectangulares y polares.
- Resultados paso a paso estilo cuaderno.
- Corrientes y voltajes por elemento.
- Potencias P, Q, S y factor de potencia.
- Interpretación automática del resultado.
- Gráficas temporales, fasores, triángulo de impedancia/potencia
  y esquema eléctrico.
- Exportación del desarrollo a TXT y resultados a CSV.

Modelo ideal en régimen sinusoidal permanente. No es un reemplazo de
un simulador SPICE para transitorios, semiconductores o circuitos arbitrarios.
"""

import csv
import math
import cmath
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------- Utilidades de fasores ----------

def fmt_num(value, digits=4):
    if abs(value) < 1e-10:
        value = 0.0
    return f"{value:.{digits}g}"


def polar(z):
    return f"{abs(z):.4f} ∠ {math.degrees(cmath.phase(z)):.3f}°"


def rectangular(z):
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:.4f} {sign} j{abs(z.imag):.4f}"


def angle(z):
    return math.degrees(cmath.phase(z))


def angle_phrase(a):
    sign = "+" if a >= 0 else "−"
    return f"{sign} {abs(a):.3f}°"


def parse_quantity(raw, dimension):
    """Acepta prefijos/sufijos SI: 4.7kΩ, 20mH, 100uF, 12V, 60Hz, 1.2rad."""
    text = str(raw).strip().replace("µ", "u").replace("μ", "u")
    text = text.replace("Ω", "ohm").replace("°", "deg").replace(",", ".")
    text = text.replace(" ", "")
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)", text)
    if not match:
        raise ValueError(
            f"Valor no válido: '{raw}'. Ejemplos: 4.7kΩ, 20mH, 100uF, 12V, 60Hz o 1.2rad."
        )
    number, tail = float(match.group(1)), match.group(2)
    if not math.isfinite(number):
        raise ValueError("Introduce valores numéricos finitos.")

    if dimension == "phase":
        low = tail.lower()
        if low in ("", "deg", "degree", "degrees", "grado", "grados"):
            return number
        if low in ("rad", "radian", "radians"):
            return math.degrees(number)
        raise ValueError("La fase acepta grados (deg, °) o radianes (rad).")

    units = {
        "R": ("ohms", "ohm", "Ω"),
        "L": ("henries", "henry", "h"),
        "C": ("farads", "farad", "f"),
        "V": ("volts", "volt", "v"),
        "f": ("hertz", "hz"),
    }
    prefixes = {
        "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
        "": 1.0, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9
    }

    # El sufijo de unidad se detecta sin distinguir mayúsculas; el prefijo sí distingue m/M.
    unit_found = None
    lower_tail = tail.lower()
    for unit in sorted(units[dimension], key=len, reverse=True):
        if lower_tail.endswith(unit.lower()):
            unit_found = unit
            prefix = tail[:-len(unit)]
            break
    else:
        # También permite escribir solo el prefijo, porque la unidad aparece en la etiqueta.
        prefix = tail

    if prefix not in prefixes:
        raise ValueError(
            f"Unidad/prefijo no reconocido en '{raw}'. Prefijos admitidos: p, n, u/µ, m, k, M y G."
        )
    return number * prefixes[prefix]

def format_eng(value, unit="", sig=4):
    """Devuelve una magnitud con prefijo de ingeniería SI."""
    value = float(value)
    if not math.isfinite(value):
        return f"{value} {unit}".strip()
    if abs(value) < 1e-30:
        return f"0 {unit}".strip()
    prefixes = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
                3: "k", 6: "M", 9: "G", 12: "T"}
    exponent = max(-12, min(12, int(math.floor(math.log10(abs(value)) / 3) * 3)))
    return f"{value / (10 ** exponent):.{sig}g} {prefixes[exponent]}{unit}".strip()


def safe_complex_div(a, b):
    if abs(b) < 1e-12:
        raise ValueError("La división produciría un valor infinito. Revisa los componentes.")
    return a / b


# ---------- Motor de cálculo ----------

def solve_circuit(config):
    topology = config["topology"]
    kind = config["kind"]
    R, L, C = config["R"], config["L"], config["C"]
    Vp, f, phase_deg = config["Vp"], config["f"], config["phase"]
    w = 2 * math.pi * f
    phase_rad = math.radians(phase_deg)
    Vs = Vp * cmath.exp(1j * phase_rad)

    has_R = kind in ("R", "RL", "RC", "RLC")
    has_L = kind in ("L", "RL", "RLC")
    has_C = kind in ("C", "RC", "RLC")
    if has_R and R <= 0:
        raise ValueError("R debe ser mayor que cero para este circuito.")
    if has_L and L <= 0:
        raise ValueError("L debe ser mayor que cero para este circuito.")
    if has_C and C <= 0:
        raise ValueError("C debe ser mayor que cero para este circuito.")

    ZR = complex(R, 0) if has_R else None
    ZL = complex(0, w * L) if has_L else None
    ZC = complex(0, -1 / (w * C)) if has_C else None
    impedances = {}
    if has_R:
        impedances["R"] = ZR
    if has_L:
        impedances["L"] = ZL
    if has_C:
        impedances["C"] = ZC

    if topology == "Serie":
        Zt = sum(impedances.values(), 0j)
        It = safe_complex_div(Vs, Zt)
        voltages = {key: It * z for key, z in impedances.items()}
        currents = {key: It for key in impedances}
        # Potencia compleja de carga: S = V * conjugado(I) / 2, usando fasores pico
        Scomplex = 0.5 * Vs * It.conjugate()
        method_lines = [
            "1) Calculamos las reactancias:",
            f"   ω = 2πf = 2π({f:g}) = {w:.4f} rad/s",
        ]
        if has_L:
            method_lines.append(f"   XL = ωL = {w:.4f} × {L:g} = {w*L:.4f} Ω")
        if has_C:
            method_lines.append(f"   XC = 1/(ωC) = 1/({w:.4f} × {C:g}) = {1/(w*C):.4f} Ω")
        method_lines += [
            "",
            "2) Impedancia total en serie:",
            "   ZT = suma de las impedancias de los elementos",
            f"   ZT = {rectangular(Zt)} Ω",
            f"   |ZT| = {abs(Zt):.4f} Ω",
            f"   θZ = {angle(Zt):.3f}°",
            f"   ZT = {polar(Zt)} Ω",
            "",
            "3) Corriente total usando la ley de Ohm fasorial:",
            "   I = Vs / ZT",
            f"   I = ({polar(Vs)} V) / ({polar(Zt)} Ω)",
            f"   I = {polar(It)} A",
            f"   I rectangular = {rectangular(It)} A",
        ]
    else:
        admittances = {key: safe_complex_div(1+0j, z) for key, z in impedances.items()}
        Yt = sum(admittances.values(), 0j)
        It = Vs * Yt
        Zt = safe_complex_div(1+0j, Yt)
        voltages = {key: Vs for key in impedances}
        currents = {key: safe_complex_div(Vs, z) for key, z in impedances.items()}
        Scomplex = sum((0.5 * voltages[k] * currents[k].conjugate() for k in currents), 0j)
        method_lines = [
            "1) Calculamos las impedancias de cada rama:",
            f"   ω = 2πf = 2π({f:g}) = {w:.4f} rad/s",
        ]
        if has_L:
            method_lines.append(f"   XL = ωL = {w*L:.4f} Ω; ZL = j{w*L:.4f} Ω")
        if has_C:
            method_lines.append(f"   XC = 1/(ωC) = {1/(w*C):.4f} Ω; ZC = −j{1/(w*C):.4f} Ω")
        if has_R:
            method_lines.append(f"   ZR = {R:.4f} Ω")
        method_lines += [
            "",
            "2) Admitancia total en paralelo:",
            "   YT = suma de las admitancias de las ramas, Y = 1/Z",
            f"   YT = {rectangular(Yt)} S",
            f"   |YT| = {abs(Yt):.4f} S; ángulo = {angle(Yt):.3f}°",
            "",
            "3) Impedancia equivalente:",
            "   ZT = 1 / YT",
            f"   ZT = {rectangular(Zt)} Ω",
            f"   ZT = {polar(Zt)} Ω",
            "",
            "4) Corriente total:",
            "   IT = Vs × YT = suma de las corrientes de las ramas",
            f"   IT = {polar(It)} A",
            f"   IT rectangular = {rectangular(It)} A",
        ]

    Vrms = Vp / math.sqrt(2)
    Irms = abs(It) / math.sqrt(2)
    P = Scomplex.real
    Q = Scomplex.imag
    Sapp = abs(Scomplex)
    pf = 0.0 if Sapp < 1e-12 else max(-1.0, min(1.0, P / Sapp))
    phi = angle(Zt)

    method_lines += ["", "5) Voltajes y corrientes de los elementos:"]
    for key in impedances:
        v = voltages[key]
        i = currents[key]
        names = {"R": "Resistencia", "L": "Bobina", "C": "Capacitor"}
        method_lines += [
            f"   {names[key]}:",
            f"      V{key} = {polar(v)} V",
            f"      I{key} = {polar(i)} A",
        ]

    method_lines += [
        "",
        "6) Valores eficaces y potencias:",
        f"   Vs,rms = Vp/√2 = {Vrms:.4f} V",
        f"   IT,rms = |IT|/√2 = {Irms:.4f} A",
        f"   P = potencia activa = {P:.4f} W",
        f"   Q = potencia reactiva = {Q:.4f} var",
        f"   S = potencia aparente = {Sapp:.4f} VA",
        f"   FP = P/S = {pf:.4f}",
    ]

    if abs(Q) < 1e-7:
        interpretation = "Comportamiento predominantemente resistivo: voltaje y corriente están aproximadamente en fase."
    elif Q > 0:
        interpretation = "Comportamiento inductivo: la corriente total se retrasa respecto al voltaje."
    else:
        interpretation = "Comportamiento capacitivo: la corriente total se adelanta respecto al voltaje."
    if has_L and has_C:
        if abs(w * L - 1/(w*C)) / max(w*L, 1/(w*C)) < 0.05:
            interpretation += " El circuito está cerca de la resonancia (XL ≈ XC)."
    method_lines += ["", "7) Interpretación:", "   " + interpretation]

    # Convención de señales temporales: coseno.
    t = np.linspace(0, 3/f, 1500)
    signals = {"t": t, "Vs": Vp*np.cos(w*t + phase_rad)}
    for key, v in voltages.items():
        signals["V"+key] = abs(v)*np.cos(w*t + cmath.phase(v))
    for key, i in currents.items():
        signals["I"+key] = abs(i)*np.cos(w*t + cmath.phase(i))
    signals["IT"] = abs(It)*np.cos(w*t + cmath.phase(It))

    return {
        **config, "w": w, "Vs_phasor": Vs, "impedances": impedances,
        "Zt": Zt, "It": It, "voltages": voltages, "currents": currents,
        "Vrms": Vrms, "Irms": Irms, "P": P, "Q": Q, "S": Sapp,
        "pf": pf, "phi": phi, "interpretation": interpretation,
        "steps": method_lines, "signals": signals, "has_R": has_R,
        "has_L": has_L, "has_C": has_C
    }


# ---------- Interfaz ----------

class VoltLabPro:
    BG = "#0b1220"
    PANEL = "#132238"
    FIELD = "#091525"
    TEXT = "#eef5ff"
    MUTED = "#a9bbd1"
    ACCENT = "#43dfc0"
    BLUE = "#79b4ff"
    ORANGE = "#ffbd75"
    BORDER = "#2c405c"

    def __init__(self, root):
        self.root = root
        root.title("VoltLab Pro | Simulador de circuitos AC")
        root.geometry("1280x850")
        root.minsize(1000, 700)
        root.configure(bg=self.BG)
        self.last = None
        self._style()
        self._build()
        self.calculate()

    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("TNotebook", background=self.BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=self.PANEL, foreground=self.TEXT,
                     padding=(13, 8), font=("Segoe UI", 9, "bold"))
        st.map("TNotebook.Tab", background=[("selected", "#24534f")],
               foreground=[("selected", self.TEXT)])
        st.configure("TCombobox", fieldbackground=self.FIELD, background=self.PANEL,
                     foreground=self.TEXT, arrowcolor=self.TEXT)

    def _build(self):
        header = tk.Frame(self.root, bg=self.BG, padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="⚡ VOLTLAB PRO", bg=self.BG, fg=self.ACCENT,
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="SIMULACIÓN AC  •  FASORES  •  ANÁLISIS  •  GRÁFICAS",
                 bg=self.BG, fg=self.MUTED, font=("Segoe UI", 9, "bold")).pack(side="right")

        main = tk.Frame(self.root, bg=self.BG, padx=12, pady=4)
        main.pack(fill="both", expand=True)
        side = tk.Frame(main, bg=self.PANEL, padx=15, pady=14,
                        highlightbackground=self.BORDER, highlightthickness=1, width=270)
        side.pack(side="left", fill="y", padx=(0, 10))
        side.pack_propagate(False)
        tk.Label(side, text="CONFIGURAR CIRCUITO", bg=self.PANEL, fg=self.ACCENT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 10))

        self.vars = {
            "topology": tk.StringVar(value="Serie"),
            "kind": tk.StringVar(value="RLC"),
            "R": tk.StringVar(value="4"),
            "L": tk.StringVar(value="0.2"),
            "C": tk.StringVar(value="0.0001"),
            "Vp": tk.StringVar(value="20"),
            "f": tk.StringVar(value="60"),
            "phase": tk.StringVar(value="0"),
        }
        self._label(side, "Conexión")
        cb = ttk.Combobox(side, textvariable=self.vars["topology"], state="readonly",
                          values=["Serie", "Paralelo"])
        cb.pack(fill="x", pady=(0, 8))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.calculate())

        self._label(side, "Elementos incluidos")
        cb2 = ttk.Combobox(side, textvariable=self.vars["kind"], state="readonly",
                           values=["R", "RL", "RC", "RLC", "L", "C"])
        cb2.pack(fill="x", pady=(0, 8))
        cb2.bind("<<ComboboxSelected>>", lambda _e: self.calculate())

        for key, label in [
            ("R", "Resistencia R (Ω)"), ("L", "Inductancia L (H)"),
            ("C", "Capacitancia C (F)"), ("Vp", "Voltaje pico Vp (V)"),
            ("f", "Frecuencia f (Hz)"), ("phase", "Fase fuente (° o rad)")
        ]:
            self._label(side, label)
            ent = tk.Entry(side, textvariable=self.vars[key], bg=self.FIELD, fg=self.TEXT,
                           insertbackground="white", relief="flat", font=("Segoe UI", 10),
                           highlightthickness=1, highlightbackground=self.BORDER,
                           highlightcolor=self.ACCENT)
            ent.pack(fill="x", ipady=6, pady=(0, 7))
            ent.bind("<Return>", lambda _e: self.calculate())

        tk.Button(side, text="▶  SIMULAR CIRCUITO", command=self.calculate,
                  bg=self.ACCENT, fg="#05241e", activebackground="#7cf2dc",
                  relief="flat", font=("Segoe UI", 10, "bold"), pady=9,
                  cursor="hand2").pack(fill="x", pady=(5, 7))
        tk.Button(side, text="Cargar ejemplo del cuaderno (RL)", command=self.example_rl,
                  bg="#29415e", fg=self.TEXT, relief="flat", pady=7,
                  cursor="hand2").pack(fill="x", pady=3)
        tk.Button(side, text="Cargar ejemplo RC", command=self.example_rc,
                  bg="#29415e", fg=self.TEXT, relief="flat", pady=7,
                  cursor="hand2").pack(fill="x", pady=3)
        tk.Button(side, text="Exportar desarrollo TXT", command=self.export_txt,
                  bg="#29415e", fg=self.TEXT, relief="flat", pady=7,
                  cursor="hand2").pack(fill="x", pady=3)
        tk.Button(side, text="Exportar datos CSV", command=self.export_csv,
                  bg="#29415e", fg=self.TEXT, relief="flat", pady=7,
                  cursor="hand2").pack(fill="x", pady=3)
        tk.Label(side, text="Acepta prefijos y sufijos: 4.7kΩ, 15mH, 220µF, 20V, 60Hz, -30deg o 1.2rad.\nPrefijos: p, n, u/µ, m, k, M, G.\nPuedes omitir la unidad si el campo ya la indica.\n\nFasores de pico; potencias RMS.",
                 bg=self.PANEL, fg=self.MUTED, justify="left", wraplength=230,
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(12, 0))

        right = tk.Frame(main, bg=self.BG)
        right.pack(side="left", fill="both", expand=True)
        self.cards = tk.Frame(right, bg=self.BG)
        self.cards.pack(fill="x", pady=(0, 9))
        self.card_values = {}
        for key, title in [("z", "IMPEDANCIA TOTAL"), ("i", "CORRIENTE PICO"),
                           ("phi", "ÁNGULO Z"), ("pf", "FACTOR POTENCIA")]:
            card = tk.Frame(self.cards, bg=self.PANEL, padx=10, pady=10,
                            highlightbackground=self.BORDER, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0, 7))
            tk.Label(card, text=title, bg=self.PANEL, fg=self.MUTED,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            val = tk.Label(card, text="—", bg=self.PANEL, fg=self.ACCENT,
                           font=("Segoe UI", 14, "bold"))
            val.pack(anchor="w", pady=(4, 0))
            self.card_values[key] = val

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        tab_names = ["Desarrollo paso a paso", "Gráficas", "Fasores", "Esquema", "Interpretación"]
        self.tab_frames = []
        for name in tab_names:
            frame = tk.Frame(self.tabs, bg=self.PANEL)
            self.tabs.add(frame, text="  " + name + "  ")
            self.tab_frames.append(frame)

        self.report = tk.Text(self.tab_frames[0], bg=self.FIELD, fg=self.TEXT,
                              insertbackground="white", font=("Consolas", 10),
                              wrap="word", relief="flat", padx=15, pady=14)
        self.report.pack(fill="both", expand=True, padx=8, pady=8)

        self.fig_time, self.ax_time = plt.subplots(2, 2, figsize=(9, 6), dpi=100)
        self.fig_time.patch.set_facecolor("#132238")
        for _ax in self.ax_time.flat:
            _ax.set_facecolor("#091525")
            _ax.tick_params(colors="#dce8f7", labelsize=8)
            for _sp in _ax.spines.values():
                _sp.set_color("#405673")
        self.fig_time.tight_layout(pad=2.2)
        self.canvas_time = FigureCanvasTkAgg(self.fig_time, master=self.tab_frames[1])
        self.canvas_time.get_tk_widget().pack(fill="both", expand=True)

        self.fig_ph, self.ax_ph = plt.subplots(figsize=(7, 5), dpi=95)
        self.canvas_ph = FigureCanvasTkAgg(self.fig_ph, master=self.tab_frames[2])
        self.canvas_ph.get_tk_widget().pack(fill="both", expand=True)

        self.fig_sch, self.ax_sch = plt.subplots(figsize=(7, 4), dpi=95)
        self.canvas_sch = FigureCanvasTkAgg(self.fig_sch, master=self.tab_frames[3])
        self.canvas_sch.get_tk_widget().pack(fill="both", expand=True)

        self.interpret = tk.Text(self.tab_frames[4], bg=self.FIELD, fg=self.TEXT,
                                 font=("Segoe UI", 11), wrap="word", relief="flat",
                                 padx=18, pady=18)
        self.interpret.pack(fill="both", expand=True, padx=8, pady=8)
        self.status = tk.Label(self.root, text="Listo.", bg="#101b2b", fg=self.MUTED,
                               anchor="w", padx=14, pady=6)
        self.status.pack(fill="x", side="bottom")

    def _label(self, parent, label):
        tk.Label(parent, text=label, bg=self.PANEL, fg=self.MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 3))

    def example_rl(self):
        vals = {"topology": "Serie", "kind": "RL", "R": "4", "L": "0.2",
                "C": "0.0001", "Vp": "20", "f": str(377/(2*math.pi)), "phase": "-60"}
        self._set_values(vals)
        self.calculate()

    def example_rc(self):
        vals = {"topology": "Serie", "kind": "RC", "R": "5", "L": "0.2",
                "C": "0.1", "Vp": "10", "f": str(4/(2*math.pi)), "phase": "0"}
        self._set_values(vals)
        self.calculate()

    def _set_values(self, values):
        for k, v in values.items():
            self.vars[k].set(v)

    def calculate(self):
        try:
            cfg = {
                "topology": self.vars["topology"].get(),
                "kind": self.vars["kind"].get(),
                "R": parse_quantity(self.vars["R"].get(), "R"),
                "L": parse_quantity(self.vars["L"].get(), "L"),
                "C": parse_quantity(self.vars["C"].get(), "C"),
                "Vp": parse_quantity(self.vars["Vp"].get(), "V"),
                "f": parse_quantity(self.vars["f"].get(), "f"),
                "phase": parse_quantity(self.vars["phase"].get(), "phase"),
            }
            if not all(math.isfinite(v) for v in [cfg[k] for k in ("R", "L", "C", "Vp", "f", "phase")]):
                raise ValueError("Usa valores numéricos finitos.")
            if cfg["Vp"] < 0 or cfg["f"] <= 0 or cfg["R"] < 0 or cfg["L"] < 0 or cfg["C"] < 0:
                raise ValueError("V, R, L y C no pueden ser negativos; la frecuencia debe ser positiva.")
            self.last = solve_circuit(cfg)
            self._update_ui()
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            messagebox.showerror("No se pudo simular", str(e))

    def _update_ui(self):
        r = self.last
        self.card_values["z"].config(text=f"{abs(r['Zt']):.4g} Ω")
        self.card_values["i"].config(text=f"{abs(r['It']):.4g} A")
        self.card_values["phi"].config(text=f"{r['phi']:.3f}°")
        self.card_values["pf"].config(text=f"{r['pf']:.4f}")

        self.report.config(state="normal")
        self.report.delete("1.0", "end")
        heading = [
            "VOLTLAB PRO — RESOLUCIÓN TIPO CUADERNO",
            "=" * 64,
            f"Configuración: circuito {r['kind']} en {r['topology'].lower()}",
            f"Componentes: R={format_eng(r['R'], 'Ω')}, L={format_eng(r['L'], 'H')}, C={format_eng(r['C'], 'F')}",
            f"Fuente: Vs(t) = {r['Vp']:g} cos({r['w']:.4g}t {angle_phrase(r['phase'])}) V",
            "",
        ]
        if r["topology"] == "Serie":
            for key, z in r["impedances"].items():
                heading.append(f"Impedancia de {key}: Z{key} = {rectangular(z)} Ω = {polar(z)} Ω")
        else:
            for key, z in r["impedances"].items():
                heading.append(f"Rama {key}: Z{key} = {rectangular(z)} Ω = {polar(z)} Ω")
        self.report.insert("1.0", "\n".join(heading + r["steps"]))
        self.report.config(state="disabled")

        self.interpret.config(state="normal")
        self.interpret.delete("1.0", "end")
        lines = [
            "INTERPRETACIÓN AUTOMÁTICA",
            "=" * 38,
            r["interpretation"],
            "",
            f"• Impedancia equivalente: {polar(r['Zt'])} Ω ({format_eng(abs(r['Zt']), 'Ω')}).",
            f"• Corriente total pico: {polar(r['It'])} A ({format_eng(abs(r['It']), 'A')}).",
            f"• Voltaje RMS de fuente: {format_eng(r['Vrms'], 'V')}.",
            f"• Corriente RMS total: {format_eng(r['Irms'], 'A')}.",
            f"• Potencia activa: {r['P']:.4f} W ({format_eng(r['P'], 'W')}).",
            f"• Potencia reactiva: {r['Q']:.4f} var.",
            f"• Potencia aparente: {r['S']:.4f} VA.",
            f"• Factor de potencia: {r['pf']:.4f}.",
            "",
            "Cómo leer el factor de potencia:",
            "• Cercano a 1: el circuito consume principalmente potencia activa.",
            "• Q positivo: comportamiento inductivo.",
            "• Q negativo: comportamiento capacitivo.",
            "",
            "Importante: simulación ideal en régimen sinusoidal permanente. No modela arranque, conmutación ni componentes semiconductores."
        ]
        self.interpret.insert("1.0", "\n".join(lines))
        self.interpret.config(state="disabled")
        self._plot_time()
        self._plot_phasors()
        self._draw_schematic()
        self.status.config(text=f"Simulación lista: {r['kind']} en {r['topology'].lower()} · f={r['f']:.4g} Hz")

    def _plot_time(self):
        r = self.last
        sig = r["signals"]
        t_ms = sig["t"] * 1000.0
        axv, axi, axz, axs = self.ax_time.flat
        palette = {
            "source": "#00a896", "R": "#3678c8", "L": "#d97706",
            "C": "#d94675", "grid": "#dbe3ed", "ink": "#203247",
            "purple": "#7956a8"
        }
        self.fig_time.patch.set_facecolor("#f4f7fb")
        for ax in (axv, axi, axz, axs):
            ax.clear()
            ax.set_facecolor("#ffffff")
            ax.grid(True, color=palette["grid"], alpha=0.95, linestyle="--", linewidth=0.7)
            ax.tick_params(colors=palette["ink"], labelsize=8)
            ax.xaxis.label.set_color(palette["ink"])
            ax.yaxis.label.set_color(palette["ink"])
            ax.title.set_color(palette["ink"])
            for sp in ax.spines.values():
                sp.set_color("#b9c6d5")
            ax.axhline(0, color="#7b8794", linewidth=0.7, alpha=0.65)

        # Voltajes: en serie se ven las caídas de cada elemento; en paralelo cada rama comparte Vs.
        axv.plot(t_ms, sig["Vs"], label="Fuente Vs", linewidth=2.5, color=palette["source"])
        for key, color in (("VR", palette["R"]), ("VL", palette["L"]), ("VC", palette["C"])):
            if key in sig:
                axv.plot(t_ms, sig[key], label=key, linewidth=1.7, color=color)
        axv.set_title("Voltajes instantáneos", fontsize=10, fontweight="bold", loc="left")
        axv.set_xlabel("Tiempo (ms)")
        axv.set_ylabel("Voltaje (V)")
        axv.legend(loc="upper right", fontsize=7, ncol=2, frameon=True,
                   facecolor="#f8fafc", edgecolor="#dbe3ed")
        axv.margins(x=0.02)

        axi.plot(t_ms, sig["IT"], label="Corriente total", linewidth=2.5, color=palette["source"])
        if r["topology"] == "Paralelo":
            for key, color in (("IR", palette["R"]), ("IL", palette["L"]), ("IC", palette["C"])):
                if key in sig:
                    axi.plot(t_ms, sig[key], label=key, linewidth=1.7, color=color)
        axi.set_title("Corrientes instantáneas", fontsize=10, fontweight="bold", loc="left")
        axi.set_xlabel("Tiempo (ms)")
        axi.set_ylabel("Corriente (A)")
        axi.legend(loc="upper right", fontsize=7, frameon=True,
                   facecolor="#f8fafc", edgecolor="#dbe3ed")
        axi.margins(x=0.02)

        Z = r["Zt"]
        axz.plot([0, Z.real], [0, Z.imag], marker="o", linewidth=2.8,
                 color=palette["purple"], markersize=5)
        axz.plot([Z.real, Z.real], [0, Z.imag], linestyle="--", color="#7b8794", linewidth=1.2)
        axz.plot([0, Z.real], [Z.imag, Z.imag], linestyle="--", color="#7b8794", linewidth=1.2)
        axz.scatter([Z.real], [Z.imag], color=palette["purple"], s=30, zorder=4)
        axz.annotate(f"ZT = {format_eng(abs(Z), 'Ω')}\\nθ = {angle(Z):.2f}°",
                     (Z.real, Z.imag), xytext=(8, 8), textcoords="offset points",
                     color=palette["ink"], fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dbe3ed"))
        limz = max(abs(Z.real), abs(Z.imag), abs(Z), 1e-3) * 1.4
        axz.set_xlim(min(-0.2*limz, Z.real-0.2*limz), max(1.15*limz, Z.real+0.2*limz))
        axz.set_ylim(min(-1.15*limz, Z.imag-0.2*limz), max(1.15*limz, Z.imag+0.2*limz))
        axz.set_aspect("equal", adjustable="datalim")
        axz.set_title("Triángulo de impedancia", fontsize=10, fontweight="bold", loc="left")
        axz.set_xlabel("Parte real R (Ω)")
        axz.set_ylabel("Parte imaginaria X (Ω)")

        P, Q, S = r["P"], r["Q"], r["S"]
        axs.plot([0, P], [0, 0], linewidth=2.8, color=palette["source"],
                 label=f"P = {format_eng(P, 'W')}")
        axs.plot([P, P], [0, Q], linewidth=2.8, color=palette["R"],
                 label=f"Q = {format_eng(Q, 'var')}")
        axs.plot([0, P], [0, Q], linewidth=2.8, color=palette["purple"],
                 label=f"S = {format_eng(S, 'VA')}")
        axs.scatter([P], [Q], color=palette["purple"], s=30, zorder=4)
        axs.set_title("Triángulo de potencias", fontsize=10, fontweight="bold", loc="left")
        axs.set_xlabel("Potencia activa P (W)")
        axs.set_ylabel("Potencia reactiva Q (var)")
        lim = max(abs(P), abs(Q), S, 1e-3) * 1.35
        axs.set_xlim(min(-0.15*lim, P-0.15*lim), max(1.15*lim, P+0.15*lim))
        axs.set_ylim(min(-1.15*lim, Q-0.15*lim), max(1.15*lim, Q+0.15*lim))
        axs.legend(loc="best", fontsize=7, frameon=True,
                   facecolor="#f8fafc", edgecolor="#dbe3ed")

        self.fig_time.suptitle(
            f"ANÁLISIS GRÁFICO · {r['kind']} en {r['topology'].lower()} · f = {format_eng(r['f'], 'Hz')}",
            color="#203247", fontsize=12, fontweight="bold"
        )
        self.fig_time.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        self.canvas_time.draw_idle()

    def _plot_phasors(self):
        r, ax = self.last, self.ax_ph
        ax.clear()
        self.fig_ph.patch.set_facecolor("#f4f7fb")
        ax.set_facecolor("#ffffff")
        ax.axhline(0, color="#8190a3", linewidth=0.9)
        ax.axvline(0, color="#8190a3", linewidth=0.9)
        if r["topology"] == "Serie":
            vectors = [(r["voltages"][k], "V" + k) for k in r["voltages"]]
            vectors.append((r["Vs_phasor"], "Vs total"))
            unit = "V"
            subtitle = "En serie: los voltajes se suman fasorialmente"
        else:
            vectors = [(r["currents"][k], "I" + k) for k in r["currents"]]
            vectors.append((r["It"], "It total"))
            unit = "A"
            subtitle = "En paralelo: las corrientes se suman fasorialmente"
        maxv = max([abs(z) for z, _ in vectors] + [1e-3]) * 1.45
        colors = ["#009688", "#3678c8", "#d97706", "#d94675"]
        for idx, (z, label) in enumerate(vectors):
            color = colors[idx % len(colors)]
            ax.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=color, mutation_scale=14))
            ax.scatter([z.real], [z.imag], color=color, s=22, zorder=3)
            tx = z.real + (0.035 * maxv if z.real >= 0 else -0.035 * maxv)
            ty = z.imag + (0.035 * maxv if z.imag >= 0 else -0.06 * maxv)
            ax.text(tx, ty, f"{label}\\n{format_eng(abs(z), unit)} ∠ {angle(z):.1f}°",
                    fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dbe3ed", alpha=0.92))
        ax.set_xlim(-maxv, maxv)
        ax.set_ylim(-maxv, maxv)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Parte real ({unit})", color="#203247")
        ax.set_ylabel(f"Parte imaginaria ({unit})", color="#203247")
        ax.set_title("DIAGRAMA FASORIAL\\n" + subtitle, color="#203247",
                     fontsize=11, fontweight="bold")
        ax.tick_params(colors="#203247")
        for sp in ax.spines.values():
            sp.set_color("#b9c6d5")
        ax.grid(True, color="#dbe3ed", alpha=0.9, linestyle="--")
        self.fig_ph.tight_layout()
        self.canvas_ph.draw_idle()

    def _draw_schematic(self):
        """Esquema claro para cada circuito seleccionado: R, RL, RC, RLC en serie/paralelo."""
        r, ax = self.last, self.ax_sch
        ax.clear()
        self.fig_sch.patch.set_facecolor("#f4f7fb")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 7)
        ax.axis("off")
        wire, ink = "#26384d", "#172b45"
        colors = {"R": "#2878b5", "L": "#0b9b83", "C": "#bd5b91"}
        ax.set_title(f"ESQUEMA DEL CIRCUITO · {r['kind']} EN {r['topology'].upper()}",
                     fontsize=12, fontweight="bold", color=ink, pad=12)
        elements = [k for k in ("R", "L", "C") if k in r["impedances"]]
        labels = {
            "R": f"R = {format_eng(r['R'], 'Ω')}",
            "L": f"L = {format_eng(r['L'], 'H')}",
            "C": f"C = {format_eng(r['C'], 'F')}",
        }

        def line(x1, y1, x2, y2, lw=2.0):
            ax.plot([x1, x2], [y1, y2], color=wire, lw=lw,
                    solid_capstyle="round", zorder=1)

        def source(cx, cy, radius=0.43):
            ax.add_patch(plt.Circle((cx, cy), radius, facecolor="#eef6ff",
                                    edgecolor="#2878b5", lw=2.1, zorder=3))
            ax.text(cx, cy, "~", ha="center", va="center", fontsize=21,
                    color="#2878b5", fontweight="bold")
            ax.text(cx-0.05, cy+radius+0.3, "FUENTE AC", ha="center",
                    color=ink, fontsize=8, fontweight="bold")
            ax.text(cx-0.05, cy-radius-0.3, f"{format_eng(r['Vp'], 'V')} pico",
                    ha="center", color="#52667c", fontsize=8)

        def component_h(key, x1, x2, y):
            mid = (x1+x2)/2
            color = colors[key]
            line(x1, y, mid-0.48, y)
            line(mid+0.48, y, x2, y)
            if key == "R":
                ax.add_patch(plt.Rectangle((mid-0.48, y-0.22), 0.96, 0.44,
                                           facecolor="#eaf3fb", edgecolor=color, lw=2.2, zorder=3))
            elif key == "L":
                xx = np.linspace(mid-0.48, mid+0.48, 180)
                yy = y + 0.17*np.sin(np.linspace(0, 8*np.pi, 180))
                ax.plot(xx, yy, color=color, lw=2.4, zorder=3)
            else:
                ax.plot([mid-0.12, mid-0.12], [y-0.40, y+0.40], color=color, lw=2.8, zorder=3)
                ax.plot([mid+0.12, mid+0.12], [y-0.40, y+0.40], color=color, lw=2.8, zorder=3)
            ax.text(mid, y+0.55, key, ha="center", color=color, fontsize=10, fontweight="bold")
            ax.text(mid, y-0.58, labels[key], ha="center", color=ink, fontsize=8)

        def component_v(key, x, y_top, y_bottom):
            mid = (y_top+y_bottom)/2
            color = colors[key]
            if key == "R":
                line(x, y_top, x, mid+0.43)
                ax.add_patch(plt.Rectangle((x-0.22, mid-0.43), 0.44, 0.86,
                                           facecolor="#eaf3fb", edgecolor=color, lw=2.2, zorder=3))
                line(x, mid-0.43, x, y_bottom)
            elif key == "L":
                line(x, y_top, x, mid+0.48)
                yy = np.linspace(mid+0.48, mid-0.48, 180)
                xx = x + 0.16*np.sin(np.linspace(0, 8*np.pi, 180))
                ax.plot(xx, yy, color=color, lw=2.4, zorder=3)
                line(x, mid-0.48, x, y_bottom)
            else:
                line(x, y_top, x, mid+0.12)
                ax.plot([x-0.40, x+0.40], [mid+0.12, mid+0.12], color=color, lw=2.8, zorder=3)
                ax.plot([x-0.40, x+0.40], [mid-0.12, mid-0.12], color=color, lw=2.8, zorder=3)
                line(x, mid-0.12, x, y_bottom)
            ax.text(x+0.50, mid+0.18, key, ha="left", color=color, fontsize=10, fontweight="bold")
            ax.text(x+0.50, mid-0.18, labels[key], ha="left", color=ink, fontsize=8)

        if r["topology"] == "Serie":
            y_top, y_bottom, left, right = 4.55, 1.45, 1.35, 10.7
            source(left, (y_top+y_bottom)/2)
            line(left, y_top, left, 4.55)
            line(left, 4.55, 2.15, 4.55)
            line(left, y_bottom, left, 1.45)
            line(left, y_bottom, right, y_bottom)
            line(right, y_bottom, right, y_top)
            if len(elements) == 1:
                spans = [(4.25, 7.75)]
            elif len(elements) == 2:
                spans = [(2.85, 4.95), (6.25, 8.35)]
            else:
                spans = [(2.55, 4.15), (5.05, 6.65), (7.55, 9.15)]
            line(2.15, y_top, spans[0][0], y_top)
            for idx, key in enumerate(elements):
                x1, x2 = spans[idx]
                component_h(key, x1, x2, y_top)
                if idx < len(elements)-1:
                    line(x2, y_top, spans[idx+1][0], y_top)
            line(spans[-1][1], y_top, right, y_top)
            ax.text(6, 0.62, "Serie: misma corriente en todos los elementos; los voltajes se suman.",
                    ha="center", color="#52667c", fontsize=9)
        else:
            y_top, y_bottom, left, right = 5.15, 1.55, 2.35, 10.15
            source(1.25, (y_top+y_bottom)/2)
            line(1.25, y_top, left, y_top)
            line(left, y_top, right, y_top)
            line(1.25, y_bottom, left, y_bottom)
            line(left, y_bottom, right, y_bottom)
            xs = np.linspace(left+1.1, right-1.1, len(elements)) if len(elements) > 1 else [6.2]
            for x, key in zip(xs, elements):
                line(x, y_top, x, y_top-0.50)
                component_v(key, x, y_top-0.50, y_bottom+0.50)
                line(x, y_bottom+0.50, x, y_bottom)
                ax.add_patch(plt.Circle((x, y_top), 0.055, color=wire, zorder=4))
                ax.add_patch(plt.Circle((x, y_bottom), 0.055, color=wire, zorder=4))
            ax.text(6.2, 0.62, "Paralelo: el voltaje es común en cada rama; las corrientes se suman.",
                    ha="center", color="#52667c", fontsize=9)
        ax.text(11.6, 6.55, f"f = {format_eng(r['f'], 'Hz')}", ha="right",
                fontsize=8, color="#52667c")
        ax.text(6, 0.20, "Esquema ideal generado según la selección. No es una netlist SPICE.",
                ha="center", color="#64748b", fontsize=7.5)
        self.fig_sch.tight_layout(pad=1.3)
        self.canvas_sch.draw_idle()

    def export_txt(self):
        if not self.last:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Archivo de texto", "*.txt")], initialfile="VoltLab_desarrollo.txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.last["steps"]))
                f.write("\n\nINTERPRETACIÓN:\n" + self.last["interpretation"] + "\n")
            messagebox.showinfo("Exportado", "Desarrollo guardado correctamente.")

    def export_csv(self):
        if not self.last:
            return
        r = self.last
        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("Archivo CSV", "*.csv")], initialfile="VoltLab_resultados.csv")
        if path:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["Parámetro", "Valor", "Unidad"])
                rows = [
                    ("Topología", r["topology"], ""), ("Tipo", r["kind"], ""),
                    ("Frecuencia", r["f"], "Hz"), ("Omega", r["w"], "rad/s"),
                    ("Impedancia total", rectangular(r["Zt"]), "Ω"),
                    ("Módulo de impedancia", abs(r["Zt"]), "Ω"),
                    ("Ángulo de Z", r["phi"], "grados"),
                    ("Corriente pico", abs(r["It"]), "A"),
                    ("Potencia activa", r["P"], "W"),
                    ("Potencia reactiva", r["Q"], "var"),
                    ("Potencia aparente", r["S"], "VA"),
                    ("Factor de potencia", r["pf"], ""),
                    ("Interpretación", r["interpretation"], "")
                ]
                writer.writerows(rows)
            messagebox.showinfo("Exportado", "Resultados CSV guardados correctamente.")


def main():
    root = tk.Tk()
    VoltLabPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()
