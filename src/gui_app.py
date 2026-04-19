from __future__ import annotations

import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from heuristics.tabu_search import run_tabu_from_method
from init.methods import get_initial_methods
from utils.checker import evaluate_solution
from utils.parser import parse_instance

ALL_METHODS = ["greedy", "solomon", "clarke_wright", "random", "sweep"]
TABU_OPERATORS = ["relocate", "swap", "two_opt_intra", "two_opt_inter", "or_opt", "cross_exchange"]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VRPTW Heuristics Desktop")
        self.geometry("1200x760")
        self.minsize(1020, 680)

        self._worker_thread: threading.Thread | None = None
        self._result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._customer_xy: dict[int, tuple[float, float]] = {}
        self._depot_id: int = 0
        self._last_rendered_iteration: int = -1
        self._map_points_screen: dict[int, tuple[float, float]] = {}
        self._current_routes: list[list[int]] = []
        self._map_padding = 28

        self._configure_style()
        self._build_ui()
        self.after(120, self._poll_worker)

        self._route_line_ids: list[int] = []
        self._max_drawn_routes = 100  # 0 means draw all routes
        self._max_iteration_log_lines = 500

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Root.TFrame", background="#f4f6f8")
        style.configure("Panel.TLabelframe", background="#ffffff")
        style.configure("Panel.TLabelframe.Label", background="#ffffff", foreground="#203244", font=("Segoe UI", 10, "bold"))
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#f4f6f8", foreground="#12263a", font=("Segoe UI", 17, "bold"))
        style.configure("Hint.TLabel", background="#f4f6f8", foreground="#4a5a6a", font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#f4f6f8", foreground="#123a5a", font=("Segoe UI", 10, "bold"))
        style.configure("Field.TLabel", background="#ffffff", foreground="#1f2d3a")
        style.configure("Done.TLabel", background="#f4f6f8", foreground="#166534", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, style="Root.TFrame", padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="VRPTW Heuristics", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            root,
            text="Choose instance and settings, then run Initial or Tabu search.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        content = ttk.Frame(root, style="Root.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=2)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(content, text="Configuration", style="Panel.TLabelframe", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(1, weight=1)

        right = ttk.LabelFrame(content, text="Results", style="Panel.TLabelframe", padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self.instance_var = tk.StringVar(value="R108")
        self.instance_file_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value="tabu")
        self.init_method_var = tk.StringVar(value="solomon")

        self.iterations_var = tk.StringVar(value="100")
        self.tabu_tenure_var = tk.StringVar(value="15")
        self.aspiration_var = tk.BooleanVar(value=True)
        self.div_interval_var = tk.StringVar(value="35")
        self.int_interval_var = tk.StringVar(value="15")
        self.per_op_moves_var = tk.StringVar(value="80")
        self.enable_improvement_operator_var = tk.BooleanVar(value=True)
        self.improvement_interval_var = tk.StringVar(value="30")
        self.improvement_regret_k_var = tk.StringVar(value="2")
        self.show_iteration_details_var = tk.BooleanVar(value=True)

        self.operator_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in TABU_OPERATORS
        }

        self._row(left, 0, "Instance name", self._instance_picker(left))
        self._row(left, 1, "Mode", self._mode_picker(left))
        self._row(left, 2, "Init method", ttk.Combobox(left, textvariable=self.init_method_var, values=ALL_METHODS, state="readonly"))

        ttk.Separator(left).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        self._row(left, 4, "Tabu iterations", ttk.Entry(left, textvariable=self.iterations_var))
        self._row(left, 5, "Tabu tenure", ttk.Entry(left, textvariable=self.tabu_tenure_var))

        aspiration_check = ttk.Checkbutton(left, text="Use aspiration", variable=self.aspiration_var)
        aspiration_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=4)

        self._row(left, 7, "Diversification", ttk.Entry(left, textvariable=self.div_interval_var))
        self._row(left, 8, "Intensification", ttk.Entry(left, textvariable=self.int_interval_var))
        self._row(left, 9, "Moves / operator", ttk.Entry(left, textvariable=self.per_op_moves_var))

        ttk.Checkbutton(
            left,
            text="Enable improvement operator",
            variable=self.enable_improvement_operator_var,
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(6, 2))
        self._row(left, 11, "Improvement interval", ttk.Entry(left, textvariable=self.improvement_interval_var))
        self._row(left, 12, "Improvement regret_k", ttk.Entry(left, textvariable=self.improvement_regret_k_var))

        ttk.Checkbutton(
            left,
            text="Show iteration details",
            variable=self.show_iteration_details_var,
        ).grid(row=13, column=0, columnspan=2, sticky="w", pady=(6, 2))

        ops_frame = ttk.Frame(left, style="Panel.TFrame")
        ops_frame.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(ops_frame, text="Tabu operators", style="Field.TLabel").grid(row=0, column=0, sticky="w")

        for idx, name in enumerate(TABU_OPERATORS, start=1):
            ttk.Checkbutton(ops_frame, text=name, variable=self.operator_vars[name]).grid(
                row=idx,
                column=0,
                sticky="w",
                pady=1,
            )

        self.run_button = ttk.Button(left, text="Run", command=self._run_clicked)
        self.run_button.grid(row=15, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(left, text="Clear Output", command=self._clear_output).grid(row=15, column=1, sticky="ew", pady=(12, 0), padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready")
        status_row = ttk.Frame(right, style="Panel.TFrame")
        status_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_row.columnconfigure(1, weight=1)
        ttk.Label(status_row, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        self.done_var = tk.StringVar(value="")
        ttk.Label(status_row, textvariable=self.done_var, style="Done.TLabel").grid(row=0, column=1, sticky="e")

        tabs = ttk.Notebook(right)
        tabs.grid(row=1, column=0, sticky="nsew")

        map_tab = ttk.Frame(tabs, style="Panel.TFrame")
        log_tab = ttk.Frame(tabs, style="Panel.TFrame")
        iter_tab = ttk.Frame(tabs, style="Panel.TFrame")
        tabs.add(map_tab, text="Live Map")
        tabs.add(log_tab, text="Run Log")
        tabs.add(iter_tab, text="Iterations")

        map_tab.rowconfigure(0, weight=1)
        map_tab.columnconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(
            map_tab,
            bg="#f8fafc",
            highlightthickness=1,
            highlightbackground="#d0d7de",
        )
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.bind("<Configure>", self._on_map_resize)

        log_tab.rowconfigure(0, weight=1)
        log_tab.columnconfigure(0, weight=1)
        output_frame = ttk.Frame(log_tab, style="Panel.TFrame")
        output_frame.grid(row=0, column=0, sticky="nsew")
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)

        self.output = tk.Text(
            output_frame,
            wrap="word",
            font=("Consolas", 10),
            bg="#0f1720",
            fg="#edf2f7",
            insertbackground="#edf2f7",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.output.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=scroll.set)

        iter_tab.rowconfigure(0, weight=1)
        iter_tab.columnconfigure(0, weight=1)
        self.iteration_text = tk.Text(
            iter_tab,
            wrap="none",
            font=("Consolas", 9),
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="#c9d1d9",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.iteration_text.grid(row=0, column=0, sticky="nsew")
        iter_scroll_y = ttk.Scrollbar(iter_tab, orient="vertical", command=self.iteration_text.yview)
        iter_scroll_y.grid(row=0, column=1, sticky="ns")
        iter_scroll_x = ttk.Scrollbar(iter_tab, orient="horizontal", command=self.iteration_text.xview)
        iter_scroll_x.grid(row=1, column=0, sticky="ew")
        self.iteration_text.configure(yscrollcommand=iter_scroll_y.set, xscrollcommand=iter_scroll_x.set)

        self._append_output("Ready. Configure parameters then click Run.\n")

    def _instance_picker(self, parent: ttk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.columnconfigure(0, weight=1)

        archive_instances = self._list_archive_instances()
        instance_combo = ttk.Combobox(
            frame,
            textvariable=self.instance_var,
            values=archive_instances,
        )
        instance_combo.grid(row=0, column=0, sticky="ew")

        ttk.Button(frame, text="Browse file", command=self._browse_instance_file).grid(row=0, column=1, padx=(8, 0))
        return frame

    def _mode_picker(self, parent: ttk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Panel.TFrame")
        ttk.Radiobutton(frame, text="Initial", value="initial", variable=self.mode_var).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Tabu", value="tabu", variable=self.mode_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        return frame

    def _row(self, parent: ttk.Widget, row: int, label: str, widget: ttk.Widget) -> None:
        ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)

    def _list_archive_instances(self) -> list[str]:
        archive_dir = Path(__file__).resolve().parents[1] / "Archive"
        if not archive_dir.exists():
            return []

        names: list[str] = []
        for path in sorted(archive_dir.glob("*.txt")):
            stem = path.stem
            if stem.startswith("__"):
                continue
            names.append(stem)
        return names

    def _browse_instance_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select VRPTW instance",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if selected:
            self.instance_file_var.set(selected)
            self.instance_var.set(selected)

    def _clear_output(self) -> None:
        self.output.delete("1.0", tk.END)
        self.iteration_text.delete("1.0", tk.END)

    def _append_output(self, text: str) -> None:
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def _append_iteration_detail(self, text: str) -> None:
        self.iteration_text.insert(tk.END, text)
        # Keep the iteration log bounded to avoid Text widget slowdown.
        total_lines = int(float(self.iteration_text.index("end-1c").split(".")[0]))
        if total_lines > self._max_iteration_log_lines:
            remove_lines = total_lines - self._max_iteration_log_lines
            self.iteration_text.delete("1.0", f"{remove_lines + 1}.0")
        self.iteration_text.see(tk.END)

    def _run_clicked(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Still running", "A run is already in progress.")
            return

        try:
            config = self._collect_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Running...")
        self.done_var.set("")
        self._last_rendered_iteration = -1
        self._current_routes = []
        self._route_line_ids = []
        self._append_output("\n" + "=" * 70 + "\n")
        self._append_output(f"Starting {config['mode']} run on {config['instance']}...\n")
        self._append_iteration_detail("\n" + "=" * 70 + "\n")
        self._append_iteration_detail(f"Starting {config['mode']} run on {config['instance']}...\n")

        self._worker_thread = threading.Thread(target=self._worker_run, args=(config,), daemon=True)
        self._worker_thread.start()

    def _collect_config(self) -> dict:
        instance_value = self.instance_var.get().strip()
        if not instance_value:
            raise ValueError("Instance is required.")

        return {
            "instance": instance_value,
            "mode": self.mode_var.get(),
            "init_method": self.init_method_var.get(),
            "iterations": int(self.iterations_var.get()),
            "tabu_tenure": int(self.tabu_tenure_var.get()),
            "aspiration": self.aspiration_var.get(),
            "diversification_interval": int(self.div_interval_var.get()),
            "intensification_interval": int(self.int_interval_var.get()),
            "per_operator_moves": int(self.per_op_moves_var.get()),
            "enable_improvement_operator": self.enable_improvement_operator_var.get(),
            "improvement_interval": int(self.improvement_interval_var.get()),
            "improvement_regret_k": int(self.improvement_regret_k_var.get()),
            "enabled_operators": [name for name, enabled in self.operator_vars.items() if enabled.get()],
            "show_iteration_details": self.show_iteration_details_var.get(),
        }

    def _worker_run(self, config: dict) -> None:
        try:
            if not config["enabled_operators"] and config["mode"] == "tabu":
                raise ValueError("Select at least one operator for tabu mode.")

            if config["mode"] == "initial":
                report = self._run_initial(config)
            else:
                report = self._run_tabu(config)

            self._result_queue.put(("success", report))
        except Exception:
            self._result_queue.put(("error", traceback.format_exc()))

    def _run_initial(self, config: dict) -> str:
        problem = parse_instance(config["instance"])
        self._result_queue.put(("map_init", self._map_payload(problem)))
        methods = get_initial_methods()
        method_name = config["init_method"]
        if method_name not in methods:
            raise ValueError(f"Unknown init method: {method_name}")

        routes = methods[method_name](problem)
        routes = [list(route) for route in routes if len(route) >= 2 and route[0] == problem.depot_id and route[-1] == problem.depot_id]
        feasible, distance, message = evaluate_solution(problem, routes)

        lines = [
            f"Mode: initial ({method_name})",
            f"Instance: {problem.name}",
            f"Feasible: {feasible}",
            f"Distance: {distance:.3f}" if feasible else "Distance: inf",
            f"Routes used: {len(routes)} / {problem.vehicle_count}",
            f"Status: {message}",
            "",
            "Routes:",
        ]
        for idx, route in enumerate(routes, start=1):
            lines.append(f"  Route {idx}: {' -> '.join(str(node) for node in route)}")

        self._result_queue.put(
            (
                "iter",
                {
                    "iteration": 0,
                    "event": "initial",
                    "current_cost": distance,
                    "best_cost": distance,
                    "routes": routes,
                },
            )
        )

        return "\n".join(lines)

    def _run_tabu(self, config: dict) -> str:
        problem = parse_instance(config["instance"])
        self._result_queue.put(("map_init", self._map_payload(problem)))

        def ui_iteration_callback(payload: dict) -> None:
            # Keep all iteration payloads for the Iterations panel.
            self._result_queue.put(("iter", payload))

        row = run_tabu_from_method(
            instance=config["instance"],
            method=config["init_method"],
            iterations=config["iterations"],
            tabu_tenure=config["tabu_tenure"],
            aspiration=config["aspiration"],
            diversification_interval=config["diversification_interval"],
            intensification_interval=config["intensification_interval"],
            per_operator_moves=config["per_operator_moves"],
            enabled_operators=config["enabled_operators"],
            print_iterations=False,
            iteration_callback=ui_iteration_callback,
            enable_improvement_operator=config["enable_improvement_operator"],
            improvement_interval=config["improvement_interval"],
            regret_k=config["improvement_regret_k"],
        )

        lines = [
            f"Mode: tabu (start={row['method']})",
            f"Instance: {row['instance']}",
            f"Initial distance: {row['init_distance']:.3f}",
            f"Best distance: {row['tabu_distance']:.3f}",
            f"Feasible: {row['tabu_feasible']}",
            f"Routes used: {row['routes_used']} / {row['vehicles_available']}",
            f"Elapsed: {row['total_elapsed_seconds']:.3f}s",
            f"Status: {row['message']}",
            "",
            "Best routes:",
        ]
        for idx, route in enumerate(row["best_routes"], start=1):
            lines.append(f"  Route {idx}: {' -> '.join(str(node) for node in route)}")

        return "\n".join(lines)

    def _map_payload(self, problem) -> dict:
        return {
            "customer_xy": {cid: (cust.x, cust.y) for cid, cust in problem.customers.items()},
            "depot_id": problem.depot_id,
        }

    def _on_map_resize(self, _event=None) -> None:
        if not self._customer_xy:
            return
        self._compute_screen_points()
        self._draw_static_nodes()
        self._draw_routes(self._current_routes, iteration=self._last_rendered_iteration, event="resize")

    def _compute_screen_points(self) -> None:
        if not self._customer_xy:
            return

        width = max(200, int(self.map_canvas.winfo_width()))
        height = max(200, int(self.map_canvas.winfo_height()))

        xs = [pt[0] for pt in self._customer_xy.values()]
        ys = [pt[1] for pt in self._customer_xy.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = max(1e-9, max_x - min_x)
        span_y = max(1e-9, max_y - min_y)
        inner_w = max(10.0, float(width - 2 * self._map_padding))
        inner_h = max(10.0, float(height - 2 * self._map_padding))
        scale = min(inner_w / span_x, inner_h / span_y)

        used_w = span_x * scale
        used_h = span_y * scale
        pad_x = (width - used_w) / 2.0
        pad_y = (height - used_h) / 2.0

        self._map_points_screen = {}
        for cid, (x, y) in self._customer_xy.items():
            sx = pad_x + (x - min_x) * scale
            sy = height - (pad_y + (y - min_y) * scale)
            self._map_points_screen[cid] = (sx, sy)

    def _draw_static_nodes(self) -> None:
        self.map_canvas.delete("all")
        # Canvas delete invalidates previous route item IDs.
        self._route_line_ids = []
        self.map_canvas.create_text(
            12,
            12,
            anchor="nw",
            text="Live routes",
            fill="#0f172a",
            font=("Segoe UI", 10, "bold"),
            tags=("map_title",),
        )

        customers = [(cid, xy) for cid, xy in self._customer_xy.items() if cid != self._depot_id]
        if customers:
            for cid, _ in customers:
                sx, sy = self._map_points_screen[cid]
                self.map_canvas.create_oval(
                    sx - 2,
                    sy - 2,
                    sx + 2,
                    sy + 2,
                    fill="#94a3b8",
                    outline="",
                    tags=("customer_node",),
                )

        depot_xy = self._map_points_screen.get(self._depot_id)
        if depot_xy is not None:
            sx, sy = depot_xy
            self.map_canvas.create_rectangle(
                sx - 4,
                sy - 4,
                sx + 4,
                sy + 4,
                fill="#dc2626",
                outline="#7f1d1d",
                tags=("depot_node",),
            )

    def _draw_routes(self, routes: list[list[int]], iteration: int, event: str, move: str = "-") -> None:
        if not self._customer_xy:
            return

        self._current_routes = [list(route) for route in routes]

        colors = [
            "#1d4ed8", "#16a34a", "#ea580c", "#9333ea",
            "#0f766e", "#be185d", "#475569", "#0891b2",
            "#d97706", "#4f46e5",
        ]

        # Optional route cap for very large runs (0 means no cap).
        if self._max_drawn_routes > 0:
            routes = routes[:self._max_drawn_routes]

        # Ensure enough line objects exist
        while len(self._route_line_ids) < len(routes):
            line_id = self.map_canvas.create_line(
                0, 0, 0, 0,
                width=2,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                tags=("route_line",),
            )
            self._route_line_ids.append(line_id)

        # Update each route line instead of recreating
        for idx, route in enumerate(routes):
            coords: list[float] = []
            point_count = 0
            for node in route:
                point = self._map_points_screen.get(node)
                if point is None:
                    continue
                point_count += 1
                coords.extend([point[0], point[1]])

            if point_count < 2:
                self.map_canvas.itemconfig(self._route_line_ids[idx], state="hidden")
                continue

            self.map_canvas.coords(self._route_line_ids[idx], *coords)
            self.map_canvas.itemconfig(
                self._route_line_ids[idx],
                fill=colors[idx % len(colors)],
                state="normal",
            )

        # Hide unused lines
        for idx in range(len(routes), len(self._route_line_ids)):
            self.map_canvas.itemconfig(self._route_line_ids[idx], state="hidden")

        title = f"Live routes | iter={iteration} | event={event} | move={move} | routes={len(routes)}"
        self.map_canvas.itemconfigure("map_title", text=title)
        self.map_canvas.tag_raise("customer_node")
        self.map_canvas.tag_raise("depot_node")
        self.map_canvas.update_idletasks()

    def _format_iteration_line(self, payload: dict) -> str:
        it = payload.get("iteration", -1)
        event = payload.get("event", "-")
        current_cost = payload.get("current_cost")
        best_cost = payload.get("best_cost")
        candidate_count = payload.get("candidate_count")
        move_key = payload.get("move_key")

        current_text = f"{current_cost:.3f}" if isinstance(current_cost, (int, float)) else "-"
        best_text = f"{best_cost:.3f}" if isinstance(best_cost, (int, float)) else "-"
        move_text = str(move_key[0]) if isinstance(move_key, list) and move_key else "-"
        cand_text = str(candidate_count) if candidate_count is not None else "-"

        return (
            f"it={it:>4} | event={event:<26} | move={move_text:<15} | "
            f"current={current_text:<12} | best={best_text:<12} | cand={cand_text}\n"
        )

    def _poll_worker(self) -> None:
        latest_iter_payload = None
        iteration_lines: list[str] = []
        other_events = []

        # Drain queue -> keep only latest iteration
        while True:
            try:
                status, payload = self._result_queue.get_nowait()
            except queue.Empty:
                break

            if status == "iter":
                latest_iter_payload = payload
                if self.show_iteration_details_var.get() and isinstance(payload, dict):
                    iteration_lines.append(self._format_iteration_line(payload))
            else:
                other_events.append((status, payload))

        # Handle non-iteration events normally
        for status, payload in other_events:
            if status == "map_init":
                if not isinstance(payload, dict):
                    continue
                customer_xy = payload.get("customer_xy", {})
                depot_id = payload.get("depot_id", 0)
                if isinstance(customer_xy, dict):
                    self._customer_xy = customer_xy
                    self._depot_id = int(depot_id)
                    self._compute_screen_points()
                    self._draw_static_nodes()
                    self._draw_routes([], iteration=0, event="start", move="-")
                continue

            self.run_button.configure(state="normal")
            if status == "success":
                if not isinstance(payload, str):
                    payload = str(payload)
                self.status_var.set("Done")
                finished_at = datetime.now().strftime("%H:%M:%S")
                self.done_var.set(f"Job finished at {finished_at}")
                self._append_output(payload + "\n")
            elif status == "error":
                if not isinstance(payload, str):
                    payload = str(payload)
                self.status_var.set("Failed")
                self.done_var.set("Job failed")
                self._append_output("ERROR\n" + payload + "\n")

        # Render ONLY latest iteration (prevents lag buildup)
        if isinstance(latest_iter_payload, dict):
            iteration = int(latest_iter_payload.get("iteration", -1))
            routes = (
                latest_iter_payload.get("current_routes")
                or latest_iter_payload.get("routes")
                or latest_iter_payload.get("best_routes")
                or self._current_routes
            )
            event = str(latest_iter_payload.get("event", "iter"))
            move_key = latest_iter_payload.get("move_key")
            move_name = str(move_key[0]) if isinstance(move_key, list) and move_key else "-"

            if isinstance(routes, list):
                self._draw_routes(routes, iteration=iteration, event=event, move=move_name)
                self._last_rendered_iteration = iteration

        if iteration_lines:
            self._append_iteration_detail("".join(iteration_lines))

        # Smooth ~60 FPS loop
        self.after(16, self._poll_worker)


if __name__ == "__main__":
    app = App()
    app.mainloop()
