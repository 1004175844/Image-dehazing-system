import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from dehaze import dehaze_bgr
from metrics import calculate_metrics
from metrics_viz import build_bar_figure, build_radar_figure, save_metric_figures


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DISPLAY_FALLBACK = (520, 360)
COLORS = {
    "app_bg": "#eef3f8",
    "header_bg": "#f8fbff",
    "header_text": "#1e293b",
    "header_subtle": "#64748b",
    "panel_bg": "#f8fbff",
    "card_bg": "#ffffff",
    "text_main": "#1f2937",
    "text_muted": "#64748b",
    "accent": "#1d4ed8",
    "accent_dark": "#1e40af",
    "accent_soft": "#dbeafe",
}


def _read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path, image_bgr):
    ext = os.path.splitext(path)[1].lower() or ".png"
    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError("Failed to encode image.")
    encoded.tofile(path)


def _fit_image(pil_image, max_size):
    max_w, max_h = max_size
    w, h = pil_image.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil_image.resize(new_size, Image.LANCZOS)


class DehazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Dehazing System")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)

        self.folder_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready. Choose an image folder first.")
        self.images = []
        self.current_path = None
        self.current_image = None
        self.output_image = None
        self.metrics_result = None

        self.input_tk = None
        self.output_tk = None
        self._resize_job = None
        self.bar_canvas = None
        self.radar_canvas = None

        self._build_style()
        self._build_ui()
        self.root.bind("<Configure>", self._on_root_resize)

    def _build_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.configure(bg=COLORS["app_bg"])
        style.configure("App.TFrame", background=COLORS["app_bg"])
        style.configure("Header.TFrame", background=COLORS["header_bg"])
        style.configure(
            "HeaderTitle.TLabel",
            background=COLORS["header_bg"],
            foreground=COLORS["header_text"],
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=COLORS["header_bg"],
            foreground=COLORS["header_subtle"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "Sidebar.TLabelframe",
            background=COLORS["panel_bg"],
            bordercolor="#dbe2ea",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Sidebar.TLabelframe.Label",
            background=COLORS["panel_bg"],
            foreground=COLORS["text_main"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure("Card.TFrame", background=COLORS["card_bg"])
        style.configure(
            "CardTitle.TLabel",
            background=COLORS["card_bg"],
            foreground=COLORS["text_main"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "CardHint.TLabel",
            background=COLORS["card_bg"],
            foreground=COLORS["text_muted"],
            font=("Segoe UI", 9),
        )
        style.configure(
            "FieldLabel.TLabel",
            background=COLORS["panel_bg"],
            foreground=COLORS["text_main"],
            font=("Segoe UI", 9, "bold"),
        )
        style.configure("Field.TEntry", padding=(8, 7), fieldbackground="#ffffff")
        style.configure(
            "Primary.TButton",
            background=COLORS["accent"],
            foreground="#ffffff",
            borderwidth=0,
            padding=(14, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["accent_dark"]), ("pressed", COLORS["accent_dark"])],
        )
        style.configure(
            "Secondary.TButton",
            background="#e2e8f0",
            foreground=COLORS["text_main"],
            borderwidth=0,
            padding=(12, 9),
            font=("Segoe UI", 10),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#cbd5e1"), ("pressed", "#cbd5e1")],
        )
        style.configure(
            "Status.TLabel",
            background="#e9eff6",
            foreground=COLORS["text_muted"],
            padding=(14, 10),
            font=("Segoe UI", 9),
        )
        style.configure(
            "Metric.Treeview",
            font=("Segoe UI", 10),
            rowheight=28,
        )
        style.configure(
            "Metric.Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
        )

    def _build_ui(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Image Dehazing", style="HeaderTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Traditional dark channel prior + objective metric visualization",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=14, pady=(10, 8))

        self.work_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.metrics_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.work_tab, text="Dehaze Workspace")
        self.notebook.add(self.metrics_tab, text="Metrics")

        self._build_work_tab()
        self._build_metrics_tab()

        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel")
        status.grid(row=2, column=0, sticky="ew")

    def _build_work_tab(self):
        self.work_tab.rowconfigure(0, weight=1)
        self.work_tab.columnconfigure(0, weight=1)

        main = ttk.Frame(self.work_tab, style="App.TFrame", padding=(0, 0, 0, 8))
        main.grid(row=0, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        sidebar = ttk.LabelFrame(main, text="Image Source", style="Sidebar.TLabelframe")
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text="Folder", style="FieldLabel.TLabel").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 4)
        )
        folder_row = ttk.Frame(sidebar, style="App.TFrame")
        folder_row.grid(row=1, column=0, sticky="ew", padx=12)
        folder_row.columnconfigure(0, weight=1)

        folder_entry = ttk.Entry(folder_row, textvariable=self.folder_var, style="Field.TEntry")
        folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            folder_row, text="Browse", style="Secondary.TButton", command=self.pick_folder
        ).grid(row=0, column=1, sticky="e")

        ttk.Label(sidebar, text="Image List", style="FieldLabel.TLabel").grid(
            row=2, column=0, sticky="w", padx=12, pady=(14, 6)
        )
        list_panel = tk.Frame(
            sidebar,
            bg="#ffffff",
            highlightbackground="#dbe2ea",
            highlightthickness=1,
            bd=0,
        )
        list_panel.grid(row=3, column=0, sticky="nsew", padx=12)
        list_panel.rowconfigure(0, weight=1)
        list_panel.columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(
            list_panel,
            bd=0,
            highlightthickness=0,
            activestyle="none",
            bg="#ffffff",
            fg=COLORS["text_main"],
            selectbackground=COLORS["accent_soft"],
            selectforeground=COLORS["accent_dark"],
            font=("Segoe UI", 10),
            exportselection=False,
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        scroll = ttk.Scrollbar(list_panel, orient="vertical", command=self.listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.listbox.config(yscrollcommand=scroll.set)

        ttk.Button(
            sidebar,
            text="Load Selected",
            style="Secondary.TButton",
            command=self.load_selected,
        ).grid(row=4, column=0, sticky="ew", padx=12, pady=(10, 12))

        workspace = ttk.Frame(main, style="App.TFrame")
        workspace.grid(row=0, column=1, sticky="nsew")
        workspace.rowconfigure(0, weight=1)
        workspace.columnconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=1)

        self.input_label = self._build_preview_card(
            workspace,
            col=0,
            title="Input Preview",
            hint="Before dehazing",
            empty_text="Load an image to preview input.",
            padx=(0, 8),
        )
        self.output_label = self._build_preview_card(
            workspace,
            col=1,
            title="Output Preview",
            hint="After dehazing",
            empty_text="Click Dehaze to generate output.",
            padx=(8, 0),
        )

        actions = ttk.Frame(self.work_tab, style="App.TFrame", padding=(0, 0, 0, 0))
        actions.grid(row=1, column=0, sticky="w")
        ttk.Button(
            actions, text="Run Dehaze", style="Primary.TButton", command=self.run_dehaze
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            actions, text="Save Output", style="Secondary.TButton", command=self.save_output
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

    def _build_metrics_tab(self):
        self.metrics_tab.rowconfigure(1, weight=1)
        self.metrics_tab.columnconfigure(0, weight=1)

        top = ttk.Frame(self.metrics_tab, style="App.TFrame", padding=(0, 0, 0, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        self.metric_tree = ttk.Treeview(
            top,
            columns=("metric", "input", "output", "delta", "ratio"),
            show="headings",
            style="Metric.Treeview",
            height=6,
        )
        self.metric_tree.heading("metric", text="Metric")
        self.metric_tree.heading("input", text="Input")
        self.metric_tree.heading("output", text="Dehazed")
        self.metric_tree.heading("delta", text="Delta")
        self.metric_tree.heading("ratio", text="Improve %")
        self.metric_tree.column("metric", width=160, anchor="center")
        self.metric_tree.column("input", width=130, anchor="center")
        self.metric_tree.column("output", width=130, anchor="center")
        self.metric_tree.column("delta", width=130, anchor="center")
        self.metric_tree.column("ratio", width=130, anchor="center")
        self.metric_tree.grid(row=0, column=0, sticky="ew")

        ttk.Button(
            top,
            text="Export Metric Charts",
            style="Secondary.TButton",
            command=self.export_metric_charts,
        ).grid(row=0, column=1, sticky="e", padx=(10, 0))

        charts = ttk.Frame(self.metrics_tab, style="App.TFrame")
        charts.grid(row=1, column=0, sticky="nsew")
        charts.rowconfigure(0, weight=1)
        charts.columnconfigure(0, weight=1)
        charts.columnconfigure(1, weight=1)

        self.bar_card = self._build_chart_card(
            charts,
            col=0,
            title="Grouped Bar Chart",
            hint="Direct value comparison",
            padx=(0, 8),
        )
        self.radar_card = self._build_chart_card(
            charts,
            col=1,
            title="Radar Chart",
            hint="Normalized profile comparison",
            padx=(8, 0),
        )

    def _build_preview_card(self, parent, col, title, hint, empty_text, padx):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=0, column=col, sticky="nsew", padx=padx)
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        header = ttk.Frame(card, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=hint, style="CardHint.TLabel").grid(row=0, column=1, sticky="e")

        preview_panel = tk.Frame(
            card,
            bg="#ffffff",
            highlightbackground="#dbe2ea",
            highlightthickness=1,
            bd=0,
        )
        preview_panel.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        preview_panel.rowconfigure(0, weight=1)
        preview_panel.columnconfigure(0, weight=1)

        label = tk.Label(
            preview_panel,
            text=empty_text,
            bg="#ffffff",
            fg=COLORS["text_muted"],
            anchor="center",
            justify="center",
            font=("Segoe UI", 10),
        )
        label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        return label

    def _build_chart_card(self, parent, col, title, hint, padx):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=0, column=col, sticky="nsew", padx=padx)
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        header = ttk.Frame(card, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text=hint, style="CardHint.TLabel").grid(row=0, column=1, sticky="e")

        content = tk.Frame(
            card,
            bg="#ffffff",
            highlightbackground="#dbe2ea",
            highlightthickness=1,
            bd=0,
        )
        content.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        placeholder = tk.Label(
            content,
            text="Run Dehaze to generate this chart.",
            bg="#ffffff",
            fg=COLORS["text_muted"],
            font=("Segoe UI", 10),
        )
        placeholder.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        return content

    def set_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def pick_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.folder_var.set(folder)
        self.load_folder(folder)

    def load_folder(self, folder):
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Invalid folder path.")
            return

        self.images = []
        self.listbox.delete(0, tk.END)
        try:
            names = sorted(os.listdir(folder))
        except OSError as exc:
            messagebox.showerror("Error", str(exc))
            return

        for name in names:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                self.images.append(name)
                self.listbox.insert(tk.END, name)

        if self.images:
            self.listbox.selection_set(0)
            self.listbox.activate(0)
            self.load_selected()
        else:
            self.current_image = None
            self.current_path = None
            self.output_image = None
            self.metrics_result = None
            self.clear_input()
            self.clear_output()
            self.clear_metrics()
            self.set_status("No supported images found in this folder.")

    def on_select(self, _event):
        self.load_selected()

    def load_selected(self):
        if not self.images:
            messagebox.showinfo("Info", "No available images in this folder.")
            return
        selection = self.listbox.curselection()
        if not selection:
            return

        name = self.images[selection[0]]
        path = os.path.join(self.folder_var.get(), name)
        image = _read_image(path)
        if image is None:
            messagebox.showerror("Error", f"Failed to read image: {name}")
            return

        self.current_path = path
        self.current_image = image
        self.output_image = None
        self.metrics_result = None
        self.show_image(image, self.input_label, is_output=False)
        self.clear_output()
        self.clear_metrics()
        self.set_status(f"Loaded: {name}")

    def show_image(self, image_bgr, target_label, is_output):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        target_label.update_idletasks()
        max_w = max(260, target_label.winfo_width() - 24)
        max_h = max(220, target_label.winfo_height() - 24)
        if max_w <= 1 or max_h <= 1:
            max_w, max_h = DISPLAY_FALLBACK
        pil = _fit_image(pil, (max_w, max_h))
        tk_img = ImageTk.PhotoImage(pil)
        target_label.configure(image=tk_img, text="")
        if is_output:
            self.output_tk = tk_img
        else:
            self.input_tk = tk_img

    def clear_input(self):
        self.input_label.configure(image="", text="Load an image to preview input.")
        self.input_tk = None

    def clear_output(self):
        self.output_label.configure(image="", text="Click Dehaze to generate output.")
        self.output_tk = None

    def clear_metrics(self):
        for item in self.metric_tree.get_children():
            self.metric_tree.delete(item)
        self._render_chart(self.bar_card, None, "bar_canvas")
        self._render_chart(self.radar_card, None, "radar_canvas")

    def run_dehaze(self):
        if self.current_image is None:
            messagebox.showinfo("Info", "Please load an image first.")
            return
        self.set_status("Processing dehaze and metrics...")
        try:
            output = dehaze_bgr(self.current_image)
            metrics_result = calculate_metrics(self.current_image, output)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.set_status("Dehaze failed.")
            return

        self.output_image = output
        self.metrics_result = metrics_result
        self.show_image(output, self.output_label, is_output=True)
        self._update_metrics_tab()
        self.set_status("Dehaze complete. Metrics generated in Metrics tab.")

    def _update_metrics_tab(self):
        self.clear_metrics()
        rows = self.metrics_result.get("metrics", [])
        for row in rows:
            self.metric_tree.insert(
                "",
                "end",
                values=(
                    row["name"],
                    f"{row['input']:.6f}",
                    f"{row['output']:.6f}",
                    f"{row['delta']:+.6f}",
                    f"{row['improve_ratio']:+.2f}%",
                ),
            )

        bar_fig = build_bar_figure(self.metrics_result)
        radar_fig = build_radar_figure(self.metrics_result)
        self._render_chart(self.bar_card, bar_fig, "bar_canvas")
        self._render_chart(self.radar_card, radar_fig, "radar_canvas")

    def _render_chart(self, parent, figure, canvas_attr):
        old_canvas = getattr(self, canvas_attr, None)
        if old_canvas is not None:
            old_canvas.get_tk_widget().destroy()
            setattr(self, canvas_attr, None)

        if figure is None:
            placeholder = tk.Label(
                parent,
                text="Run Dehaze to generate this chart.",
                bg="#ffffff",
                fg=COLORS["text_muted"],
                font=("Segoe UI", 10),
            )
            placeholder.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            return

        for child in parent.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        setattr(self, canvas_attr, canvas)

    def save_output(self):
        if self.output_image is None:
            messagebox.showinfo("Info", "There is no dehazed output to save.")
            return

        initial = None
        if self.current_path:
            base, ext = os.path.splitext(os.path.basename(self.current_path))
            initial = f"{base}_dehazed{ext or '.png'}"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial,
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg;*.jpeg"),
                ("BMP Image", "*.bmp"),
                ("TIFF Image", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return
        try:
            _write_image(path, self.output_image)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.set_status("Save failed.")
            return
        self.set_status(f"Saved output: {os.path.basename(path)}")

    def export_metric_charts(self):
        if self.metrics_result is None:
            messagebox.showinfo("Info", "Run dehaze first to generate metric charts.")
            return

        folder = filedialog.askdirectory(title="Select folder to save charts")
        if not folder:
            return

        prefix = "metrics"
        if self.current_path:
            prefix = os.path.splitext(os.path.basename(self.current_path))[0]
        try:
            bar_path, radar_path = save_metric_figures(
                self.metrics_result, folder, prefix=f"{prefix}_metrics"
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.set_status("Metric chart export failed.")
            return

        self.set_status(
            f"Metric charts exported: {os.path.basename(bar_path)}, {os.path.basename(radar_path)}"
        )

    def _on_root_resize(self, event):
        if event.widget is not self.root:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._refresh_previews)

    def _refresh_previews(self):
        self._resize_job = None
        if self.current_image is not None:
            self.show_image(self.current_image, self.input_label, is_output=False)
        if self.output_image is not None:
            self.show_image(self.output_image, self.output_label, is_output=True)


def main():
    root = tk.Tk()
    DehazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

