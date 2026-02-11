import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from dehaze import dehaze_bgr


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
    new_size = (int(w * scale), int(h * scale))
    return pil_image.resize(new_size, Image.LANCZOS)


class DehazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像去雾系统")
        self.root.geometry("1280x780")
        self.root.minsize(1080, 680)

        self.folder_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪，请先选择文件夹。")
        self.images = []
        self.current_path = None
        self.current_image = None
        self.output_image = None
        self.input_tk = None
        self.output_tk = None
        self._resize_job = None

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
        style.configure("HeaderTitle.TLabel", background=COLORS["header_bg"])
        style.configure(
            "HeaderTitle.TLabel",
            foreground=COLORS["header_text"],
            font=("Segoe UI", 19, "bold"),
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
            foreground=[("disabled", "#e2e8f0")],
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

    def _build_ui(self):
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_box = ttk.Frame(header, style="Header.TFrame")
        title_box.grid(row=0, column=0, sticky="w")
        ttk.Label(title_box, text="图像去雾系统", style="HeaderTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            title_box,
            text="支持图片浏览、处理预览与结果导出",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        main = ttk.Frame(self.root, style="App.TFrame", padding=(14, 12))
        main.grid(row=1, column=0, sticky="nsew")
        main.rowconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)

        sidebar = ttk.LabelFrame(main, text="图片资源", style="Sidebar.TLabelframe")
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text="文件夹", style="FieldLabel.TLabel").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 4)
        )
        folder_row = ttk.Frame(sidebar, style="App.TFrame")
        folder_row.grid(row=1, column=0, sticky="ew", padx=12)
        folder_row.columnconfigure(0, weight=1)

        folder_entry = ttk.Entry(folder_row, textvariable=self.folder_var, style="Field.TEntry")
        folder_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(
            folder_row, text="浏览", style="Secondary.TButton", command=self.pick_folder
        ).grid(row=0, column=1, sticky="e")

        ttk.Label(sidebar, text="图片列表", style="FieldLabel.TLabel").grid(
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
            text="加载选中",
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
            title="原图预览",
            hint="待处理图像",
            empty_text="请选择并加载图片",
            padx=(0, 8),
        )
        self.output_label = self._build_preview_card(
            workspace,
            col=1,
            title="结果预览",
            hint="处理后图像",
            empty_text="点击处理后显示结果",
            padx=(8, 0),
        )

        actions = ttk.Frame(self.root, style="App.TFrame", padding=(14, 0, 14, 10))
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="开始处理", style="Primary.TButton", command=self.run_dehaze).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            actions, text="保存结果", style="Secondary.TButton", command=self.save_output
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel")
        status.grid(row=3, column=0, sticky="ew")

    def _build_preview_card(self, parent, col, title, hint, empty_text, padx):
        card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        card.grid(row=0, column=col, sticky="nsew", padx=padx)
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)

        header = ttk.Frame(card, style="Card.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
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
            messagebox.showerror("错误", "文件夹路径无效。")
            return

        self.images = []
        self.listbox.delete(0, tk.END)
        try:
            names = sorted(os.listdir(folder))
        except OSError as exc:
            messagebox.showerror("错误", str(exc))
            return

        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                self.images.append(name)
                self.listbox.insert(tk.END, name)

        if self.images:
            self.listbox.selection_set(0)
            self.listbox.activate(0)
            self.load_selected()
        else:
            self.current_image = None
            self.current_path = None
            self.clear_input()
            self.clear_output()
            self.set_status("所选文件夹中未找到支持的图片格式。")

    def on_select(self, _event):
        self.load_selected()

    def load_selected(self):
        if not self.images:
            messagebox.showinfo("提示", "当前文件夹中没有可用图片。")
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        name = self.images[selection[0]]
        path = os.path.join(self.folder_var.get(), name)
        image = _read_image(path)
        if image is None:
            messagebox.showerror("错误", f"读取图片失败：{name}")
            return
        self.current_path = path
        self.current_image = image
        self.output_image = None
        self.show_image(image, self.input_label, is_output=False)
        self.clear_output()
        self.set_status(f"已加载：{name}")

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
        self.input_label.configure(image="", text="请选择并加载图片")
        self.input_tk = None

    def clear_output(self):
        self.output_label.configure(image="", text="点击处理后显示结果")
        self.output_tk = None

    def run_dehaze(self):
        if self.current_image is None:
            messagebox.showinfo("提示", "请先加载一张图片。")
            return
        self.set_status("处理中，请稍候...")
        try:
            output = dehaze_bgr(self.current_image)
        except Exception as exc:
            messagebox.showerror("错误", str(exc))
            self.set_status("处理失败。")
            return
        self.output_image = output
        self.show_image(output, self.output_label, is_output=True)
        self.set_status("处理完成。")

    def save_output(self):
        if self.output_image is None:
            messagebox.showinfo("提示", "当前没有可保存的结果。")
            return
        initial = None
        if self.current_path:
            base, ext = os.path.splitext(os.path.basename(self.current_path))
            initial = f"{base}_dehazed{ext or '.png'}"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial,
            filetypes=[
                ("PNG 图片", "*.png"),
                ("JPEG 图片", "*.jpg;*.jpeg"),
                ("BMP 图片", "*.bmp"),
                ("TIFF 图片", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return
        try:
            _write_image(path, self.output_image)
        except Exception as exc:
            messagebox.showerror("错误", str(exc))
            self.set_status("保存失败。")
            return
        self.set_status(f"已保存：{os.path.basename(path)}")

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
