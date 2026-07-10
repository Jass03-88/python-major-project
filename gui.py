import ttkbootstrap as tb
from ttkbootstrap.widgets import ToolTip
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime
import os

# 1. Define UI Colors & Dimensions First
CAMERA_W, CAMERA_H = 480, 360

# 2. Build and Paint the Window Immediately
root = tb.Window(themename="darkly")
root.title("Biometric Attendance System")
root.geometry("980x760")

# Force the OS to draw the window instantly before anything else happens!
root.update()

# 3. NOW Import the Heavy Computer Vision & AI Modules
import cv2
from PIL import Image, ImageTk
import config
import security_utils
import telegram_utils
import attendance_manager
import recognition_core
import env_utils

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

camera_busy = False
import threading
import queue

ui_queue = queue.Queue()
stop_event = threading.Event()

def camera_worker_loop(session_type, user_id=None):
    try:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            ui_queue.put({"type": "error"})
            return
            
        if session_type == "login":
            session = recognition_core.LoginSession()
        else:
            session = recognition_core.RegistrationSession(user_id)
            
        ui_queue.put({"type": "started", "session": session})

        while not stop_event.is_set():
            ret, frame = cam.read()
            if not ret:
                ui_queue.put({"type": "error"})
                break
            
            annotated = session.process_frame(frame)
            
            frame_resized = cv2.resize(annotated, (CAMERA_W, CAMERA_H))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            ui_queue.put({
                "type": "frame",
                "image": img,
                "is_complete": session.is_complete,
                "count": getattr(session, 'count', 0)
            })
            
            if session.is_complete:
                break
    except Exception as e:
        import traceback
        traceback.print_exc()
        ui_queue.put({"type": "error"})
    finally:
        if 'cam' in locals() and cam.isOpened():
            cam.release()

def async_retrain_model(on_complete):
    def worker():
        ok, msg = recognition_core.retrain_model()
        root.after(0, lambda: on_complete(ok, msg))
    threading.Thread(target=worker, daemon=True).start()

# ------------------------------------------------------------------
# Small reusable UI helpers
# ------------------------------------------------------------------
def play_sound(kind):
    """kind: 'granted' or 'denied'."""
    try:
        if HAS_WINSOUND:
            if kind == "granted":
                winsound.Beep(1200, 150)
            else:
                winsound.Beep(400, 300)
        else:
            root.bell()
    except Exception:
        pass

def show_toast(message, kind="info", duration_ms=3000):
    """Small auto-dismissing popup in the corner."""
    bootstyle = "inverse-" + ("danger" if kind == "error" else kind)
    toast = tb.Toplevel(root)
    toast.overrideredirect(True)
    toast.attributes("-alpha", 0.95)
    
    label = tb.Label(
        toast,
        text=message,
        bootstyle=bootstyle,
        font=("Segoe UI", 10, "bold"),
        wraplength=320,
        justify="left"
    )
    label.pack(padx=16, pady=10)
    root.update_idletasks()
    x = root.winfo_rootx() + root.winfo_width() - 340
    y = root.winfo_rooty() + root.winfo_height() - 100
    toast.geometry(f"+{x}+{y}")
    toast.after(duration_ms, toast.destroy)

def cv2_to_tk(frame):
    frame_resized = cv2.resize(frame, (CAMERA_W, CAMERA_H))
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(image=img)

def set_camera_placeholder(text="Camera preview will appear here"):
    placeholder_img = Image.new("RGB", (CAMERA_W, CAMERA_H), "black")
    photo = ImageTk.PhotoImage(image=placeholder_img)
    camera_label.config(
        image=photo,
        text=text,
        font=("Segoe UI", 12),
        compound="center",
    )
    camera_label.image = photo

# ------------------------------------------------------------------
# Attendance table
# ------------------------------------------------------------------
def refresh_attendance_table():
    for row in attendance_tree.get_children():
        attendance_tree.delete(row)
    rows = attendance_manager.get_today_summary()
    for name, check_in, check_out, status in rows:
        checkout_display = check_out if check_out else "— still in —"
        attendance_tree.insert(
            "", "end", values=(name, check_in, checkout_display, status)
        )

def auto_refresh_attendance():
    refresh_attendance_table()
    root.after(10000, auto_refresh_attendance)

# ------------------------------------------------------------------
# Face login flow
# ------------------------------------------------------------------
login_cam = None
login_session = None

def set_main_buttons_state(state):
    login_button.config(state=state)
    admin_button.config(state=state)

def start_login():
    global login_cam, login_session, camera_busy
    if camera_busy:
        show_toast("Camera is already in use — finish the other action first.", "error")
        return
    if security_utils.is_locked_out():
        start_lockout_countdown()
        return

    camera_busy = True
    set_main_buttons_state("disabled")
    status_label.config(text="Scanning face...", bootstyle="success")
    stop_event.clear()
    threading.Thread(target=camera_worker_loop, args=("login",), daemon=True).start()
    poll_login_queue()

def poll_login_queue():
    global login_session, camera_busy
    if not camera_busy:
        return
        
    try:
        while True:
            msg = ui_queue.get_nowait()
            if msg["type"] == "error":
                finish_login(no_camera=True)
                return
            elif msg["type"] == "started":
                login_session = msg["session"]
            elif msg["type"] == "frame":
                photo = ImageTk.PhotoImage(image=msg["image"])
                camera_label.config(image=photo, text="")
                camera_label.image = photo

                if msg["is_complete"]:
                    finish_login()
                    return
    except queue.Empty:
        pass

    root.after(15, poll_login_queue)

def finish_login(no_camera=False):
    global login_session, camera_busy
    stop_event.set()
    if no_camera:
        status_label.config(
            text="Camera error — could not read frame.", bootstyle="danger"
        )
        set_camera_placeholder("Camera error")
    else:
        result = login_session.finalize()
        outcome = recognition_core.handle_login_result(result)
        if outcome["granted"]:
            status_label.config(text=outcome["message"], bootstyle="success")
            play_sound("granted")
            show_toast(outcome["message"], "success")
        else:
            status_label.config(text=outcome["message"], bootstyle="danger")
            play_sound("denied")
            show_toast(outcome["message"], "error")
        set_camera_placeholder("Scan complete — camera off")
        refresh_attendance_table()

    login_session = None
    camera_busy = False
    set_main_buttons_state("normal")

def start_lockout_countdown():
    remaining = security_utils.seconds_until_unlock()
    login_button.config(state="disabled")
    def tick(secs_left):
        if secs_left <= 0:
            login_button.config(text="📷 FACE LOGIN", state="normal")
            status_label.config(text="System Ready", bootstyle="success")
            return
        login_button.config(text=f"🔒 Locked ({secs_left}s)")
        status_label.config(
            text=f"Too many failed attempts. Locked for {secs_left}s.",
            bootstyle="danger"
        )
        root.after(1000, lambda: tick(secs_left - 1))
    tick(remaining)

# ------------------------------------------------------------------
# Admin login
# ------------------------------------------------------------------
def admin_setup():
    if not config.ADMIN_PASSWORD_HASH:
        messagebox.showerror(
            "Not configured",
            "No admin password is set yet.\n\nRun this once from a terminal:\n"
            "python set_admin_password.py"
        )
        return
    if security_utils.is_locked_out():
        wait_s = security_utils.seconds_until_unlock()
        show_toast(f"Locked out. Try again in {wait_s}s.", "error")
        return
    password = simpledialog.askstring("Admin Login", "Enter Admin Password:", show="*")
    if password is None:
        return
    if security_utils.verify_password(
        password, config.ADMIN_PASSWORD_SALT, config.ADMIN_PASSWORD_HASH
    ):
        security_utils.reset_failures()
        telegram_utils.send_message(
            f"🔑 Admin panel accessed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        root.after(10, open_admin_panel)
    else:
        triggered = security_utils.record_failure(
            config.LOCKOUT_THRESHOLD,
            config.LOCKOUT_WINDOW_SECONDS,
            config.LOCKOUT_DURATION_SECONDS,
        )
        if triggered:
            telegram_utils.send_message(
                f"🔒 Admin panel locked for {config.LOCKOUT_DURATION_SECONDS}s after repeated wrong passwords."
            )
            show_toast("Too many wrong attempts. Admin panel locked.", "error")
        else:
            show_toast("Wrong password.", "error")

# ------------------------------------------------------------------
# Admin panel
# ------------------------------------------------------------------
def open_admin_panel():
    admin_window = tb.Toplevel(root)
    admin_window.title("Admin Panel")
    admin_window.geometry("640x780")

    tb.Label(
        admin_window,
        text="ADMIN PANEL",
        font=("Segoe UI", 18, "bold")
    ).pack(pady=(16, 8))

    notebook = tb.Notebook(admin_window)
    notebook.pack(fill="both", expand=True, padx=16, pady=8)

    dashboard_tab = tb.Frame(notebook)
    register_tab = tb.Frame(notebook)
    users_tab = tb.Frame(notebook)
    settings_tab = tb.Frame(notebook)
    telegram_tab = tb.Frame(notebook)

    notebook.add(dashboard_tab, text="Dashboard / Export")
    notebook.add(register_tab, text="Register / Train")
    notebook.add(users_tab, text="Manage Users")
    notebook.add(settings_tab, text="Settings")
    notebook.add(telegram_tab, text="Telegram")

    build_dashboard_tab(dashboard_tab)
    build_register_tab(register_tab)
    build_users_tab(users_tab)
    build_settings_tab(settings_tab)
    build_telegram_tab(telegram_tab)

def build_dashboard_tab(parent):
    tb.Label(
        parent,
        text="Attendance Dashboard",
        font=("Segoe UI", 13, "bold")
    ).pack(pady=(16, 8))

    def export_csv():
        path = attendance_manager.export_to_csv()
        show_toast(f"Exported to {path}", "success")

    tb.Button(
        parent,
        text="📥 Export All to CSV",
        bootstyle="success",
        command=export_csv,
    ).pack(pady=8)

    # Plot
    counts = attendance_manager.get_daily_counts(7)
    if not counts:
        tb.Label(parent, text="No attendance data to display.", bootstyle="warning").pack(pady=20)
        return

    dates = [row[0] for row in counts]
    people = [row[1] for row in counts]

    # Dark theme figure
    fig = Figure(figsize=(6, 4), dpi=100, facecolor="#222222")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#222222")
    ax.bar(dates, people, color="#3498db")
    
    ax.set_title("Unique Attendees (Last 7 Days)", color="white")
    ax.set_ylabel("People", color="white")
    ax.tick_params(colors="white")
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('#222222') 
    ax.spines['right'].set_color('#222222')
    ax.spines['left'].set_color('white')
    
    fig.autofmt_xdate(rotation=45)

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=16, fill="both", expand=True)

def build_register_tab(parent):
    global camera_busy
    tb.Label(
        parent,
        text="Register a new face",
        font=("Segoe UI", 13, "bold")
    ).pack(pady=(16, 8))

    id_frame = tb.Frame(parent)
    id_frame.pack(pady=4)
    tb.Label(
        id_frame, text="Faculty ID:"
    ).pack(side="left", padx=6)
    id_entry = tb.Entry(id_frame, width=20)
    id_entry.pack(side="left")

    reg_camera_label = tb.Label(parent)
    reg_camera_label.pack(pady=10)

    placeholder_img = Image.new("RGB", (CAMERA_W, CAMERA_H), "black")
    photo = ImageTk.PhotoImage(image=placeholder_img)
    reg_camera_label.config(
        image=photo,
        text="Camera preview will appear here",
        font=("Segoe UI", 12),
        compound="center",
    )
    reg_camera_label.image = photo
    progress = tb.Progressbar(
        parent, orient="horizontal", length=300, mode="determinate", maximum=15, bootstyle="success"
    )
    progress.pack(pady=6)
    progress_label = tb.Label(
        parent, text="0 / 15 photos"
    )
    progress_label.pack()

    reg_status = tb.Label(parent, text="", bootstyle="success")
    reg_status.pack(pady=6)

    state = {"cam": None, "session": None}

    def poll_reg_queue():
        global camera_busy
        if not camera_busy:
            return
            
        try:
            while True:
                msg = ui_queue.get_nowait()
                if msg["type"] == "error":
                    finish_registration(error=True)
                    return
                elif msg["type"] == "started":
                    state["session"] = msg["session"]
                elif msg["type"] == "frame":
                    photo = ImageTk.PhotoImage(image=msg["image"])
                    reg_camera_label.config(image=photo, text="")
                    reg_camera_label.image = photo
                    progress["value"] = msg["count"]
                    progress_label.config(text=f"{msg['count']} / 15 photos")

                    if msg["is_complete"]:
                        finish_registration()
                        return
        except queue.Empty:
            pass
            
        parent.after(15, poll_reg_queue)

    def finish_registration(error=False):
        global camera_busy
        stop_event.set()
        camera_busy = False
        start_btn.config(state="normal")
        if error:
            reg_status.config(
                text="Camera error during registration.", bootstyle="danger"
            )
            return
        reg_status.config(
            text=f"Registration complete for '{state['session'].user_id}'. Retraining model...",
            bootstyle="success",
        )
        parent.update_idletasks()

        def on_done(ok, msg):
            reg_status.config(
                text=msg, bootstyle="success" if ok else "danger"
            )
            if ok:
                telegram_utils.send_message(
                    f"✅ New face registered and model retrained for user: {state['session'].user_id}"
                )
                show_toast(msg, "success")
        async_retrain_model(on_done)

    def start_registration():
        global camera_busy
        user_id = id_entry.get().strip()
        if not user_id:
            show_toast("Enter a faculty ID first.", "error")
            return
        if camera_busy:
            show_toast(
                "Camera is already in use — finish the other action first.", "error"
            )
            return
        camera_busy = True
        start_btn.config(state="disabled")
        progress["value"] = 0
        reg_status.config(
            text="Registration started. Please look at the camera...",
            bootstyle="success",
        )
        stop_event.clear()
        threading.Thread(target=camera_worker_loop, args=("register", user_id), daemon=True).start()
        poll_reg_queue()

    start_btn = tb.Button(
        parent,
        text="▶ Start Registration",
        bootstyle="primary",
        command=start_registration,
    )
    start_btn.pack(pady=10)

    tb.Label(
        parent,
        text="Train the model on everyone currently in dataset/ (no new photos captured):",
        wraplength=500,
    ).pack(pady=(20, 4))

    def train_only():
        train_btn.config(state="disabled")
        train_progress.start(10)
        parent.update_idletasks()
        def on_done(ok, msg):
            train_progress.stop()
            train_btn.config(state="normal")
            show_toast(msg, "success" if ok else "error")
        async_retrain_model(on_done)

    train_btn = tb.Button(
        parent,
        text="🧠 Train Model Only",
        bootstyle="secondary",
        command=train_only,
    )
    train_btn.pack(pady=6)

    train_progress = tb.Progressbar(
        parent, orient="horizontal", length=300, mode="indeterminate", bootstyle="info-striped"
    )
    train_progress.pack(pady=6)


def build_users_tab(parent):
    tb.Label(
        parent,
        text="Registered Users",
        font=("Segoe UI", 13, "bold"),
    ).pack(pady=(16, 8))

    listbox = tk.Listbox(parent, width=40, height=14, font=("Segoe UI", 11))
    listbox.pack(pady=8)

    def refresh_users():
        listbox.delete(0, "end")
        for user in recognition_core.list_registered_users():
            listbox.insert("end", user)
    refresh_users()

    btn_frame = tb.Frame(parent)
    btn_frame.pack(pady=8)

    def delete_selected():
        selection = listbox.curselection()
        if not selection:
            show_toast("Select a user first.", "error")
            return
        user_id = listbox.get(selection[0])
        if not messagebox.askyesno(
            "Confirm delete",
            f"Permanently delete all photos for '{user_id}' and retrain the model?",
        ):
            return
        recognition_core.delete_user(user_id)
        refresh_users()
        show_toast(f"Deleted '{user_id}'. Retraining model in background...", "info")
        def on_done(ok, msg):
            show_toast(msg, "success" if ok else "error")
        async_retrain_model(on_done)

    tb.Button(
        btn_frame,
        text="🔄 Refresh",
        bootstyle="info",
        command=refresh_users,
    ).grid(row=0, column=0, padx=8)
    
    tb.Button(
        btn_frame,
        text="🗑 Delete Selected",
        bootstyle="danger",
        command=delete_selected,
    ).grid(row=0, column=1, padx=8)

def build_settings_tab(parent):
    tb.Label(
        parent,
        text="Settings",
        font=("Segoe UI", 13, "bold"),
    ).pack(pady=(16, 8))
    tb.Label(
        parent,
        text="Changes are saved to .env and take effect the next time the app is started.",
        bootstyle="warning",
        wraplength=500,
    ).pack(pady=(0, 12))

    form = tb.Frame(parent)
    form.pack(pady=6)

    fields = [
        ("Confidence threshold", "CONFIDENCE_THRESHOLD", str(config.CONFIDENCE_THRESHOLD), "Lower = stricter face matching (e.g. 0.3 means faces must look very similar)"),
        ("Lockout threshold", "LOCKOUT_THRESHOLD", str(config.LOCKOUT_THRESHOLD), "Number of failed attempts before temporary lockout"),
        ("Lockout window", "LOCKOUT_WINDOW_SECONDS", str(config.LOCKOUT_WINDOW_SECONDS), "Time window to accumulate failed attempts (seconds)"),
        ("Lockout duration", "LOCKOUT_DURATION_SECONDS", str(config.LOCKOUT_DURATION_SECONDS), "How long the system stays locked out (seconds)"),
        ("Denial alert cooldown", "DENIAL_ALERT_COOLDOWN_SECONDS", str(config.DENIAL_ALERT_COOLDOWN_SECONDS), "Wait time before sending another Telegram alert for denied access (seconds)"),
        ("Half-day cutoff (HH:MM)", "HALF_DAY_CUTOFF", config.HALF_DAY_CUTOFF, "Check-ins after this time will be marked as 'Half Day'"),
    ]

    entries = {}
    for i, (label_text, key, default_val, tooltip_text) in enumerate(fields):
        lbl = tb.Label(form, text=label_text)
        lbl.grid(row=i, column=0, sticky="w", pady=4, padx=6)
        
        # Add tooltip to the label
        ToolTip(lbl, text=tooltip_text, bootstyle="info")
        
        entry = tb.Entry(form, width=15)
        entry.insert(0, default_val)
        entry.grid(row=i, column=1, pady=4, padx=6)
        entries[key] = entry

    liveness_var = tk.BooleanVar(value=config.LIVENESS_ENABLED)
    tb.Checkbutton(
        form,
        text="Liveness (blink) check enabled",
        variable=liveness_var,
        bootstyle="success-round-toggle"
    ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=8, padx=6)

    def save_settings():
        for key, entry in entries.items():
            env_utils.upsert_env_var(key, entry.get().strip())
        env_utils.upsert_env_var(
            "LIVENESS_ENABLED", "true" if liveness_var.get() else "false"
        )
        show_toast("Settings saved. Restart the app for changes to take effect.", "success")

    tb.Button(
        parent,
        text="💾 Save Settings",
        bootstyle="success",
        command=save_settings,
    ).pack(pady=16)

    def change_password():
        old_pwd = simpledialog.askstring("Change Password", "Enter Current Admin Password:", show="*")
        if not old_pwd: return
        if not security_utils.verify_password(old_pwd, config.ADMIN_PASSWORD_SALT, config.ADMIN_PASSWORD_HASH):
            show_toast("Current password incorrect.", "error")
            return
        new_pwd = simpledialog.askstring("Change Password", "Enter New Admin Password:", show="*")
        if not new_pwd: return
        confirm = simpledialog.askstring("Change Password", "Confirm New Password:", show="*")
        if not confirm: return
        if new_pwd != confirm:
            show_toast("Passwords didn't match.", "error")
            return
        salt = security_utils.generate_salt()
        pwd_hash = security_utils.hash_password(new_pwd, salt)
        env_utils.upsert_env_var("ADMIN_PASSWORD_SALT", salt)
        env_utils.upsert_env_var("ADMIN_PASSWORD_HASH", pwd_hash)
        config.ADMIN_PASSWORD_SALT = salt
        config.ADMIN_PASSWORD_HASH = pwd_hash
        show_toast("Admin password successfully changed!", "success")

    tb.Button(
        parent,
        text="🔑 Change Admin Password",
        bootstyle="warning",
        command=change_password,
    ).pack(pady=(10, 16))


def build_telegram_tab(parent):
    tb.Label(
        parent,
        text="Telegram Alerts",
        font=("Segoe UI", 13, "bold")
    ).pack(pady=(16, 8))

    configured = bool(config.TELEGRAM_BOT_TOKEN) and bool(config.TELEGRAM_CHAT_ID)
    status_text = (
        "✅ Configured"
        if configured
        else "⚠️ Not configured — add TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID to .env"
    )
    tb.Label(
        parent,
        text=status_text,
        bootstyle="success" if configured else "danger"
    ).pack(pady=6)

    def send_test():
        if not configured:
            show_toast("Telegram not configured in .env.", "error")
            return
        telegram_utils.send_message("✅ Test message from Biometric Attendance System.")
        show_toast("Test message sent — check your Telegram.", "success")

    tb.Button(
        parent,
        text="📨 Send Test Message",
        bootstyle="info",
        command=send_test,
    ).pack(pady=10)


# ------------------------------------------------------------------
# Main window layout
# ------------------------------------------------------------------
tb.Label(
    root,
    text="BIOMETRIC ATTENDANCE SYSTEM",
    font=("Segoe UI", 22, "bold")
).pack(pady=(16, 2))

tb.Label(
    root,
    text="Face Recognition Attendance & Authentication",
    font=("Segoe UI", 12),
    bootstyle="info"
).pack(pady=2)

time_label = tb.Label(
    root, font=("Segoe UI", 13, "bold"), bootstyle="secondary"
)
time_label.pack(pady=6)

def update_time():
    time_label.config(text=datetime.now().strftime("%d-%m-%Y   %H:%M:%S"))
    root.after(1000, update_time)

update_time()

camera_label = tb.Label(root)
camera_label.pack(pady=8)
set_camera_placeholder()

status_label = tb.Label(
    root,
    text="System Ready",
    font=("Segoe UI", 13),
    bootstyle="success"
)
status_label.pack(pady=6)

button_frame = tb.Frame(root)
button_frame.pack(pady=10)

login_button = tb.Button(
    button_frame,
    text="📷 FACE LOGIN",
    width=20,
    bootstyle="primary",
    command=start_login,
)
login_button.grid(row=0, column=0, padx=14, pady=8)

admin_button = tb.Button(
    button_frame,
    text="🛡 ADMIN SETUP",
    width=20,
    bootstyle="secondary",
    command=admin_setup,
)
admin_button.grid(row=0, column=1, padx=14, pady=8)

exit_button = tb.Button(
    root,
    text="🚪 EXIT",
    width=14,
    bootstyle="danger",
    command=root.quit,
)
exit_button.pack(pady=10)

# ------------------------------------------------------------------
# Attendance table (today)
# ------------------------------------------------------------------
table_frame = tb.Frame(root)
table_frame.pack(pady=10, fill="x", padx=30)

tb.Label(
    table_frame,
    text="Today's Attendance",
    font=("Segoe UI", 13, "bold")
).pack(anchor="w")

columns = ("Name", "Check-In", "Check-Out", "Status")
attendance_tree = tb.Treeview(table_frame, columns=columns, show="headings", height=6, bootstyle="info")
for col in columns:
    attendance_tree.heading(col, text=col)
    attendance_tree.column(col, width=150, anchor="center")
attendance_tree.pack(fill="x", pady=6)

refresh_btn = tb.Button(
    table_frame,
    text="🔄 Refresh",
    bootstyle="info",
    command=refresh_attendance_table,
)
refresh_btn.pack(pady=4)

refresh_attendance_table()
auto_refresh_attendance()

root.mainloop()
