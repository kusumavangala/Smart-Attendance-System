import os
import cv2
import datetime
import numpy as np
import pandas as pd
import face_recognition
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

DATASET_DIR = "student_dataset"
ATTENDANCE_FILE = "attendance.csv"

class SmartAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System (AI/CV)")
        self.root.geometry("1100x650")
        self.root.configure(bg="#2c3e50")

        self.is_running = False
        self.cap = None
        self.known_encodings = []
        self.known_names = []
        self.known_rolls = []
        self.marked_today = set()

        self.init_attendance_file()
        self.load_known_faces()
        self.build_ui()

    def init_attendance_file(self):
        if not os.path.exists(ATTENDANCE_FILE):
            df = pd.DataFrame(columns=["RollNo", "Name", "Date", "Time", "Status"])
            df.to_csv(ATTENDANCE_FILE, index=False)
        else:
            df = pd.read_csv(ATTENDANCE_FILE)
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            today_records = df[df["Date"] == today_str]
            self.marked_today = set(today_records["RollNo"].tolist())

    def load_known_faces(self):
        print("[INFO] Encoding student faces. Please wait...")
        if not os.path.exists(DATASET_DIR):
            messagebox.showerror("Error", "Dataset folder not found! Run setup_data.py first.")
            return

        for file in os.listdir(DATASET_DIR):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                parts = os.path.splitext(file)[0].split("_", 1)
                roll_no = parts[0]
                name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

                img_path = os.path.join(DATASET_DIR, file)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    self.known_rolls.append(roll_no)

        print(f"[INFO] Successfully loaded and encoded {len(self.known_encodings)} student faces.")

    def build_ui(self):
        header = tk.Label(self.root, text="🎓 AI Smart Attendance System", 
                          font=("Helvetica", 20, "bold"), fg="#ecf0f1", bg="#34495e", pady=10)
        header.pack(fill=tk.X)

        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        left_frame = tk.Frame(main_frame, bg="#1a252f", width=640, height=480)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.video_label = tk.Label(left_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = tk.Frame(left_frame, bg="#1a252f")
        btn_frame.pack(fill=tk.X, pady=8)

        self.start_btn = tk.Button(btn_frame, text="▶ Start Taking Attendance", 
                                   font=("Helvetica", 12, "bold"), bg="#27ae60", fg="white",
                                   command=self.start_attendance, padx=15, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop Camera", 
                                  font=("Helvetica", 12, "bold"), bg="#c0392b", fg="white",
                                  command=self.stop_attendance, state=tk.DISABLED, padx=15, pady=5)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        right_frame = tk.Frame(main_frame, bg="#34495e", width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        table_title = tk.Label(right_frame, text="📋 Today's Attendance Log", 
                               font=("Helvetica", 14, "bold"), fg="#ecf0f1", bg="#34495e", pady=5)
        table_title.pack(fill=tk.X)

        columns = ("RollNo", "Name", "Time", "Status")
        self.tree = ttk.Treeview(right_frame, columns=columns, show="headings", height=18)
        self.tree.heading("RollNo", text="Roll No")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Time", text="Time")
        self.tree.heading("Status", text="Status")

        self.tree.column("RollNo", width=70, anchor=tk.CENTER)
        self.tree.column("Name", width=120)
        self.tree.column("Time", width=70, anchor=tk.CENTER)
        self.tree.column("Status", width=80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_table()

    def start_attendance(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.")
                return
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.process_frame()

    def stop_attendance(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image="")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def mark_attendance(self, roll_no, name):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if roll_no not in self.marked_today:
            new_record = pd.DataFrame([{
                "RollNo": roll_no,
                "Name": name,
                "Date": date_str,
                "Time": time_str,
                "Status": "Present"
            }])
            new_record.to_csv(ATTENDANCE_FILE, mode='a', header=not os.path.exists(ATTENDANCE_FILE), index=False)
            self.marked_today.add(roll_no)
            self.tree.insert("", 0, values=(roll_no, name, time_str, "Present"))

    def process_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                label = "Random Unidentified Person"
                box_color = (0, 0, 255)

                if len(self.known_encodings) > 0:
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if face_distances[best_match_index] < 0.50:
                        name = self.known_names[best_match_index]
                        roll_no = self.known_rolls[best_match_index]
                        label = f"{name} ({roll_no})"
                        box_color = (0, 255, 0)
                        self.mark_attendance(roll_no, name)

                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 8), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        self.root.after(15, self.process_frame)

    def refresh_table(self):
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_csv(ATTENDANCE_FILE)
            today_str = datetime.date.today().strftime("%Y-%m-%d")
            today_df = df[df["Date"] == today_str]

            for _, row in today_df.iterrows():
                self.tree.insert("", "end", values=(row["RollNo"], row["Name"], row["Time"], row["Status"]))

    def on_closing(self):
        self.stop_attendance()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartAttendanceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()