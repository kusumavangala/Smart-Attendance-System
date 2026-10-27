
import os
import base64
import datetime
import threading

import cv2
import numpy as np
import pandas as pd
import face_recognition
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

DATASET_DIR = "student_dataset"
ATTENDANCE_FILE = "attendance.csv"
THRESHOLD = 0.50

known_encodings = []
known_names = []
known_rolls = []
marked_today = set()
attendance_lock = threading.Lock()


def load_known_faces():
    known_encodings.clear()
    known_names.clear()
    known_rolls.clear()

    if not os.path.isdir(DATASET_DIR):
        return

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        parts = os.path.splitext(filename)[0].split("_", 1)
        roll = parts[0]
        name = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

        try:
            image = face_recognition.load_image_file(
                os.path.join(DATASET_DIR, filename)
            )
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
                known_rolls.append(roll)
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")

    print(f"Loaded {len(known_encodings)} recognizable face(s).")


def get_today_records():
    if not os.path.exists(ATTENDANCE_FILE):
        return []

    try:
        df = pd.read_csv(ATTENDANCE_FILE).fillna("")
        today = datetime.date.today().isoformat()
        return df[df["Date"].astype(str) == today].to_dict("records")
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return []


def mark_attendance(roll, name):
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now()

    with attendance_lock:
        # Reload today's records to avoid duplicate entries.
        existing = get_today_records()
        present_rolls = {str(row["RollNo"]) for row in existing}

        if roll in present_rolls:
            return False

        record = {
            "RollNo": roll,
            "Name": name,
            "Date": today,
            "Time": now.strftime("%H:%M:%S"),
            "Status": "Present",
        }

        columns = ["RollNo", "Name", "Date", "Time", "Status"]
        pd.DataFrame([record], columns=columns).to_csv(
            ATTENDANCE_FILE,
            mode="a",
            header=not os.path.exists(ATTENDANCE_FILE)
            or os.path.getsize(ATTENDANCE_FILE) == 0,
            index=False,
        )
        marked_today.add(roll)
        return True


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Attendance System</title>
<style>
body{margin:0;background:#101827;color:#e5e7eb;font-family:Arial,sans-serif}
main{max-width:1050px;margin:35px auto;padding:20px}
h1{margin-bottom:6px} .muted{color:#9ca3af}
.layout{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}
.card{background:#1f2937;padding:20px;border-radius:14px}
video{width:100%;border-radius:10px;background:#000}
button{padding:11px 16px;border:0;border-radius:8px;cursor:pointer;
font-weight:bold;margin:8px 5px 8px 0}
.start{background:#22c55e;color:#052e16}.stop{background:#ef4444;color:white}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:10px 5px;border-bottom:1px solid #374151}
.status{color:#86efac;min-height:24px}
@media(max-width:750px){.layout{grid-template-columns:1fr}main{margin:10px auto}}
</style>
</head>
<body>
<main>
<h1>Smart Attendance System</h1>
<p class="muted">Face recognition · OpenCV · Flask</p>
<div class="layout">
<section class="card">
<h2>Camera</h2>
<video id="video" autoplay playsinline muted></video>
<div>
<button class="start" id="start">Start Camera</button>
<button class="stop" id="stop">Stop Camera</button>
</div>
<p class="status" id="status">Camera stopped.</p>
<p class="muted">Camera frames are sent to this server for face matching.</p>
</section>
<section class="card">
<h2>Today's Attendance</h2>
<p id="count">Loading records...</p>
<div style="overflow-x:auto">
<table>
<thead><tr><th>Roll No.</th><th>Name</th><th>Time</th><th>Status</th></tr></thead>
<tbody id="records"></tbody>
</table>
</div>
</section>
</div>
</main>
<script>
const video=document.getElementById("video");
const statusText=document.getElementById("status");
const records=document.getElementById("records");
const count=document.getElementById("count");
let stream=null, timer=null, busy=false;

async function refresh(){
  try{
    const r=await fetch("/api/attendance");
    const data=await r.json();
    records.innerHTML="";
    data.records.forEach(row=>{
      const tr=document.createElement("tr");
      [row.RollNo,row.Name,row.Time,row.Status].forEach(value=>{
        const td=document.createElement("td");
        td.textContent=value;
        tr.appendChild(td);
      });
      records.appendChild(tr);
    });
    count.textContent=data.records.length+" student(s) marked present today";
  }catch(e){count.textContent="Could not load attendance."}
}

async function recognize(){
  if(!stream || busy || video.readyState<2) return;
  busy=true;
  try{
    const canvas=document.createElement("canvas");
    canvas.width=640; canvas.height=480;
    canvas.getContext("2d").drawImage(video,0,0,640,480);
    const r=await fetch("/api/recognize",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({image:canvas.toDataURL("image/jpeg",0.8)})
    });
    const data=await r.json();
    statusText.textContent=data.message;
    if(data.new_attendance) await refresh();
  }catch(e){statusText.textContent="Recognition request failed."}
  finally{busy=false}
}

document.getElementById("start").onclick=async()=>{
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    video.srcObject=stream;
    statusText.textContent="Camera started. Looking for a registered face...";
    timer=setInterval(recognize,1200);
  }catch(e){
    statusText.textContent="Camera unavailable. Allow camera access and try again.";
  }
};

document.getElementById("stop").onclick=()=>{
  if(timer) clearInterval(timer);
  timer=null;
  if(stream) stream.getTracks().forEach(track=>track.stop());
  stream=null;
  video.srcObject=null;
  statusText.textContent="Camera stopped.";
};

window.addEventListener("beforeunload",()=>{
  if(stream) stream.getTracks().forEach(track=>track.stop());
});
refresh();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/api/attendance")
def attendance():
    return jsonify(records=get_today_records())


@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not known_encodings:
        return jsonify(
            message="No recognizable faces found in student_dataset. "
                    "Register real face photos first.",
            new_attendance=False,
        )

    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image", "")

    try:
        encoded = data_url.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        frame = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise ValueError("Invalid image")

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for encoding in encodings:
            distances = face_recognition.face_distance(
                known_encodings, encoding
            )
            best = int(np.argmin(distances))

            if distances[best] < THRESHOLD:
                roll = known_rolls[best]
                name = known_names[best]
                added = mark_attendance(roll, name)

                return jsonify(
                    message=f"Recognized {name} ({roll}). "
                            + ("Attendance marked." if added
                               else "Already marked today."),
                    new_attendance=added,
                )

        return jsonify(
            message="No registered face recognized.",
            new_attendance=False,
        )

    except (ValueError, IndexError, TypeError):
        return jsonify(
            message="Could not process the camera frame.",
            new_attendance=False,
        )


if __name__ == "__main__":
    load_known_faces()
    app.run(host="127.0.0.1", port=5000, debug=False)
