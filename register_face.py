
import os
import cv2

DATASET_DIR = "student_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

roll_no = input("Enter a demo roll number (e.g. DEMO001): ").strip()
name = input("Enter your name for local testing: ").strip()

if not roll_no or not name or "_" in roll_no:
    raise SystemExit("Enter a valid roll number and name.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise SystemExit("Could not open webcam. Close other camera apps and retry.")

print("Camera opened. Press SPACE to capture, or Q to cancel.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Waiting for webcam...")
            time.sleep(0.5)
            continue

        cv2.putText(
            frame,
            "SPACE: s | Q: quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Register Demo Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            safe_name = "_".join(name.split())
            path = os.path.join(
                DATASET_DIR, f"{roll_no}_{safe_name}.jpg"
            )
            if cv2.imwrite(path, frame):
                print(f"Saved local demo photo: {path}")
            else:
                print("Failed to save photo.")
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
