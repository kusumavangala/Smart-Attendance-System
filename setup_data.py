import os
import cv2
import numpy as np
import pandas as pd

DATASET_DIR = "student_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

student_names = [
    "Aarav Sharma", "Aditi Rao", "Amit Patel", "Ananya Iyer", "Aryan Verma",
    "Deepak Nair", "Diya Sen", "Ishaan Gupta", "Kavya Reddy", "Manish Joshi",
    "Meera Pillai", "Naveen Kumar", "Neha Choudhury", "Nikhil Bhat", "Pooja Hegde",
    "Pranav Mehta", "Priya Das", "Rahul Saxena", "Rhea Kapoor", "Rohan Mishra",
    "Sakshi Bansal", "Sameer Deshmukh", "Sanya Malhotra", "Siddharth Kaul", "Sneha Roy",
    "Tanvi Kulkarni", "Varun Dhawan", "Vidya Balan", "Yash Chopra", "Zoya Akhtar"
]

def generate_students():
    records = []
    print("Generating 30 student profiles...")

    for i, name in enumerate(student_names, start=1):
        roll_no = f"STU{i:03d}"
        filename = f"{roll_no}_{name.replace(' ', '_')}.jpg"
        filepath = os.path.join(DATASET_DIR, filename)

        # Generate a clean 8-bit synthetic profile image
        img = np.full((300, 300, 3), (230, 220, 210), dtype=np.uint8)
        # Draw avatar placeholder
        cv2.circle(img, (150, 110), 45, (80, 80, 80), -1)
        cv2.ellipse(img, (150, 230), (70, 50), 0, 0, 180, (80, 80, 80), -1)
        # Label with Roll No & Name
        cv2.putText(img, roll_no, (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(filepath, img)

        records.append({"RollNo": roll_no, "Name": name, "ImageFile": filename})

    pd.DataFrame(records).to_csv("students.csv", index=False)
    print("✓ 30 student dataset initialized in 'student_dataset/' and 'students.csv'.")

def capture_my_face():
    print("\n[Optional] Capture YOUR face to register as Student 001:")
    choice = input("Do you want to snap your photo now? (y/n): ").strip().lower()
    if choice == 'y':
        cam = cv2.VideoCapture(0)
        print("Camera started. Look at the lens and press 'SPACE' to capture (or 'q' to skip)...")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to access camera.")
                break
            cv2.putText(frame, "Press SPACE to Save Face | 'q' to Quit", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Register Face", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 32:  # SPACE
                target_file = os.path.join(DATASET_DIR, "STU001_Aarav_Sharma.jpg")
                cv2.imwrite(target_file, frame)
                print("✓ Successfully saved your photo as STU001_Aarav_Sharma.jpg!")
                break
            elif k == ord('q'):
                break
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_students()
    capture_my_face()