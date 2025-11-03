# 👣 Footfall Counter using Computer Vision

## 🎯 Objective
Develop a computer vision-based system that counts how many people **enter** and **exit** through a defined region (doorway or corridor).

---

## 🧰 Tools & Libraries
- Python 3.8+
- YOLOv8 (Ultralytics)
- OpenCV
- DeepSORT (for tracking)
- NumPy

---

## ⚙️ Working Methodology
1. Detects humans in each frame using **YOLOv8**.
2. Tracks people using **DeepSORT** tracker.
3. Defines a **virtual line (ROI)** in the video.
4. Counts when a person crosses the line **(enter or exit)**.
5. Displays live results and saves processed output video.

---

## 🧠 Algorithm Logic
For each tracked person:
- Store previous and current centroid (Y-coordinate).
- If person moves **up → down**, count as **enter**.
- If person moves **down → up**, count as **exit**.

---

## 📂 Folder Structure
