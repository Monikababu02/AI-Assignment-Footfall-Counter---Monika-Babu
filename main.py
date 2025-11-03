import cv2
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSortTracker

os.makedirs("output", exist_ok=True)

model = YOLO('yolov8n.pt') 

tracker = DeepSortTracker(max_age=15)

video_path = "videos/test_video.mp4"   
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/output.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

line_y = 300   
enter_count = 0
exit_count = 0
track_memory = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = []

    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        if int(cls) == 0 and score > 0.5:  
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        if track_id not in track_memory:
            track_memory[track_id] = cy
        else:
            prev_y = track_memory[track_id]
            if prev_y < line_y and cy > line_y:
                enter_count += 1
            elif prev_y > line_y and cy < line_y:
                exit_count += 1
            track_memory[track_id] = cy

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
    cv2.putText(frame, f'IN: {enter_count} | OUT: {exit_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Footfall Counter", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f" Final Count -> Entered: {enter_count}, Exited: {exit_count}")
print(" Output video saved at: output/output.mp4")
