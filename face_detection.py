import cv2
import pickle
import face_recognition
import numpy as np

# Load dữ liệu đã train
with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

TOLERANCE = data.get("tolerance", 0.4)

def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize để tăng tốc
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_frame, model="cnn")
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        for (top, right, bottom, left), encoding in zip(boxes, encodings):
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=TOLERANCE)
            name = "Unknown"

            if True in matches:
                distances = face_recognition.face_distance(data["encodings"], encoding)
                idx = np.argmin(distances)
                name = data["names"][idx]

            # Scale lại khung về size gốc
            top, right, bottom, left = [v*2 for v in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
