import pickle
import face_recognition
import cv2

# Load dữ liệu đã train
with open("encodings/face_encodings.pkl", "rb") as f:
     data = pickle.load(f)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, boxes)

            for (top, right, bottom, left), encoding in zip(boxes, encodings):
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
