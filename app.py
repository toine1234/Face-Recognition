from flask import Flask, render_template, request, jsonify, Response
import face_recognition, numpy as np, pickle, os, cv2, base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load dữ liệu đã train
ENCODINGS_FILE = "encodings/faces.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

# Camera
cap = cv2.VideoCapture(0)

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')

# Upload ảnh để nhận diện
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files["file"]
    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)

    return recognize_face(path)

# Nhận diện từ webcam (ảnh base64)
@app.route('/webcam', methods=['POST'])
def webcam():
    img_data = request.json.get("image")
    if not img_data:
        return jsonify({"success": False, "message": "No image data"}), 400
    
    img_bytes = base64.b64decode(img_data.split(",")[1])
    image = Image.open(BytesIO(img_bytes))
    path = "uploads/webcam.jpg"
    image.save(path)
    return recognize_face(path)

# Hàm xử lý nhận diện
def recognize_face(path):
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return jsonify({"success": False, "message": "No face detected"}), 400
    
    encoding = encodings[0]
    matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
        distances = face_recognition.face_distance(data["encodings"], encoding)
        idx = np.argmin(distances)
        name = data["names"][idx]

    return jsonify({"success": True, "name": name})

# Validate accuracy với folder validation/
@app.route("/validate")
def validate():
    correct, total = 0, 0
    for person in os.listdir("validation"):
        img_path = os.path.join("validation", person)
        image = cv2.imread(img_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, boxes)

        for e in encs:
            matches = face_recognition.compare_faces(data["encodings"], e, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                idx = np.argmin(face_recognition.face_distance(data["encodings"], e))
                name = data["names"][idx]

            if name == person:
                correct += 1
            total += 1

    if total == 0:
        return "Không có ảnh để validate."
    acc = 100.0 * correct / total
    return f"Độ chính xác: {correct}/{total} = {acc:.2f}%"

# Hàm tạo video frames cho livestream
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, boxes)

            for (top, right, bottom, left), encoding in zip(boxes, encs):
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    distances = face_recognition.face_distance(data["encodings"], encoding)
                    idx = np.argmin(distances)
                    name = data["names"][idx]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route video stream
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(debug=True)
