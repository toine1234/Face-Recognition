from flask import Flask, render_template, request, jsonify
import face_recognition, numpy as np, pickle, os

app = Flask(__name__)

ENCODINGS_FILE = "encodings/faces.pkl"
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {"encodings": [], "names": []}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files["file"]
    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)

    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return jsonify({"success": False, "message": "No face detected"}), 400
    
    encoding = encodings[0]
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        idx = np.argmax(matches)
        name = data["names"][idx]

    return jsonify({"success": True, "name": name})

@app.route('/webcam', methods=['POST'])
def webcam():
    img_data = request.json.get("image")
    if not img_data:
        return jsonify({"success": False, "message": "No image data"}), 400
    
    # Chuyển từ base64 -> PIL Image -> lưu tạm
    img_bytes = base64.b64decode(img_data.split(",")[1])
    image = Image.open(BytesIO(img_bytes))
    path = "uploads/webcam.jpg"
    image.save(path)
    return recogize_face(path)

def recogize_face(path):
    img = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(img)

    if not encodings:
        return jsonify({"success": False, "message": "No face detected"}), 400
    
    encoding = encodings[0]
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        idx = np.argmax(matches)
        name = data["names"][idx]

    return jsonify({"success": True, "name": name})

if __name__ == '__main__':
    app.run(debug=True)