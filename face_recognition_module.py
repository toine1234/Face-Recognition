import face_recognition
import os, pickle
import numpy as np
import pickle

ENCODINGS_FILE = "encodings.pkl"

# Tạo embeddings 128D
def encode_faces(data_dir="validation"):
    """
    Encode tất cả ảnh sinh viên trong thư mục và lưu vào file pickle.
    """
    known_encodings = []
    known_names = []

    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(data_dir, filename)
            name = os.path.splitext(filename)[0]

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
            else:
                print(f"Không tìm thấy khuôn mặt trong ảnh {filename}")

    # Lưu file pickle
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
    print("Đã lưu embeddings của sinh viên vào file.")

def recognize_face(image_file):
    """
    Nhận diện khuôn mặt từ file ảnh upload.
    Trả về tên sinh viên và distance.
    """
    # Load embeddings
    if not os.path.exists(ENCODINGS_FILE):
        return {"error": "Chưa có embeddings. Chạy encode_faces trước."}

    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)

    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return {"error": "Không tìm thấy khuôn mặt trong ảnh."}
    
    face_encoding = encodings[0]

    # So sánh database
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(distances)

    # If distance nhỏ hơn 0.6 thì coi là đúng
    if distances[best_match_index] < 0.6:
        return {"name": known_names[best_match_index], "distance": float(distances[best_match_index])}
    else:
        return {"name": None, "distance": float(distances[best_match_index])}
    
if __name__ == "__main__":
    # Chạy encode_faces một lần để tạo embeddings
    encode_faces()