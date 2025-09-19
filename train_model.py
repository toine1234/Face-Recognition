import os, cv2, pickle, face_recognition

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings/face_encodings.pkl"

def train(model="hog"):
    known_encodings, known_names = [], []
    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir): 
            continue

        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[!] Không đọc được ảnh {image_path}")
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb, model=model)
            if not boxes:
                print(f"[!] Không tìm thấy mặt trong {image_path}, bỏ qua")
                continue

            encs = face_recognition.face_encodings(rgb, boxes)
            for e in encs:
                known_encodings.append(e)
                known_names.append(person)

    data = {"encodings": known_encodings, "names": known_names, "tolerance": 0.5}
    os.makedirs("encodings", exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Training complete, {len(known_encodings)} encodings saved.")

if __name__ == "__main__":
    train(model="hog")
