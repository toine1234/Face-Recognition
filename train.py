import face_recognition
import os
import pickle

DATASET_DIR = "training"
ENCODINGS_FILE = "encodings/faces.pkl"

def train():
    know_encodings = []
    know_names = []

    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                know_encodings.append(encodings[0])
                know_names.append(person)
            
    data = {"encodings": know_encodings, "names": know_names}
    os.makedirs("encodings", exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    
    print(f"[INFO] Training completed. Encodings saved to {ENCODINGS_FILE}")

if __name__ == "__main__":
    train()