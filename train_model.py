import os
import cv2
import numpy as np
import face_recognition
import pickle

dataset_dir = 'dataset'
known_encodings = []
known_names = []

for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}
with open('encodings/face_encodings.pkl', 'wb') as f:
    pickle.dump(data, f)

print("✅ Training complete, encodings saved to face_encodings.pkl")