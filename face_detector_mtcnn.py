import cv2
import mtcnn
import os

# lấy thư mục gốc
file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
# print(file_path)

face_detector = mtcnn.MTCNN()
img = cv2.imread(file_path + 'data'+os.sep+'1.jpg')

conf_t =0.99
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detector.detect_faces(img_rgb)
# print(result)

for res in results:
    x1,y1,width,height = res['box']
    x1,y1 = abs(x1), abs(y1)
    x2,y2 = x1 + width, y1 + height
    confidence = res['confidence']
    if confidence < conf_t:
        continue
    key_points = res['keypoints'].valueas()

    cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0,0), thickness=2)
    cv2.putText(img, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,255), 1)

    for point in key_points:
        cv2.circle(img, key_points[point], 3, (0,255,0), 1)

cv2.imshow('image', img)
cv2.waitKey(0)