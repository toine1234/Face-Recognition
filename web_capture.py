import cv2
from face_recognition_module import recognize_face

def capture_and_recognize():
    """
    Mở webcam, chụp ảnh và nhận diện khuôn mặt.
    Nhấn 'q' để thoát.
    """
    cap = cv2.VideoCapture(0)  # 0: webcam mặc định

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể mở webcam.")
            break

        # Hiển thị video
        cv2.imshow("Webcam", frame)

        # Nhấn 'c' để capture và nhận diện
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            # Lưu ảnh tạm
            cv2.imwrite("temp.jpg", frame)
            result = recognize_face("temp.jpg")
            print("Kết quả nhận diện:", result)

        # Nhấn 'q' để thoát
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_recognize()