from fastapi import FastAPI, UploadFile, File
from face_recognition_module import recognize_face

app = FastAPI(title="FaceID Attendance API")

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    Nhận file ảnh từ client và trả về tên sinh viên nếu nhận diện được.
    """
    result = recognize_face(file.file)
    return result
