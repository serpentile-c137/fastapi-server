from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import shutil
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load YOLO model
model = YOLO("yolov8n_bench.pt")

# Create folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Welcome to YOLO FastAPI Object Detection"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_image_path, occupied_count, empty_count = detect_objects(file_path)

    return JSONResponse(content={
        "filename": file.filename,
        "result_image_url": result_image_path,
        "occupied_count": occupied_count,
        "empty_count": empty_count
    })

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/avi", "video/quicktime"]:
        raise HTTPException(status_code=400, detail="Invalid video file type. Only mp4, avi, and mov are allowed.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_video_path = detect_objects_in_video(file_path)

    return JSONResponse(content={
        "filename": file.filename,
        "result_video_url": result_video_path
    })

@app.get("/live")
def live_video():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model.predict(source=img)
    annotated_image = results[0].plot()
    result_image_path = os.path.join(RESULT_FOLDER, "result_" + os.path.basename(image_path))
    cv2.imwrite(result_image_path, annotated_image)

    occupied_count, empty_count = 0, 0
    if results:
        predictions = results[0].boxes.cls
        for prediction in predictions:
            class_names = {0: "Occupied", 1: "Unoccupied"}
            predicted_class = class_names.get(int(prediction.cpu().numpy()), "Unknown")
            if predicted_class == "Occupied":
                occupied_count += 1
            elif predicted_class == "Unoccupied":
                empty_count += 1

    return result_image_path, occupied_count, empty_count

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    result_video_path = os.path.join(RESULT_FOLDER, "result_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, verbose=False)
        annotated_frame = results[0].plot()

        occupied_count, empty_count = 0, 0
        if results:
            predictions = results[0].boxes.cls
            for prediction in predictions:
                class_names = {0: "Occupied", 1: "Unoccupied"}
                predicted_class = class_names.get(int(prediction.cpu().numpy()), "Unknown")
                if predicted_class == "Occupied":
                    occupied_count += 1
                elif predicted_class == "Unoccupied":
                    empty_count += 1

        cv2.putText(annotated_frame, f"Occupied: {occupied_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Unoccupied: {empty_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return result_video_path

def gen_frames():
    cap = cv2.VideoCapture(0)  # Webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, verbose=False)
        annotated_frame = results[0].plot()

        occupied_count, empty_count = 0, 0
        if results:
            predictions = results[0].boxes.cls
            for prediction in predictions:
                class_names = {0: "Occupied", 1: "Unoccupied"}
                predicted_class = class_names.get(int(prediction.cpu().numpy()), "Unknown")
                if predicted_class == "Occupied":
                    occupied_count += 1
                elif predicted_class == "Unoccupied":
                    empty_count += 1

        cv2.putText(annotated_frame, f"Occupied: {occupied_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Unoccupied: {empty_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
