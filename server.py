from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
from ultralytics import YOLO
import cv2
import numpy as np
from fastapi.responses import JSONResponse
 
app = FastAPI()
 
# Load YOLO model
model = YOLO("yolov8n_bench.pt")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
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
 
def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model.predict(source=img)
    annotated_image = results[0].plot()
    result_image_path = os.path.join(UPLOAD_FOLDER, "result_" + os.path.basename(image_path))
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
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)