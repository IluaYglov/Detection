from ultralytics import YOLO
import os


model_name = "yolo11m_dataset2.0_C.N.H_24.02.2026_70ep"
#"yolo11m_coupler_numder_helm_15.02.26" "yolo11n_coupler_number_train" "artifacts\detection"
model_path = os.path.join("artifacts\detection", f"{model_name}.pt")

# Загружаем модель
model = YOLO(model_path, task="detect")  

model.predict("Test_video_2.mp4", save=True, conf=0.5,max_det=10,project="predict", name=f"predict_{model_name}")






