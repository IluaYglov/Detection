from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os


model_name = "yolo11n_coupler_number_train"

model_path = os.path.join("artifacts\detection", f"{model_name}.pt")

# Загружаем модель
model = YOLO(model_path, task="detect")  


#region_points = [(1300, 1400),(1300, 800)]                           # line counting
region_points = [(400, 800), (2000, 800),(2000, 1400),  (400, 1400)]  # rectangular region

def count_objects_in_region(video_path, output_video_path, models, region_points):
    
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


    counter = solutions.ObjectCounter(
        show=False,  
        region=region_points,  
        model=models, 
        device=0  
    )   

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = counter(im0)
        #print(results)  
        classwise_count = getattr(results, 'classwise_count', {})
        # Получаем OUT для конкретных классов
        coupler_out = classwise_count.get('coupler', {}).get('OUT', 0) + classwise_count.get('coupler', {}).get('IN', 0)
        video_writer.write(results.plot_im) 

    cap.release()
    video_writer.release()
    #cv2.destroyAllWindows()
    return  coupler_out

result = count_objects_in_region("Test_video_1.mp4", "coupler_counting_video.avi", model, region_points)
print(f"Количество вагонов: {result}")
with open("wagons_count.txt", "w") as f:
    f.write(str(result))
































