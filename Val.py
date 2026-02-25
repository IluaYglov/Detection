
from ultralytics import YOLO
import os
import multiprocessing  


model_name ="yolo11m_dataset2.0_C.N.H_24.02.2026_70ep" 

#"yolo11n_coupler_number_train"
model_path = os.path.join("artifacts/detection", f"{model_name}.pt")

if __name__ == '__main__':

    model = YOLO(model_path, task="detect")
    
    print("-----Валидация-----")
    
    metrics = model.val(batch=16, name=f"{model_name}_TEST")
    # Метрика Полнота(Recall): Все ли обЪекты найдены?
    print(f"Mean Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
    # Метрика точность (Precision): Все ли найденные обЪекты правильные?
    print(f"Mean Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")