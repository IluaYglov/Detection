#n s m l x

from ultralytics import YOLO
import os
# import multiprocessing  
# import torch




model_name = "yolo11m"
batch_size=16
data_path = "dataset_coupler_ult_detect1.yaml"

model_path = os.path.join("models", f"{model_name}.pt")
#model_path = os.path.join("artifacts/detection", f"{model_name}.pt")

if __name__ == '__main__':

    #multiprocessing.freeze_support()  \\ выше точность

    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    
    # Загружаем модель
    model = YOLO(model_path, task="detect")
    
    print("НАЧАЛО ОБУЧЕНИЯ")
    
    # Обучение
    results = model.train(
        data=data_path,
        device=0, 
        epochs=70, 
        batch=batch_size, 
        #workers=0,  
        show_boxes=False, 
        exist_ok = True,
        name=f"{model_name}_dataset2.0_C.N.H_24.02.2026_70ep"
    )
    
    output_dir = "artifacts/detection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем модель
    model.save(os.path.join(output_dir, f"{model_name}_dataset2.0_C.N.H_24.02.2026_70ep.pt"))
    print("МОДЕЛЬ СОХРАНЕНА")









    
    # print("-----Валидация-----")
    

    # metric = model.val(batch=32, name=f"{model_name}_coupler_numder_helm_20.02.26_val")
    # print("ВАЛИДАЦИЯ ЗАВЕРШЕНА")

















# print("-----Предсказания-----" )

# start_time = end_time

# results = model.predict(source=os.path.expanduser("F:\1_Practika\Detection\dataset_coupler_ult_detect1/images/test"),
#                             batch=batch_size,
#                             conf=0.25,
#                             show_labels=True,
#                             show_boxes=True,
#                             show_conf=True,
#                             save=True,
#                             verbose=False,
#                             name=f"{model_name}_test")

# end_time = time.time()
# predict_duration = end_time - start_time



#print(f"Обучение (сек): {train_duration:.2f}")
#print(f"Валидация (сек): {validation_duration:.2f}")
#print(f"Предсказание (сек): {predict_duration:.2f}")