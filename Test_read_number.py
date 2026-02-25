from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os
import easyocr
import re
import pytesseract 


model_name = "yolo11n_coupler_number_train"

model_path = os.path.join("artifacts\detection", f"{model_name}.pt")

# Загружаем модель
models = YOLO(model_path, task="detect")  

# Открываем видеофайл
cap = cv2.VideoCapture("Test_video_1.mp4")
assert cap.isOpened(), "Ошибка чтения видеофайла"

# Получаем параметры видео и создаем video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("check_text_number_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Инициализируем OCR EasyOCR
# reader = easyocr.Reader(['en', 'ru'])  # Английский и русский языки для номеров

padding = 10  # Отступ для обрезки области номера

correct_number = [] #список для хранения найденных номеров

def check_number(number_str):
    # Оставляем только цифры
    digits_only = re.sub(r'\D', '', number_str)
    
    if len(digits_only) != 8:
        return False
    
    wagon_digits = [int(d) for d in digits_only[:7]]
    control_digit = int(digits_only[7])
    
    # Поразрядное произведение
    total = 0
    for i, digit in enumerate(wagon_digits):
        if i % 2 == 0:  
            product = digit * 2
        else:  
            product = digit * 1
        
        # Поразрядная сумма
        total += product // 10 + product % 10
    
    total = (10-(total % 10))%10
    
    if total == control_digit:
        return digits_only
    else:
        return False

while cap.isOpened():
    success, im0 = cap.read()
    
    if not success:
        break
    
    # Детекция объектов моделью YOLO
    results = models.predict(im0, verbose=False)[0].boxes
    boxes = results.xyxy.cpu()
    clss = results.cls.cpu()
    
    ann = Annotator(im0, line_width=3)

    for cls, box in zip(clss, boxes):
        height, width = im0.shape[:2]  # Размеры изображения
        
        # Вычисляем координаты с отступом
        x1 = max(int(box[0]) - padding, 0)
        y1 = max(int(box[1]) - padding, 0)
        x2 = min(int(box[2]) + padding, width)
        y2 = min(int(box[3]) + padding, height)
        
        # Вырезаем область номера
        plate_roi = im0[y1:y2, x1:x2]
        
        # Улучшенная предобработка для Tesseract [web:11][web:12]
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Увеличение контрастности
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Бинаризация (критично для Tesseract)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Морфологическое закрытие — залатать дырки в цифрах
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Увеличение размера для лучшего распознавания
        scale_factor = 3
        enhanced_resized = cv2.resize(closed, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        #Сохраниение изображений, с которых считывался текст
        #save_path = f"artifacts/detection_{cap.get(cv2.CAP_PROP_POS_FRAMES)}_{cls.item()}.png"  # каталог plates должен существовать
        #cv2.imwrite(save_path, enhanced_resized)

        # Распознаем текст с помощью Tesseract [web:11]
        # Конфигурация для номеров: одиночная строка текста, только цифры
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        extracted_text = pytesseract.image_to_string(enhanced_resized, config=custom_config).strip()
        #print(f"Распознанный номер: {extracted_text}")
        # Проверяем номер
        valid_number = check_number(extracted_text)
        print(f"Распознанный номер: {extracted_text}")
        if valid_number:
            print(f"Распознанный корректный номер: {valid_number}")
            correct_number.append(valid_number)
        
        # Рисуем bounding box с номером
        label = extracted_text if extracted_text else "No text"
        ann.box_label(box, label=label, color=colors(cls, True))
    
    video_writer.write(im0)  # Записываем обработанный кадр

print("Найденные корректные номера:", set(correct_number))
cap.release()
video_writer.release()
cv2.destroyAllWindows()





#Пример с easyocr
# while cap.isOpened():
#     success, im0 = cap.read()
    
#     if not success:
#         break
    
#     # Детекция объектов моделью YOLO
#     results = models.predict(im0, verbose=False)[0].boxes
#     boxes = results.xyxy.cpu()
#     clss = results.cls.cpu()
    
#     ann = Annotator(im0, line_width=3)
    


#     for cls, box in zip(clss, boxes):
#         height, width = im0.shape[:2]  # Размеры изображения
        
#         # Вычисляем координаты с отступом
#         x1 = max(int(box[0]) - padding, 0)
#         y1 = max(int(box[1]) - padding, 0)
#         x2 = min(int(box[2]) + padding, width)
#         y2 = min(int(box[3]) + padding, height)
        
#         # Вырезаем область номера
#         plate_roi = im0[y1:y2, x1:x2]
        
#         # Предобработка изображения для OCR
#         gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
#         # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         # enhanced = clahe.apply(gray)
#         scale_factor = 3
#         enhanced_resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
#         # Распознаем текст с помощью EasyOCR
#         ocr_results = reader.readtext(enhanced_resized)
        
#         # Извлекаем распознанный текст
#         extracted_text = ""
#         if ocr_results:
#             texts = [result[1] for result in ocr_results]
#             extracted_text = ' '.join(texts).strip()
            
            
#         if check_number(extracted_text) != False:
#             print(f"Распознанный текст: {check_number(extracted_text)}")
#             correct_number.append(check_number(extracted_text))
            
        
#         # Рисуем bounding box с номером
#         ann.box_label(box, label=extracted_text, color=colors(cls, True))
         
        
    
#     video_writer.write(im0)  # Записываем обработанный кадр


# print(set(correct_number))
# cap.release()
# video_writer.release()