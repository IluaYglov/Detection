from PIL import Image
import pytesseract
import cv2
import os
import easyocr
import re
import pytesseract 
import numpy as np
# Укажите путь к исполняемому файлу Tesseract, если он не в PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Загружаем изображение

# Загружаем изображение через PIL
pil_img = Image.open('011.jpg').convert('RGB')

# Переводим PIL -> numpy -> OpenCV (BGR)
img = np.array(pil_img)
img = img[:, :, ::-1].copy()   # RGB -> BGR

# Дальше работаем в OpenCV
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
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


#extracted_text = pytesseract.image_to_string(enhanced_resized, lang='rus+eng').strip()

extracted_text = pytesseract.image_to_string(pil_img, lang='rus+eng').strip()



#Сохраниение изображений, с которых считывался текст
frame_idx = 3  # или счётчик/время кадра
save_path = os.path.join("artifacts", f"plate_{frame_idx}.png")
cv2.imwrite(save_path, enhanced_resized)
# Распознаем текст с изображения
# text = pytesseract.image_to_string(image, lang='rus+eng')

# Выводим распознанный текст
print("Текст  ")
print(extracted_text)