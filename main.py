import cv2
import numpy as np
from tkinter import filedialog
import tkinter as tk
import os

# Плотность ASCII-символов от темного к светлому
ASCII_CHARS = [" ", ".", ",", "-", "~", ":", ";", "=", "!", "*", "#", "$", "@"]
CHAR_RANGE = 255 // (len(ASCII_CHARS) - 1)

def choose_video_file():
    """Окно выбора исходного видеофайла."""
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Выберите видеофайл для обработки",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

def process_frame(frame, bbox, blur_intensity, target_width=75):
    """
    Размывает фон кадра на основе выбранной интенсивности blur_intensity, 
    а внутрь bbox накладывает цветной ASCII-эффект.
    """
    x, y, w, h = [int(v) for v in bbox]
    
    # Расчет нечетного размера ядра для Гауссова размытия
    # Если на ползунке 0, размытие отключается
    if blur_intensity > 0:
        kernel_size = blur_intensity * 2 + 1
        blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    else:
        blurred_frame = frame.copy()

    if w <= 0 or h <= 0:
        return blurred_frame

    # Вырезаем область объекта (ROI)
    roi = frame[y:y+h, x:x+w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    roi_h, roi_w = roi_gray.shape
    ascii_w = target_width
    ascii_h = int((ascii_w / roi_w) * roi_h * 0.55)
    
    if ascii_h <= 0 or ascii_w <= 0:
        return blurred_frame

    small_gray = cv2.resize(roi_gray, (ascii_w, ascii_h))
    small_color = cv2.resize(roi, (ascii_w, ascii_h))

    ascii_roi_canvas = np.zeros_like(roi)
    cell_w = w / ascii_w
    cell_h = h / ascii_h

    for i in range(ascii_h):
        for j in range(ascii_w):
            intensity = small_gray[i, j]
            char = ASCII_CHARS[intensity // CHAR_RANGE]
            
            pixel_color = small_color[i, j]
            color = (int(pixel_color[0]), int(pixel_color[1]), int(pixel_color[2]))

            text_x = int(j * cell_w)
            text_y = int(i * cell_h) + int(cell_h)

            cv2.putText(ascii_roi_canvas, char, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1, cv2.LINE_AA)

    ascii_mask = cv2.cvtColor(ascii_roi_canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(ascii_mask, 10, 255, cv2.THRESH_BINARY_INV)

    roi_cleaned = cv2.bitwise_and(roi, roi, mask=mask_inv)
    combined_roi = cv2.add(roi_cleaned, ascii_roi_canvas)

    output_frame = blurred_frame.copy()
    output_frame[y:y+h, x:x+w] = combined_roi
    
    return output_frame

def nothing(x):
    """Пустая функция-заглушка для работы ползунка трекбара."""
    pass

def main():
    print("Ожидание выбора файла...")
    video_path = choose_video_file()
    if not video_path:
        print("Файл не выбран. Выход.")
        return

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if not success:
        print("Не удалось открыть или прочитать видео файл.")
        return

    # Настройки выходного файла записи
    file_dir, file_name = os.path.split(video_path)
    output_path = os.path.join(file_dir, "ascii_processed_" + os.path.splitext(file_name)[0] + ".mp4")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    window_name = "ASCII Blur & Record System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Создаем ползунок для управления размытием
    # Параметры: Название, Имя окна, Начальное значение, Максимальное значение, Функция-обработчик
    cv2.createTrackbar("Blur Strength", window_name, 12, 50, nothing)
    
    print("\n=== ИНСТРУКЦИЯ ===")
    print("1. В окне выделите МЫШКОЙ нужный объект.")
    print("2. Нажмите ENTER или ПРОБЕЛ для подтверждения.")
    print("3. Регулируйте ползунок 'Blur Strength' вверху окна для настройки размытия.")
    print("4. Нажмите 'q' для смены объекта трекинга.")
    print("5. Нажмите ESC для завершения работы и сохранения.")
    print(f"Файл записи будет сохранен здесь: {output_path}\n")

    bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Обработка завершена. Видео успешно сохранено.")
            break

        # Считываем текущее значение ползунка размытия
        blur_val = cv2.getTrackbarPos("Blur Strength", window_name)

        track_success, bbox = tracker.update(frame)

        if track_success:
            # Передаем значение ползунка в функцию обработки
            processed_frame = process_frame(frame, bbox, blur_val, target_width=75)
            
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        else:
            if blur_val > 0:
                kernel_size = blur_val * 2 + 1
                processed_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            else:
                processed_frame = frame.copy()
                
            cv2.putText(processed_frame, "Target Lost! Press 'q' to reselect.", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        out_writer.write(processed_frame)
        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Пауза для смены объекта...")
            bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            
        elif key == 27:
            print("Принудительное завершение и закрытие файла записи.")
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
