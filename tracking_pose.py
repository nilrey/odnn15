import cv2
from ultralytics import YOLO
from datetime import datetime

MODEL_TAG = "yolov8x-pose"

def main():
    model = YOLO(f'models/{MODEL_TAG}.pt')
    # НАСТРОЙКИ ОТОБРАЖЕНИЯ
    SHOW_BOUNDING_BOXES = False  # Отображать bounding boxes
    SHOW_POSE = True           # Отображать ключевые точки и скелет
    STROKE_WIDTH = 2            # Ширина отрисованных линий
    
    # Функция для генерации случайного цвета для каждого объекта
    def get_random_color(obj_id):
        import random
        random.seed(obj_id)  # Для постоянства цвета для одного и того же объекта
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    video_path = 'data/input/cars_3.mp4'
    cap = cv2.VideoCapture(video_path)
    output_path = f'data/output/cars-pose-output-{MODEL_TAG}-conf-05-003.mp4'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)

    allowed_indices = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Используем модель для анализа текущего кадра с отслеживанием
        results = model.track(frame, persist=True, imgsz=frame_width, iou=0.4)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0]
                if int(box.cls[0]) in allowed_indices and conf > 0.5:
                    xyxy = box.xyxy[0]
                    conf = box.conf[0]
                    class_name = results[0].names[int(box.cls[0])]
                    obj_id = int(results[0].boxes.id[i])  # Получаем ID объекта
                    label = f'{class_name} {obj_id}'

                    obj_color = get_random_color(obj_id)

                    # Рисуем bounding box и ID на кадре
                    if SHOW_BOUNDING_BOXES:
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, 1)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1)

                    # ДОБАВЛЕНО: Отрисовка ключевых точек позы
                    if SHOW_POSE and results[0].keypoints is not None and i < len(results[0].keypoints.data):
                        keypoints = results[0].keypoints.data[i]

                        # Рисуем прямоугольник головы
                        head_points = [0, 1, 2, 3, 4]  # нос, глаза, уши
                        visible_head_points = [kp for idx, kp in enumerate(keypoints) 
                                            if idx in head_points and kp[2] > 0.5]
                        
                        if len(visible_head_points) >= 2:
                            x_coords = [int(kp[0]) for kp in visible_head_points]
                            y_coords = [int(kp[1]) for kp in visible_head_points]
                            
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            padding = 10
                            x_min = max(0, x_min - padding)
                            y_min = max(0, y_min - padding)
                            x_max = min(frame_width, x_max + padding)
                            y_max = min(frame_height, y_max + padding)
                            
                            
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), obj_color, STROKE_WIDTH)
                        
                            # Соединение головы с туловищем
                            if (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5 and  # оба плеча видны
                                len(visible_head_points) >= 2):  # голова определена
                                
                                # Центр нижней части головы
                                head_bottom_x = (x_min + x_max) // 2
                                head_bottom_y = y_max
                                
                                # Центр линии плеч
                                shoulder_center_x = int((keypoints[5][0] + keypoints[6][0]) / 2)
                                shoulder_center_y = int((keypoints[5][1] + keypoints[6][1]) / 2)
                                
                                # Рисуем линию от низа головы к плечам
                                cv2.line(frame, (head_bottom_x, head_bottom_y), 
                                        (shoulder_center_x, shoulder_center_y), obj_color, STROKE_WIDTH)

                        # Рисуем ключевые точки
                        # for kp in keypoints:
                        #     if kp[2] > 0.5:  # Проверяем confidence ключевой точки
                        #         x, y = int(kp[0]), int(kp[1])
                        #         cv2.circle(frame, (x, y), 3, obj_color, -1)
                        
                        # Рисуем скелет (соединяем ключевые точки линиями)
                        # Пример соединений для COCO pose (17 ключевых точек)
                        skeleton = [
                            #(0, 1), (0, 2), (1, 3), (2, 4),  # Голова-плечи-локти
                            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Торс-бедра-колени-лодыжки
                            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Плечи-запястья
                            (5, 11), (6, 12)  # Соединение торса с плечами
                        ]
                        
                        for connection in skeleton:
                            start_idx, end_idx = connection
                            if (start_idx < len(keypoints) and end_idx < len(keypoints) and 
                                keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5):
                                x1_kp, y1_kp = int(keypoints[start_idx][0]), int(keypoints[start_idx][1])
                                x2_kp, y2_kp = int(keypoints[end_idx][0]), int(keypoints[end_idx][1])
                                cv2.line(frame, (x1_kp, y1_kp), (x2_kp, y2_kp), obj_color, STROKE_WIDTH)

        # Запись обработанного кадра в выходное видео
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработанное видео с трекингом и позами сохранено в {output_path}")

if __name__ == "__main__":
    time_start = datetime.now()
    main()
    print(f'Время работы: {datetime.now() - time_start} сек.')