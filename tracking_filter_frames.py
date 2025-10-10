import cv2
from ultralytics import YOLO
from datetime import datetime

MODEL_TAG = "yolov8s"

def main():
    model = YOLO(f'models/{MODEL_TAG}.pt')
    
    video_path = 'data/input/cars_1.mp4'
    cap = cv2.VideoCapture(video_path)
    output_path = f'data/output/cars-output-{MODEL_TAG}-frame-1-3-002.mp4'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)

    allowed_indices = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    frame_skip = 2  # Если значение=2, значит Пропускаем кадры (анализируем 1 из 3), если 0 - значит берем каждый кадр
    frame_count = 0  # Счетчик кадров
    
    # Храним информацию о последних bounding boxes
    last_boxes_info = []  # Список кортежей: (x1, y1, x2, y2, label, class_name, obj_id)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_frame = frame.copy()
        
        # Обрабатываем только каждый 3-й кадр (когда frame_count % (frame_skip + 1) == 0)
        if frame_count % (frame_skip + 1) == 0:
            # Очищаем предыдущие bounding boxes
            last_boxes_info = []
            
            # Используем модель для анализа текущего кадра с отслеживанием
            results = model.track(current_frame, persist=True, imgsz=frame_width, iou=0.4)

            if results[0].boxes.id is not None:
                for i, box in enumerate(results[0].boxes):
                    conf = box.conf[0]
                    if int(box.cls[0]) in allowed_indices and conf > 0.5:
                        xyxy = box.xyxy[0]
                        class_name = results[0].names[int(box.cls[0])]
                        obj_id = int(results[0].boxes.id[i])
                        label = f'{class_name} {obj_id}'

                        # Сохраняем информацию о bounding box
                        x1, y1, x2, y2 = map(int, xyxy)
                        last_boxes_info.append((x1, y1, x2, y2, label, class_name, obj_id))
            
            # Рисуем bounding boxes на текущем кадре
            for box_info in last_boxes_info:
                x1, y1, x2, y2, label, class_name, obj_id = box_info
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), color_yellow, 1)
                cv2.putText(current_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 1)
            
        else:
            # Для пропущенных кадров используем последние известные bounding boxes
            for box_info in last_boxes_info:
                x1, y1, x2, y2, label, class_name, obj_id = box_info
                cv2.rectangle(current_frame, (x1, y1), (x2, y2), color_yellow, 1)
                cv2.putText(current_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 1)

        # Записываем кадр с bounding boxes (текущий кадр + последние известные разметки)
        out.write(current_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработанное видео с трекингом сохранено в {output_path}")

if __name__ == "__main__":
    time_start = datetime.now()
    main()
    print(f'Время работы: {datetime.now() - time_start} сек.')