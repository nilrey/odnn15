import cv2
from ultralytics import YOLO


def main():
    model = YOLO('models/yolov8n.pt')
    

    video_path = 'data/input/input_video.mp4'
    cap = cv2.VideoCapture(video_path)
    output_path = 'data/output/cars-yolov8n-track-001.mp4'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    allowed_indices = {2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    # frame_skip = 1  #Если значение=2, значит Пропускаем кадры (анализируем 1 из 3), если 0 - значит берем каждый кадр
    # frame_count = 0  # Счетчик кадров

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # фильтр - считываем каждый n-ый кадр, frame_skip - указывает сколько фреймов пропускаем.
        # frame_count += 1
        # # Обрабатываем только каждый 3-й кадр
        # if frame_count % frame_skip != 0:
        #     out.write(frame)
        #     continue

        # Используем модель для анализа текущего кадра с отслеживанием
        results = model.track(frame, persist=True, imgsz=640, iou=0.4) # 0.5

        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0]
                if int(box.cls[0]) in allowed_indices and conf > 0.5: # 0.7
                    xyxy = box.xyxy[0]
                    conf = box.conf[0]
                    class_name = results[0].names[int(box.cls[0])]
                    obj_id = int(results[0].boxes.id[i])  # Получаем ID объекта
                    label = f'{class_name} {obj_id} ({conf:.2f})'

                    # Рисуем bounding box и ID на кадре
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Запись обработанного кадра в выходное видео
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработанное видео с трекингом сохранено в {output_path}")


        

if __name__ == "__main__":
    main()