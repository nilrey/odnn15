import cv2
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Dataset, Logger


MODEL_TAG = "yolov8n"

def main():
    # Инициализируем ClearML Task
    task = Task.init(
        project_name="odnn15",
        task_name=f"car_tracking_{MODEL_TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=Task.TaskTypes.inference
    )
    
    # Устанавливаем параметры задачи
    task.set_parameter("model", MODEL_TAG)
    task.set_parameter("confidence_threshold", 0.5)
    task.set_parameter("iou_threshold", 0.4)
    task.set_parameter("allowed_classes", [0, 2, 3, 5, 6, 7, 8])
    
    dataset = Dataset.create(
        dataset_name="reference_dataset",
        dataset_project="odnn15"
    )    
    model = YOLO(f'models/{MODEL_TAG}.pt')
    

    video_path = 'data/input/cars_1.mp4'
    cap = cv2.VideoCapture(video_path)
    output_path = f'data/output/cars-output-{MODEL_TAG}-conf-05-001.mp4'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)


    allowed_indices = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)) 

    frame_count = 0
    total_objects_detected = 0
    object_counts = []  # для гистограммы
    frame_changes = []  # для анализа стабильности
    previous_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # Используем модель для анализа текущего кадра с отслеживанием
        results = model.track(frame, persist=True, imgsz=frame_width, iou=0.4) # 0.5

        objects_in_frame = 0
        
        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0]
                if int(box.cls[0]) in allowed_indices and conf > 0.5: # 0.7
                    objects_in_frame += 1
                    xyxy = box.xyxy[0]
                    conf = box.conf[0]
                    class_name = results[0].names[int(box.cls[0])]
                    obj_id = int(results[0].boxes.id[i])  # Получаем ID объекта
                    # label = f'{class_name} {obj_id} ({conf:.2f})'
                    label = f'{class_name} {obj_id}'

                    # Рисуем bounding box и ID на кадре
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_yellow, 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 1)

        # Сохраняем для гистограммы
        object_counts.append(objects_in_frame)
        
        # Анализ изменений между фреймами
        if frame_count > 0:
            change = abs(objects_in_frame - previous_count)
            frame_changes.append(change)
        
        previous_count = objects_in_frame
        frame_count += 1

        # Логируем количество объектов для текущего фрейма
        Logger.current_logger().report_scalar(
            title="Object Detection Statistics",
            series="Objects per Frame",
            value=objects_in_frame,
            iteration=frame_count
        )
        
        # Логируем накопленную статистику
        total_objects_detected += objects_in_frame
        Logger.current_logger().report_scalar(
            title="Object Detection Statistics", 
            series="Total Objects Detected",
            value=total_objects_detected,
            iteration=frame_count
        )
        
        # Логируем среднее количество объектов на фрейм
        if frame_count > 0:
            avg_objects = total_objects_detected / (frame_count + 1)
            Logger.current_logger().report_scalar(
                title="Object Detection Statistics",
                series="Average Objects per Frame",
                value=avg_objects,
                iteration=frame_count
            )

        # Запись обработанного кадра в выходное видео
        out.write(frame)
        frame_count += 1
        
        # Периодический вывод в консоль для отладки
        if frame_count % 30 == 0:  # Каждые 30 фреймов
            print(f"Frame {frame_count}: {objects_in_frame} objects detected")

    # После завершения видео - логируем PLOTS
    if object_counts:
        # Гистограмма распределения объектов
        Logger.current_logger().report_histogram(
            title="Object Detection Analysis",
            series=f"Objects per Frame - {MODEL_TAG}",
            values=object_counts,
            xaxis="Number of Objects",
            yaxis="Number of Frames"
        )
        
        # Гистограмма стабильности трекинга
        if frame_changes:
            Logger.current_logger().report_histogram(
                title="Tracking Stability Analysis", 
                series=f"Frame-to-Frame Changes - {MODEL_TAG}",
                values=frame_changes,
                xaxis="Objects Change Count", 
                yaxis="Frequency"
            )

    # Сохраняем итоговую статистику
    task.get_logger().report_single_value("Total Frames Processed", frame_count)
    task.get_logger().report_single_value("Total Objects Detected", total_objects_detected)
    task.get_logger().report_single_value("Average Objects per Frame", total_objects_detected / max(frame_count, 1))
    
    # Загружаем обработанное видео как артефакт
    task.upload_artifact("processed_video", output_path)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Обработанное видео с трекингом сохранено в {output_path}")
    print(f"Всего обработано фреймов: {frame_count}")
    print(f"Всего обнаружено объектов: {total_objects_detected}")

        

if __name__ == "__main__":
    time_start = datetime.now()
    main()
    print(f'Время работы: {datetime.now() - time_start} сек.')