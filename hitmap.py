import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Dataset, Logger


def main(model_name : str, input_file :str, output_file : str, confidence : float):
    # Инициализируем ClearML Task
    task = Task.init(
        project_name="odnn15",
        task_name=f"hitmap_car_tracking_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        task_type=Task.TaskTypes.inference
    )
    
    # Устанавливаем параметры задачи
    task.set_parameter("model", model_name)
    task.set_parameter("confidence_threshold", confidence)
    task.set_parameter("iou_threshold", 0.4)
    task.set_parameter("allowed_classes", [0, 2, 3, 5, 6, 7, 8])
    
    dataset = Dataset.create(
        dataset_name="reference_dataset",
        dataset_project="odnn15"
    )    
    model = YOLO(f'models/{model_name}.pt')
    

    video_path = f'data/input/{input_file}'
    cap = cv2.VideoCapture(video_path)
    output_path = f'data/output/{output_file}'

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)

    allowed_indices = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    # Параметры heatmap
    heatmap_alpha = 0.5  # Прозрачность heatmap при наложении
    heatmap_decay = 0.95  # Затухание heatmap со временем
    legend_height = 50  # Высота легенды

    # Высота итогового кадра с учетом легенды
    final_height = frame_height + legend_height
    
    # Инициализируем VideoWriter с правильными размерами
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, final_height))

    # Инициализация heatmap
    heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
    
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
        results = model.track(frame, persist=True, imgsz=frame_width, iou=0.4)

        objects_in_frame = 0
        
        # Обновляем heatmap с затуханием
        heatmap *= heatmap_decay
        
        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0]
                if int(box.cls[0]) in allowed_indices and conf > confidence:
                    objects_in_frame += 1
                    xyxy = box.xyxy[0]
                    conf = box.conf[0]
                    class_name = results[0].names[int(box.cls[0])]
                    obj_id = int(results[0].boxes.id[i])  # Получаем ID объекта
                    label = f'{class_name} {obj_id}'

                    # Рисуем bounding box и ID на кадре
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_yellow, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 2)
                    
                    # Добавляем обнаружение в heatmap
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Создаем гауссово ядро для текущего обнаружения
                    kernel_size = min((x2 - x1) // 2, (y2 - y1) // 2, 50)
                    kernel_size = max(kernel_size, 10)  # Минимальный размер ядра
                    
                    # Добавляем "горячую точку" в heatmap
                    cv2.circle(heatmap, (center_x, center_y), kernel_size, 1.0, -1)

        # Создаем визуализацию heatmap
        if np.max(heatmap) > 0:
            # Нормализуем heatmap только если есть данные
            heatmap_normalized = np.zeros_like(heatmap)
            cv2.normalize(heatmap, heatmap_normalized, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            # Если heatmap пустая, создаем черное изображение
            heatmap_colored = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Накладываем heatmap на оригинальный кадр
        frame_with_heatmap = cv2.addWeighted(frame, 1 - heatmap_alpha, heatmap_colored, heatmap_alpha, 0)
        
        # Добавляем легенду для heatmap
        legend = np.zeros((legend_height, frame_width, 3), dtype=np.uint8)
        
        # Создаем градиент для легенды
        for i in range(frame_width):
            color_value = int(i / frame_width * 255)
            color = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            cv2.line(legend, (i, 0), (i, legend_height), color.tolist(), 1)
        
        # Добавляем текст к легенде
        cv2.putText(legend, "Low", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(legend, "High", (frame_width - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(legend, "Object Density Heatmap", (frame_width // 2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Объединяем кадр с heatmap и легендой
        final_frame = np.vstack([frame_with_heatmap, legend])

        # Сохраняем для гистограммы
        object_counts.append(objects_in_frame)
        
        # Анализ изменений между фреймами
        if frame_count > 0:
            change = abs(objects_in_frame - previous_count)
            frame_changes.append(change)
        
        previous_count = objects_in_frame

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
        out.write(final_frame)
        frame_count += 1
        
        # Периодический вывод в консоль для отладки
        if frame_count % 30 == 0:  # Каждые 30 фреймов
            print(f"Frame {frame_count}: {objects_in_frame} objects detected")

    # Сохраняем финальную heatmap как отдельное изображение
    heatmap_output_path = f'data/output/heatmap_{output_file.split(".")[0]}.png'
    
    # Нормализуем и сохраняем heatmap
    if np.max(heatmap) > 0:
        heatmap_final = np.zeros_like(heatmap)
        cv2.normalize(heatmap, heatmap_final, 0, 255, cv2.NORM_MINMAX)
        heatmap_final_colored = cv2.applyColorMap(heatmap_final.astype(np.uint8), cv2.COLORMAP_JET)
    else:
        # Если heatmap пустая, создаем черное изображение
        heatmap_final_colored = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    cv2.imwrite(heatmap_output_path, heatmap_final_colored)
    
    # Загружаем heatmap как артефакт в ClearML
    task.upload_artifact("heatmap_image", heatmap_output_path)

    # После завершения видео - логируем PLOTS
    if object_counts:
        # Гистограмма распределения объектов
        Logger.current_logger().report_histogram(
            title="Object Detection Analysis",
            series=f"Objects per Frame - {model_name}",
            values=object_counts,
            xaxis="Number of Objects",
            yaxis="Number of Frames"
        )
        
        # Гистограмма стабильности трекинга
        if frame_changes:
            Logger.current_logger().report_histogram(
                title="Tracking Stability Analysis", 
                series=f"Frame-to-Frame Changes - {model_name}",
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
    print(f"Heatmap сохранена в {heatmap_output_path}")
    print(f"Всего обработано фреймов: {frame_count}")
    print(f"Всего обнаружено объектов: {total_objects_detected}")

        

if __name__ == "__main__":
    model_name = "yolov8n"
    confidence = 0.5
    input_name = "cars_1_1"
    output_name = f"hitmap-{input_name}-{model_name}-conf-{confidence}"
    time_start = datetime.now()
    main(model_name, f"{input_name}.mp4", f"{output_name}.mp4", confidence)
    print(f'Время работы: {datetime.now() - time_start} сек.')