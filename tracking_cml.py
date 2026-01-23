import cv2
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Dataset, Logger


def main(model_name: str, input_file: str, output_file: str, confidence: float, date_stamp: datetime):
    # Инициализируем ClearML Task
    task = Task.init(
        project_name="ICIE Detection Project",
        task_name=f"tracking_{model_name}_{date_stamp}",
        task_type=Task.TaskTypes.inference
    )
    
    model = YOLO(f'models/{model_name}.pt')
    video_path = f'data/input/corrupted/{input_file}'
    output_path = f'data/output/{output_file}'

    # Устанавливаем параметры задачи
    task.set_parameter("model", model_name)
    task.set_parameter("confidence_threshold", confidence)
    task.set_parameter("iou_threshold", 0.4)
    task.set_parameter("allowed_classes", [0, 2, 3, 5, 6, 7, 8])
    task.set_parameter("input_video", video_path)
    task.set_parameter("output_video", output_path)

    # cloned_task.set_user_properties(model="yolov8s")
    
    cap = cv2.VideoCapture(video_path)

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
    
    # Для отслеживания уникальных объектов
    unique_object_ids = set()
    unique_objects_cumulative = []  # Для хранения нарастающего итога уникальных объектов
    
    # Массив для подсчета объектов по диапазонам confidence
    # Индексы: 0: 0.5-0.6, 1: 0.6-0.7, 2: 0.7-0.8, 3: 0.8-0.9, 4: 0.9-1.0
    confidence_distribution = [0, 0, 0, 0, 0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        # Используем модель для анализа текущего кадра с отслеживанием
        results = model.track(frame, persist=True, imgsz=frame_width, iou=0.4)

        objects_in_frame = 0
        
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

                    # Добавляем ID в множество уникальных объектов
                    unique_object_ids.add(obj_id)

                    # Подсчет объекта по диапазону confidence
                    if 0.5 <= conf < 0.6:
                        confidence_distribution[0] += 1
                    elif 0.6 <= conf < 0.7:
                        confidence_distribution[1] += 1
                    elif 0.7 <= conf < 0.8:
                        confidence_distribution[2] += 1
                    elif 0.8 <= conf < 0.9:
                        confidence_distribution[3] += 1
                    elif 0.9 <= conf <= 1.0:
                        confidence_distribution[4] += 1

                    # Рисуем bounding box и ID на кадре
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_yellow, 1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 1)

        # Сохраняем для гистограммы
        object_counts.append(objects_in_frame)
        
        # Сохраняем нарастающий итог уникальных объектов
        unique_objects_cumulative.append(len(unique_object_ids))
        
        # Анализ изменений между фреймами
        if frame_count > 0:
            change = abs(objects_in_frame - previous_count)
            frame_changes.append(change)
        
        previous_count = objects_in_frame

        # Логируем количество объектов для текущего фрейма
        Logger.current_logger().report_scalar(
            title="Статистика обнаружения объектов",
            series="Объектов на фрейме",
            value=objects_in_frame,
            iteration=frame_count
        )
        
        # Логируем количество уникальных объектов на текущий момент
        Logger.current_logger().report_scalar(
            title="Трекируемые объекты",
            series="Трекируемые объекты (суммарно)",
            value=len(unique_object_ids),
            iteration=frame_count
        )
        
        # Логируем накопленную статистику
        total_objects_detected += objects_in_frame
        Logger.current_logger().report_scalar(
            title="Статистика обнаружения объектов", 
            series="Общее количество обнаруженных объектов",
            value=total_objects_detected,
            iteration=frame_count
        )
        
        # Логируем среднее количество объектов на фрейм
        if frame_count > 0:
            avg_objects = total_objects_detected / (frame_count + 1)
            Logger.current_logger().report_scalar(
                title="Статистика обнаружения объектов",
                series="Среднее количество объектов в кадре",
                value=avg_objects,
                iteration=frame_count
            )

        # Запись обработанного кадра в выходное видео
        out.write(frame)
        frame_count += 1
        
        # Периодический вывод в консоль для отладки
        if frame_count % 30 == 0:  # Каждые 30 фреймов
            print(f"Фрейм {frame_count}: найдено = {objects_in_frame}, всего уникальных = {len(unique_object_ids)}")

    # После завершения видео - логируем PLOTS
    if object_counts:
        # Гистограмма распределения объектов на кадре
        Logger.current_logger().report_histogram(
            title="Распознанные объекты",
            series=f"Количество объектов на фрейме - {model_name}",
            values=object_counts,
            xaxis="Количество объектов",
            yaxis="Номер фрейма"
        )
        
        # Гистограмма уникальных объектов
        if unique_objects_cumulative:
            Logger.current_logger().report_histogram(
                title="Трекирование объектов",
                series=f"Трекируемые объекты - {model_name}",
                values=unique_objects_cumulative,
                xaxis="Номер фрейма",
                yaxis="Трекируемые объекты"
            )
        
        # Гистограмма стабильности трекинга
        if frame_changes:
            Logger.current_logger().report_histogram(
                title="Анализ стабильности трекинга", 
                series=f"Изменения между фреймами - {model_name}",
                values=frame_changes,
                xaxis="Изменения в количестве объектов", 
                yaxis="Частота"
            )
    
    # Lables к колонкам для графика распределения confidence
    confidence_labels = ["0,5 - 0,6", "0,6 - 0,7", "0,7 - 0,8", "0,8 - 0,9", "0,9 - 1.0"]
    
    # Логируем распределение confidence
    Logger.current_logger().report_histogram(
        title="Распределение объектов по уровням confidence",
        series=f"Распределение confidence - {model_name}",
        iteration=0,
        xlabels=confidence_labels,
        values=confidence_distribution,
        yaxis="Количество объектов",
        xaxis="Диапазоны confidence"
    )
    
    # Также можно логировать как гистограмму
    # Logger.current_logger().report_histogram(
    #     title="Распределение confidence",
    #     series=f"Распределение объектов по confidence - {model_name}",
    #     values=[confidence_distribution],
    #     iteration=0,
    #     xaxis="Диапазоны confidence",
    #     yaxis="Количество объектов",
    #     labels=confidence_labels
    # )

    # Сохраняем итоговую статистику
    task.get_logger().report_single_value("Всего фреймов", frame_count)
    task.get_logger().report_single_value("Всего объектов на всех фреймах", total_objects_detected)
    task.get_logger().report_single_value("Всреднем объектов на фрейм", total_objects_detected / max(frame_count, 1))
    task.get_logger().report_single_value("Трекируемых объектов", len(unique_object_ids))
    
    # # Логируем распределение confidence по диапазонам
    # for i, label in enumerate(confidence_labels):
    #     task.get_logger().report_single_value(f"Объекты в диапазоне {label}", confidence_distribution[i])
    
    # Загружаем обработанное видео как артефакт
    task.upload_artifact("processed_video", output_path)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # # Выводим статистику по confidence
    # print("\nРаспределение объектов по confidence:")
    # for i, label in enumerate(confidence_labels):
    #     print(f"{label}: {confidence_distribution[i]} объектов ({confidence_distribution[i]/max(total_objects_detected, 1)*100:.2f}%)")
    
    print(f"\nОбработанное видео с трекингом сохранено в {output_path}")
    print(f"Всего обработано фреймов: {frame_count}")
    print(f"Всего обнаружено объектов: {total_objects_detected}")
    print(f"Уникальных объектов отслежено: {len(unique_object_ids)}")

    task.close()

if __name__ == "__main__":
    model_name = "yolo12x"
    confidence = 0.5
    # input_names = ["spb_dvorzovy_most_001", "spb_gostiny_dvor_001", "spb_gostiny_dvor_002", "spb_nevsky_annichkov_most_001", "spb_nevsky_annichkov_most_002", "spb_zagorodny_proezd_001"]
    input_names = ["out-corrupt-spb_gostiny_dvor_001-brightness-s5",
                   "out-corrupt-spb_gostiny_dvor_001-contrast-s5",
                   "out-corrupt-spb_gostiny_dvor_001-defocus_blur-s5",
                   "out-corrupt-spb_gostiny_dvor_001-gaussian_noise-s5",
                   "out-corrupt-spb_gostiny_dvor_001-impulse_noise-s5",
                   "out-corrupt-spb_gostiny_dvor_001-jpeg_compression-s5",
                   "out-corrupt-spb_gostiny_dvor_001-motion_blur-s5",
                   "out-corrupt-spb_gostiny_dvor_001-saturate-s5",
                   "out-corrupt-spb_gostiny_dvor_001-shot_noise-s5",
                   "out-corrupt-spb_gostiny_dvor_001-spatter-s5",
                   "out-corrupt-spb_gostiny_dvor_001-speckle_noise-s5",
                   "out-corrupt-spb_gostiny_dvor_001-zoom_blur-s5"
                   ]
    for input_name in input_names:
        time_start = datetime.now()
        date_stamp = time_start.strftime("%Y-%m-%d_%H-%M-%S")
        output_name = f"out-{input_name}-{model_name}-conf-{confidence}_{date_stamp}"
        main(model_name, f"{input_name}.mp4", f"{output_name}.mp4", confidence, date_stamp)
        print(f'Время работы: {datetime.now() - time_start} сек.')