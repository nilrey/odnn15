import os
from pathlib import Path
import cv2
import math
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Dataset, Logger


def main(cml_project_name: str, cml_task_name: str,
        model_name: str, input_video: str, output_video: str, confidence: float, date_stamp: datetime):
    # Инициализируем ClearML Task
    task = Task.init(
        project_name=cml_project_name,
        task_name=cml_task_name,
        task_type=Task.TaskTypes.inference
    )
    
    model = YOLO(f'models/{model_name}.pt')

    # Устанавливаем параметры задачи
    task.set_parameter("model", model_name)
    task.set_parameter("input_video", input_video)
    task.set_parameter("output_video", output_video)
    task.set_parameter("confidence_threshold", confidence)
    task.set_parameter("iou_threshold", 0.4)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {input_video}")
        task.close()
        return

    # Получаем ИСХОДНЫЕ параметры видео для записи
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Вычисляем размер для модели (кратный 32)
    # YOLO обычно использует высоту как основной параметр
    model_size = 32 * math.ceil(orig_height / 32)
    
    # Для записи видео используем ИСХОДНЫЕ размеры
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (orig_width, orig_height))
    
    if not out.isOpened():
        print(f"Ошибка: Не удалось создать VideoWriter для {output_video}")
        cap.release()
        task.close()
        return

    color_blue = (255, 0, 0)
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)

    classes_statistics = [{"name":"person", "count":0}, {"name":"car", "count":0}]
    classes_indexes = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов всех автомобилей
    task.set_parameter("allowed_classes", classes_indexes)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        # Сохраняем оригинальный кадр для записи
        orig_frame = frame.copy()

        # Используем модель для анализа текущего кадра с отслеживанием
        # Используем вычисленный размер, кратный 32
        results = model.track(frame, persist=True, imgsz=model_size, iou=0.4, verbose=False)

        objects_in_frame = 0
        
        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = box.conf[0]
                class_index = int(box.cls[0])
                if class_index in classes_indexes and conf > confidence:
                    objects_in_frame += 1
                    class_name = results[0].names[int(box.cls[0])]
                    obj_id = int(results[0].boxes.id[i])  # Получаем ID распознанного объекта
                    label = f'{class_name} {obj_id}'
                    xyxy = box.xyxy[0]
                    if class_index == 0:
                        classes_statistics[0]["count"] +=1
                    else:
                        classes_statistics[1]["count"] +=1


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

                    # Рисуем bounding box и ID на оригинальном кадре
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color_yellow, 1)
                    cv2.putText(orig_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 1)

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
        # Используем оригинальный кадр с аннотациями (без изменения размера)
        out.write(orig_frame)
        frame_count += 1
        
        # Периодический вывод в консоль для отладки
        if frame_count % 10 == 0:  # Каждые 30 фреймов
            print(f"Фрейм {frame_count} из {frames} ")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

    # Сохраняем итоговую статистику
    task.get_logger().report_single_value("Всего фреймов", frame_count)
    task.get_logger().report_single_value("Всего объектов на всех фреймах", total_objects_detected)
    # task.get_logger().report_single_value("Среднее объектов на фрейм", total_objects_detected / max(frame_count, 1))
    task.get_logger().report_single_value("Трекируемых объектов", len(unique_object_ids))
    
    # Статистика по категориям
    for cat_info in classes_statistics:
        task.get_logger().report_single_value( f"Объекты категории {cat_info['name']}", cat_info['count'])
    
    # Логируем распределение confidence по диапазонам
    for i, label in enumerate(confidence_labels):
        task.get_logger().report_single_value(f"Объекты в диапазоне {label}", confidence_distribution[i])
    
    # Загружаем обработанное видео как артефакт
    task.upload_artifact("processed_video", output_video)
    
    # Выводим статистику по confidence
    print("\nРаспределение объектов по confidence:")
    for i, label in enumerate(confidence_labels):
        percent = (confidence_distribution[i]/max(total_objects_detected, 1)*100)
        print(f"{label}: {confidence_distribution[i]} объектов ({percent:.2f}%)")
    
    print(f"\nОбработанное видео с трекингом сохранено в {output_video}")
    print(f"Всего обработано фреймов: {frame_count}")
    print(f"Всего обнаружено объектов: {total_objects_detected}")
    print(f"Уникальных объектов отслежено: {len(unique_object_ids)}")

    task.close()

if __name__ == "__main__":
    cml_project_name = "ICIE Detection Project New"
    model_name = "yolo12x"
    confidence = 0.5

    current_file = Path(__file__).resolve()
    path_input = current_file.parent / "data" / "input" / "001"
    # файлы из папки 
    all_files = os.listdir(path_input)
    # убираем расширение из имен
    input_names = [os.path.splitext(f)[0] for f in all_files if f.endswith('.mp4')]

    # input_names = ["spb_dvorzovy_most_001", "spb_gostiny_dvor_001", "spb_gostiny_dvor_002", "spb_nevsky_annichkov_most_001", "spb_nevsky_annichkov_most_002", "spb_zagorodny_proezd_001"]
    # input_names = ["spb-cam1-short-001"]
    
    corruption_types = [] #["brightness", "zoom_blur", "defocus_blur", "gaussian_noise", "impulse_noise", "jpeg_compression", "saturate", "shot_noise", "spatter", "speckle_noise", "motion_blur", "contrast"]
    
    for input_name in input_names:
        time_start = datetime.now()
        date_stamp = time_start.strftime("%Y-%m-%d_%H-%M-%S")
        output_name = f"out-{input_name}-{model_name}-conf-{confidence}_{date_stamp}"
        
        cml_task_name = f"{input_name}"
        # # Извлекаем corruption_type из имени файла
        # for corruption_type in corruption_types:
        #     if corruption_type in input_name:
        #         cml_task_name = f"{input_name}-{corruption_type}"
        #         break
        # else:
        #     cml_task_name = f"{input_name}-original"
        
        main(cml_project_name, 
             cml_task_name, 
             model_name, 
             f'{path_input}/{input_name}.mp4', 
             f'data/output/001/{output_name}.mp4', 
             confidence, 
             date_stamp
             )
        print(f'Время работы: {datetime.now() - time_start} сек.')