import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Logger
import os
import argparse
import sys
import imagecorruptions
from imagecorruptions import corrupt

# Константы для corruption методов
CORRUPTION_METHODS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 
    'speckle_noise', 'gaussian_blur', 'glass_blur',
    'defocus_blur', 'motion_blur', 'zoom_blur',
    'fog', 'frost', 'snow', 'spatter'
]

def apply_corruption_to_mask(frame, mask, corruption_type='gaussian_noise', severity=3):
    """
    Применяет corruption только к области, указанной маской
    
    Args:
        frame: исходный кадр
        mask: бинарная маска (белое - область применения)
        corruption_type: тип corruption
        severity: сила эффекта
    
    Returns:
        кадр с corruption только в области маски
    """
    if np.sum(mask) == 0:
        return frame
    
    result = frame.copy()
    
    # Создаем копию кадра для применения corruption
    corrupted_frame = frame.copy()
    
    try:
        # Применяем corruption ко всему кадру
        corrupted_full = corrupt(corrupted_frame, corruption_name=corruption_type, severity=severity)
        
        # Используем маску для выбора только нужных пикселей
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = frame * (1 - mask_3channel) + corrupted_full * mask_3channel
        result = result.astype(np.uint8)
        
    except Exception as e:
        print(f"Ошибка при применении corruption: {e}")
    
    return result


def apply_heatmap_based_corruption(frame, heatmap, corruption_type='gaussian_noise', 
                                   threshold=0.3, severity=3, draw_contours=False):
    """
    Применяет corruption на основе heatmap ТОЛЬКО внутри областей с высоким "теплом"
    
    Args:
        frame: исходный кадр
        heatmap: карта тепла
        corruption_type: тип corruption
        threshold: порог для применения corruption
        severity: сила эффекта
        draw_contours: рисовать ли красные контуры
    
    Returns:
        кадр с corruption в областях с высоким "теплом"
    """
    # Нормализуем heatmap
    if np.max(heatmap) > 0:
        heatmap_norm = heatmap / np.max(heatmap)
    else:
        heatmap_norm = heatmap
    
    # Создаем маску для областей с высоким теплом
    mask = (heatmap_norm > threshold).astype(np.uint8) * 255
    
    if np.sum(mask) == 0:
        return frame
    
    # Применяем морфологические операции для очистки маски
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Применяем corruption только к области маски
    result_frame = apply_corruption_to_mask(frame, mask, corruption_type, severity)
    
    # Рисуем красные контуры (опционально)
    if draw_contours:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_frame, contours, -1, (0, 0, 255), 2)
    
    return result_frame


def add_stats_overlay(frame, objects_count, frame_num, fps, corruption_type, severity, threshold):
    """
    Добавляет статистику на видео
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(frame, f"Objects: {objects_count}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Corruption: {corruption_type}", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Severity: {severity} | Threshold: {threshold}", (20, 145), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Red areas = corrupted regions", (20, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return frame


def create_corruption_legend(frame_width, legend_height, corruption_type, severity, threshold):
    """
    Создает легенду с информацией о corruption
    """
    legend = np.zeros((legend_height, frame_width, 3), dtype=np.uint8)
    
    # Добавляем информацию
    cv2.putText(legend, f"Corruption: {corruption_type} | Severity: {severity} | Threshold: {threshold}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(legend, "Red areas = corrupted regions (inside only)", 
                (frame_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return legend


def main(model_name: str, input_file: str, output_file: str, confidence: float, 
         skip_frames: int = 1, corruption_type: str = 'gaussian_noise', 
         severity: int = 3, threshold: float = 0.3, show_boxes: bool = False,
         draw_contours: bool = True):
    """
    Основная функция для создания heatmap с corruption
    
    Args:
        model_name: имя модели YOLO
        input_file: имя входного видеофайла
        output_file: имя выходного видеофайла
        confidence: порог уверенности
        skip_frames: количество пропускаемых кадров
        corruption_type: тип corruption
        severity: сила эффекта (1-5)
        threshold: порог heatmap для применения corruption
        show_boxes: показывать ли bounding boxes
        draw_contours: рисовать ли красные контуры
    """
    
    try:
        # Проверка существования файлов
        model_path = f'models/{model_name}.pt'
        video_path = f'data/input/{input_file}'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        
        # Создание директорий для output
        os.makedirs('data/output', exist_ok=True)
        
        # Инициализируем ClearML Task
        task = Task.init(
            project_name="odnn15",
            task_name=f"corruption_{corruption_type}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.inference
        )
        
        # Устанавливаем параметры задачи
        task.set_parameter("model", model_name)
        task.set_parameter("confidence_threshold", confidence)
        task.set_parameter("iou_threshold", 0.4)
        task.set_parameter("allowed_classes", [0, 2, 3, 5, 6, 7, 8])
        task.set_parameter("corruption_type", corruption_type)
        task.set_parameter("severity", severity)
        task.set_parameter("heatmap_threshold", threshold)
        task.set_parameter("skip_frames", skip_frames)
        
        # Загружаем модель
        print(f"Загрузка модели {model_name}...")
        model = YOLO(model_path)
        
        # Открываем видео
        print(f"Открытие видео {video_path}...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception("Не удалось открыть видео")
        
        # Получаем параметры видео
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Видео: {frame_width}x{frame_height}, {fps} fps, {total_frames} кадров")
        
        # Цвета для отрисовки
        color_yellow = (0, 255, 255)
        
        # Разрешенные классы (только автомобили)
        allowed_indices = {0, 2, 3, 5, 6, 7, 8}
        
        # Параметры heatmap
        heatmap_decay = 0.95  # Затухание heatmap со временем
        legend_height = 60  # Высота легенды
        
        # Высота итогового кадра с учетом легенды
        final_height = frame_height + legend_height
        
        # Инициализируем VideoWriter
        output_path = f'data/output/{output_file}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, final_height))
        
        # Инициализация heatmap
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Статистика
        frame_count = 0
        total_objects_detected = 0
        object_counts = []
        frame_changes = []
        previous_count = 0
        frame_skip_counter = 0
        last_final_frame = None
        
        # Для расчета FPS
        start_time = datetime.now()
        
        print("Начало обработки видео...")
        print(f"Тип corruption: {corruption_type}, Severity: {severity}, Threshold: {threshold}")
        print(f"Применяем corruption ТОЛЬКО внутри красных областей")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_skip_counter += 1
            if frame_skip_counter % skip_frames != 0 and last_final_frame is not None:
                # Пропускаем кадр, но записываем предыдущий результат
                out.write(last_final_frame)
                frame_count += 1
                continue
            
            # Сохраняем оригинальный кадр
            original_frame = frame.copy()
            
            # Используем модель для анализа текущего кадра с отслеживанием
            results = model.track(frame, persist=True, imgsz=frame_width, iou=0.4, verbose=False)
            
            objects_in_frame = 0
            
            # Обновляем heatmap с затуханием
            heatmap *= heatmap_decay
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                for i, box in enumerate(results[0].boxes):
                    conf = box.conf[0]
                    if int(box.cls[0]) in allowed_indices and conf > confidence:
                        objects_in_frame += 1
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        # Получаем координаты
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Добавляем в heatmap
                        kernel_size = min((x2 - x1) // 2, (y2 - y1) // 2, 30)
                        kernel_size = max(kernel_size, 5)
                        cv2.circle(heatmap, (center_x, center_y), kernel_size, 1.0, -1)
                        
                        if show_boxes:
                            class_name = results[0].names[int(box.cls[0])]
                            obj_id = int(results[0].boxes.id[i].cpu().numpy())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color_yellow, 2)
                            cv2.putText(frame, f'{class_name} {obj_id}', (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 2)
            
            # Применяем corruption на основе heatmap ТОЛЬКО внутри областей
            corrupted_frame = apply_heatmap_based_corruption(
                original_frame, heatmap, 
                corruption_type=corruption_type,
                threshold=threshold,
                severity=severity,
                draw_contours=draw_contours
            )
            
            # Добавляем статистику на кадр
            elapsed_time = (datetime.now() - start_time).total_seconds()
            current_fps = frame_count / max(elapsed_time, 0.001)
            corrupted_frame = add_stats_overlay(
                corrupted_frame, objects_in_frame, frame_count, 
                current_fps, corruption_type, severity, threshold
            )
            
            # Создаем легенду
            legend = create_corruption_legend(frame_width, legend_height, corruption_type, severity, threshold)
            
            # Объединяем кадр с легендой
            final_frame = np.vstack([corrupted_frame, legend])
            last_final_frame = final_frame.copy()
            
            # Сохраняем для статистики
            object_counts.append(objects_in_frame)
            
            # Анализ изменений между фреймами
            if frame_count > 0:
                change = abs(objects_in_frame - previous_count)
                frame_changes.append(change)
            
            previous_count = objects_in_frame
            
            # Логируем в ClearML
            if task and Logger.current_logger():
                Logger.current_logger().report_scalar(
                    title="Object Detection Statistics",
                    series="Objects per Frame",
                    value=objects_in_frame,
                    iteration=frame_count
                )
                
                # Логируем площадь corrupted областей
                if np.max(heatmap) > 0:
                    heatmap_norm = heatmap / np.max(heatmap)
                    corrupted_area = np.sum(heatmap_norm > threshold)
                    total_area = frame_height * frame_width
                    area_percent = (corrupted_area / total_area) * 100
                    
                    Logger.current_logger().report_scalar(
                        title="Corruption Statistics",
                        series="Corrupted Area %",
                        value=area_percent,
                        iteration=frame_count
                    )
            
            # Обновляем статистику
            total_objects_detected += objects_in_frame
            
            # Запись обработанного кадра
            out.write(final_frame)
            frame_count += 1
            
            # Прогресс в консоль
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Прогресс: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"Объектов в кадре: {objects_in_frame}, "
                      f"FPS: {current_fps:.1f}")
        
        print("Обработка видео завершена")
        
        # Сохраняем финальную heatmap
        heatmap_output_path = f'data/output/heatmap_{output_file.split(".")[0]}.png'
        
        if np.max(heatmap) > 0:
            heatmap_final = np.zeros_like(heatmap)
            cv2.normalize(heatmap, heatmap_final, 0, 255, cv2.NORM_MINMAX)
            heatmap_final_colored = cv2.applyColorMap(heatmap_final.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            heatmap_final_colored = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        cv2.imwrite(heatmap_output_path, heatmap_final_colored)
        
        # Логирование в ClearML
        if task:
            # Загружаем артефакты
            task.upload_artifact("heatmap_image", heatmap_output_path)
            task.upload_artifact("processed_video", output_path)
            
            # Логируем графики
            if object_counts:
                Logger.current_logger().report_histogram(
                    title="Object Detection Analysis",
                    series=f"Objects per Frame - {model_name}",
                    values=object_counts,
                    xaxis="Number of Objects",
                    yaxis="Number of Frames"
                )
            
            if frame_changes:
                Logger.current_logger().report_histogram(
                    title="Tracking Stability Analysis",
                    series=f"Frame-to-Frame Changes - {model_name}",
                    values=frame_changes,
                    xaxis="Objects Change Count",
                    yaxis="Frequency"
                )
            
            # Сохраняем итоговую статистику
            avg_objects = total_objects_detected / max(frame_count, 1)
            task.get_logger().report_single_value("Total Frames Processed", frame_count)
            task.get_logger().report_single_value("Total Objects Detected", total_objects_detected)
            task.get_logger().report_single_value("Average Objects per Frame", avg_objects)
            
            # Средний процент corrupted области
            if frame_count > 0:
                task.get_logger().report_single_value("Average Corrupted Area %", area_percent)
        
        # Вывод результатов
        print("\n" + "="*60)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("="*60)
        print(f"Обработанное видео: {output_path}")
        print(f"Heatmap сохранена: {heatmap_output_path}")
        print(f"Тип corruption: {corruption_type}")
        print(f"Severity: {severity}")
        print(f"Порог: {threshold}")
        print(f"Всего обработано фреймов: {frame_count}")
        print(f"Всего обнаружено объектов: {total_objects_detected}")
        print(f"Среднее количество объектов на кадр: {total_objects_detected / max(frame_count, 1):.2f}")
        print(f"Время обработки: {datetime.now() - start_time}")
        print("="*60)
        
        # Освобождаем ресурсы
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Создание heatmap с corruption ТОЛЬКО внутри областей обнаружения')
    parser.add_argument('--model', type=str, default='yolov8n', 
                        help='Имя модели YOLO (по умолчанию: yolov8n)')
    parser.add_argument('--input', type=str, default='cars_1_1', 
                        help='Имя входного файла (без расширения, по умолчанию: cars_1_1)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Порог уверенности (по умолчанию: 0.5)')
    parser.add_argument('--skip-frames', type=int, default=1, 
                        help='Пропуск кадров для ускорения (по умолчанию: 1)')
    parser.add_argument('--corruption', type=str, default='gaussian_noise', 
                        choices=CORRUPTION_METHODS,
                        help=f'Тип corruption (по умолчанию: gaussian_noise)')
    parser.add_argument('--severity', type=int, default=5, choices=range(1, 6),
                        help='Сила эффекта 1-5 (по умолчанию: 5)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Порог heatmap для применения corruption 0-1 (по умолчанию: 0.3)')
    parser.add_argument('--no-boxes', action='store_true',
                        help='Не показывать bounding boxes')
    parser.add_argument('--no-contours', action='store_true',
                        help='Не рисовать красные контуры')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Отключить логирование в ClearML')
    parser.add_argument('--list-corruptions', action='store_true',
                        help='Показать список доступных corruption методов и выйти')
    
    args = parser.parse_args()
    
    # Показать список corruption методов
    if args.list_corruptions:
        print("\nДоступные методы corruption:")
        for i, method in enumerate(CORRUPTION_METHODS, 1):
            print(f"  {i}. {method}")
        print()
        sys.exit(0)
    
    # Формируем имя выходного файла
    output_name = f"corrupt-inside-{args.input}-{args.model}-{args.corruption}-sev{args.severity}-th{args.threshold}"
    if args.skip_frames > 1:
        output_name += f"-skip{args.skip_frames}"
    if args.no_boxes:
        output_name += "-noboxes"
    if args.no_contours:
        output_name += "-nocontours"
    
    print("="*60)
    print("ЗАПУСК HEATMAP С CORRUPTION (ТОЛЬКО ВНУТРИ ОБЛАСТЕЙ)")
    print("="*60)
    print(f"Модель: {args.model}")
    print(f"Входной файл: {args.input}.mp4")
    print(f"Выходной файл: {output_name}.mp4")
    print(f"Порог уверенности: {args.confidence}")
    print(f"Тип corruption: {args.corruption}")
    print(f"Severity: {args.severity}")
    print(f"Порог heatmap: {args.threshold}")
    print(f"Показывать boxes: {'Нет' if args.no_boxes else 'Да'}")
    print(f"Рисовать контуры: {'Нет' if args.no_contours else 'Да'}")
    print(f"Пропуск кадров: {args.skip_frames}")
    print(f"ClearML: {'Отключен' if args.no_clearml else 'Включен'}")
    print("="*60)
    
    # Если ClearML отключен, перенаправляем логирование в файл
    if args.no_clearml:
        os.environ["CLEARML_DISABLE"] = "1"
    
    time_start = datetime.now()
    
    # Запуск основной функции
    main(
        model_name=args.model,
        input_file=f"{args.input}.mp4",
        output_file=f"{output_name}.mp4",
        confidence=args.confidence,
        skip_frames=args.skip_frames,
        corruption_type="zoom_blur", #args.corruption,
        severity=args.severity,
        threshold=args.threshold,
        show_boxes=False,
        draw_contours=False
    )
    
    print(f'\nОбщее время работы: {datetime.now() - time_start} сек.')