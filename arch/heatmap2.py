import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from clearml import Task, Logger
import os
import argparse
from scipy.ndimage import gaussian_filter
import sys

def add_gaussian_heatmap(heatmap, center_x, center_y, bbox_width, bbox_height):
    """
    Добавляет гауссово распределение в heatmap на основе размера bounding box
    """
    # Определяем размер ядра на основе размера объекта
    kernel_size = min(bbox_width, bbox_height) // 2
    kernel_size = max(kernel_size, 10)  # Минимальный размер
    kernel_size = min(kernel_size, 50)  # Максимальный размер
    
    # Создаем 2D гауссово распределение
    k = kernel_size // 2
    if k < 1:
        return
    
    y, x = np.ogrid[-k:k+1, -k:k+1]
    
    # Стандартное отклонение пропорционально размеру объекта
    sigma = kernel_size / 3
    
    # Гауссово распределение
    gaussian = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    
    # Нормализуем, чтобы максимальное значение было 1
    if np.max(gaussian) > 0:
        gaussian = gaussian / np.max(gaussian)
    
    # Вычисляем границы для вставки
    h, w = heatmap.shape
    x_start = max(0, center_x - k)
    x_end = min(w, center_x + k + 1)
    y_start = max(0, center_y - k)
    y_end = min(h, center_y + k + 1)
    
    # Вычисляем соответствующие границы в гауссовом ядре
    g_x_start = max(0, k - center_x)
    g_x_end = g_x_start + (x_end - x_start)
    g_y_start = max(0, k - center_y)
    g_y_end = g_y_start + (y_end - y_start)
    
    # Добавляем гауссово распределение в heatmap
    if x_end > x_start and y_end > y_start:
        heatmap[y_start:y_end, x_start:x_end] += gaussian[g_y_start:g_y_end, g_x_start:g_x_end]


def add_anisotropic_gaussian(heatmap, center_x, center_y, bbox_width, bbox_height):
    """
    Добавляет анизотропное гауссово распределение (растянутое по осям)
    """
    kernel_size = max(bbox_width, bbox_height)
    k = kernel_size // 2
    
    if k < 1:
        return
    
    # Создаем координатную сетку
    y, x = np.ogrid[-k:k+1, -k:k+1]
    
    # Разные сигмы для разных направлений
    sigma_x = max(bbox_width / 4, 2)
    sigma_y = max(bbox_height / 4, 2)
    
    # Анизотропное гауссово распределение
    gaussian = np.exp(-(x*x/(2*sigma_x*sigma_x) + y*y/(2*sigma_y*sigma_y)))
    
    if np.max(gaussian) > 0:
        gaussian = gaussian / np.max(gaussian)
    
    # Вычисляем границы для вставки
    h, w = heatmap.shape
    x_start = max(0, center_x - k)
    x_end = min(w, center_x + k + 1)
    y_start = max(0, center_y - k)
    y_end = min(h, center_y + k + 1)
    
    g_x_start = max(0, k - center_x)
    g_x_end = g_x_start + (x_end - x_start)
    g_y_start = max(0, k - center_y)
    g_y_end = g_y_start + (y_end - y_start)
    
    # Добавляем в heatmap
    if x_end > x_start and y_end > y_start:
        heatmap[y_start:y_end, x_start:x_end] += gaussian[g_y_start:g_y_end, g_x_start:g_x_end]


def add_smoothed_gaussian(heatmap, center_x, center_y, bbox_width, bbox_height):
    """
    Создает временный патч с гауссом и применяет фильтрацию
    """
    kernel_size = min(bbox_width, bbox_height)
    k = max(kernel_size // 2, 5)
    
    # Создаем временный патч
    patch_size = min(k * 2, 100)  # Ограничиваем размер патча
    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    # Рисуем круг в центре патча
    center = patch_size // 2
    cv2.circle(patch, (center, center), min(k, center), 1.0, -1)
    
    # Применяем гауссово размытие
    sigma = k / 2
    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Нормализуем
    if np.max(patch) > 0:
        patch = patch / np.max(patch)
    
    # Вычисляем область вставки
    h, w = heatmap.shape
    x_start = max(0, center_x - patch_size//2)
    x_end = min(w, x_start + patch_size)
    y_start = max(0, center_y - patch_size//2)
    y_end = min(h, y_start + patch_size)
    
    # Вычисляем область патча для вставки
    p_x_start = max(0, patch_size//2 - center_x)
    p_x_end = p_x_start + (x_end - x_start)
    p_y_start = max(0, patch_size//2 - center_y)
    p_y_end = p_y_start + (y_end - y_start)
    
    # Добавляем патч в heatmap
    if x_end > x_start and y_end > y_start and p_x_end > p_x_start and p_y_end > p_y_start:
        heatmap[y_start:y_end, x_start:x_end] += patch[p_y_start:p_y_end, p_x_start:p_x_end]


def add_stats_overlay(frame, objects_count, frame_num, fps):
    """
    Добавляет статистику на видео
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (280, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    cv2.putText(frame, f"Objects: {objects_count}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_num}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def main(model_name: str, input_file: str, output_file: str, confidence: float, 
         skip_frames: int = 1, heatmap_method: str = 'gaussian'):
    """
    Основная функция для создания heatmap отслеживания автомобилей
    
    Args:
        model_name: имя модели YOLO
        input_file: имя входного видеофайла
        output_file: имя выходного видеофайла
        confidence: порог уверенности
        skip_frames: количество пропускаемых кадров
        heatmap_method: метод создания heatmap ('gaussian', 'anisotropic', 'smoothed')
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
            task_name=f"hitmap_car_tracking_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.inference
        )
        
        # Устанавливаем параметры задачи
        task.set_parameter("model", model_name)
        task.set_parameter("confidence_threshold", confidence)
        task.set_parameter("iou_threshold", 0.4)
        task.set_parameter("allowed_classes", [0, 2, 3, 5, 6, 7, 8])
        task.set_parameter("heatmap_method", heatmap_method)
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
        heatmap_alpha = 0.5  # Прозрачность heatmap при наложении
        heatmap_decay = 0.95  # Затухание heatmap со временем
        legend_height = 50  # Высота легенды
        
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
                        
                        # Получаем координаты и размеры
                        x1, y1, x2, y2 = map(int, xyxy)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        class_name = results[0].names[int(box.cls[0])]
                        obj_id = int(results[0].boxes.id[i].cpu().numpy())
                        
                        # Рисуем bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_yellow, 2)
                        cv2.putText(frame, f'{class_name} {obj_id}', (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yellow, 2)
                        
                        # Добавляем гауссов шум в heatmap в зависимости от выбранного метода
                        if heatmap_method == 'gaussian':
                            add_gaussian_heatmap(heatmap, center_x, center_y, bbox_width, bbox_height)
                        elif heatmap_method == 'anisotropic':
                            add_anisotropic_gaussian(heatmap, center_x, center_y, bbox_width, bbox_height)
                        elif heatmap_method == 'smoothed':
                            add_smoothed_gaussian(heatmap, center_x, center_y, bbox_width, bbox_height)
                        else:
                            # По умолчанию - простой круг
                            kernel_size = min(bbox_width, bbox_height) // 2
                            kernel_size = max(kernel_size, 10)
                            cv2.circle(heatmap, (center_x, center_y), kernel_size, 1.0, -1)
            
            # Создаем визуализацию heatmap
            if np.max(heatmap) > 0:
                # Нормализуем heatmap
                heatmap_normalized = np.zeros_like(heatmap)
                cv2.normalize(heatmap, heatmap_normalized, 0, 255, cv2.NORM_MINMAX)
                heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            else:
                heatmap_colored = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            
            # Накладываем heatmap на оригинальный кадр
            frame_with_heatmap = cv2.addWeighted(frame, 1 - heatmap_alpha, heatmap_colored, heatmap_alpha, 0)
            
            # Добавляем статистику на кадр
            elapsed_time = (datetime.now() - start_time).total_seconds()
            current_fps = frame_count / max(elapsed_time, 0.001)
            frame_with_heatmap = add_stats_overlay(frame_with_heatmap, objects_in_frame, frame_count, current_fps)
            
            # Создаем легенду для heatmap
            legend = np.zeros((legend_height, frame_width, 3), dtype=np.uint8)
            
            # Создаем градиент для легенды
            for i in range(frame_width):
                color_value = int(i / frame_width * 255)
                color = cv2.applyColorMap(np.array([[color_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                cv2.line(legend, (i, 0), (i, legend_height), color.tolist(), 1)
            
            # Добавляем текст к легенде
            cv2.putText(legend, "Low", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(legend, "High", (frame_width - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(legend, f"Object Density Heatmap ({heatmap_method})", (frame_width // 2 - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Объединяем кадр с heatmap и легендой
            final_frame = np.vstack([frame_with_heatmap, legend])
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
        
        # Вывод результатов
        print("\n" + "="*50)
        print("ОБРАБОТКА ЗАВЕРШЕНА")
        print("="*50)
        print(f"Обработанное видео: {output_path}")
        print(f"Heatmap сохранена: {heatmap_output_path}")
        print(f"Всего обработано фреймов: {frame_count}")
        print(f"Всего обнаружено объектов: {total_objects_detected}")
        print(f"Среднее количество объектов на кадр: {total_objects_detected / max(frame_count, 1):.2f}")
        print(f"Время обработки: {datetime.now() - start_time}")
        print("="*50)
        
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
    parser = argparse.ArgumentParser(description='Создание heatmap отслеживания автомобилей с гауссовым шумом')
    parser.add_argument('--model', type=str, default='yolov8n', 
                        help='Имя модели YOLO (по умолчанию: yolov8n)')
    parser.add_argument('--input', type=str, default='cars_1_1', 
                        help='Имя входного файла (без расширения, по умолчанию: cars_1_1)')
    parser.add_argument('--confidence', type=float, default=0.5, 
                        help='Порог уверенности (по умолчанию: 0.5)')
    parser.add_argument('--skip-frames', type=int, default=1, 
                        help='Пропуск кадров для ускорения (по умолчанию: 1 - обрабатывать все)')
    parser.add_argument('--method', type=str, default='gaussian', 
                        choices=['gaussian', 'anisotropic', 'smoothed', 'circle'],
                        help='Метод создания heatmap (по умолчанию: gaussian)')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Отключить логирование в ClearML')
    
    args = parser.parse_args()
    
    # Формируем имя выходного файла
    output_name = f"hitmap-{args.input}-{args.model}-{args.method}-conf-{args.confidence}"
    if args.skip_frames > 1:
        output_name += f"-skip{args.skip_frames}"
    
    print("="*50)
    print("ЗАПУСК HEATMAP ТРЕКИНГА АВТОМОБИЛЕЙ")
    print("="*50)
    print(f"Модель: {args.model}")
    print(f"Входной файл: {args.input}.mp4")
    print(f"Выходной файл: {output_name}.mp4")
    print(f"Порог уверенности: {args.confidence}")
    print(f"Метод heatmap: {args.method}")
    print(f"Пропуск кадров: {args.skip_frames}")
    print(f"ClearML: {'Отключен' if args.no_clearml else 'Включен'}")
    print("="*50)
    
    # Если ClearML отключен, перенаправляем логирование в файл
    if args.no_clearml:
        # Временно отключаем ClearML
        os.environ["CLEARML_DISABLE"] = "1"
    
    time_start = datetime.now()
    
    # Запуск основной функции
    main(
        model_name=args.model,
        input_file=f"{args.input}.mp4",
        output_file=f"{output_name}.mp4",
        confidence=args.confidence,
        skip_frames=args.skip_frames,
        heatmap_method=args.method
    )
    
    print(f'\nОбщее время работы: {datetime.now() - time_start} сек.')