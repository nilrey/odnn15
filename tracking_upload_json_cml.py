import os
import json
from pathlib import Path
from datetime import datetime
from clearml import Task, Logger


def load_annotations(json_path: str):
    """Загружает аннотации из JSON файла"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создаем словарь для быстрого доступа к аннотациям по image_id
    annotations_dict = {}
    
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_dict:
            annotations_dict[image_id] = []
        
        annotations_dict[image_id].append({
            'category_id': annotation['category_id'],
            'bbox': annotation['bbox'],
            'track_id': annotation['attributes']['track_id'],
            'area': annotation['area']
        })
    
    return annotations_dict, data['categories']


def main(cml_project_name: str, cml_task_name: str, input_file: str):
    # Инициализируем ClearML Task
    task = Task.init(
        project_name=cml_project_name,
        task_name=cml_task_name,
        task_type=Task.TaskTypes.inference
    )
    
    print(f"Загрузка аннотаций из {input_file}")
    annotations, categories = load_annotations(input_file)
    print(f"Загружено аннотаций для {len(annotations)} изображений")
    
    task.set_parameter("annotations_file", input_file)
    task.set_parameter("total_images", len(annotations))
    
    # Подготовим статистику
    frame_counts = []
    objects_per_frame = []
    unique_track_ids = set()
    unique_tracks_cumulative = []
    
    # Распределение по категориям
    category_distribution = {}
    for category in categories:
        category_distribution[category['id']] = {
            'name': category['name'],
            'count': 0
        }
    
    # Анализ аннотаций по кадрам
    for frame_id in sorted(annotations.keys()):
        frame_annotations = annotations[frame_id]
        objects_in_frame = len(frame_annotations)
        
        # Сохраняем для графиков
        frame_counts.append(frame_id)
        objects_per_frame.append(objects_in_frame)
        
        # Собираем уникальные track_id
        for ann in frame_annotations:
            unique_track_ids.add(ann['track_id'])
            category_id = ann['category_id']
            if category_id in category_distribution:
                category_distribution[category_id]['count'] += 1
        
        # Нарастающий итог уникальных треков
        unique_tracks_cumulative.append(len(unique_track_ids))
        
        # Логируем количество объектов для текущего фрейма
        Logger.current_logger().report_scalar(
            title="Статистика обнаружения объектов",
            series="Объектов на фрейме",
            value=objects_in_frame,
            iteration=frame_id
        )
        
        # Логируем количество уникальных треков на текущий момент
        Logger.current_logger().report_scalar(
            title="Трекируемые объекты",
            series="Трекируемые объекты (суммарно)",
            value=len(unique_track_ids),
            iteration=frame_id
        )
        
        # Периодический вывод в консоль для отладки
        if frame_id % 10 == 0:  # Каждые 10 фреймов
            print(f"Фрейм {frame_id}: объектов = {objects_in_frame}, всего уникальных треков = {len(unique_track_ids)}")
    
    # Общая статистика
    total_objects = sum(objects_per_frame)
    total_frames = len(frame_counts)
    avg_objects_per_frame = total_objects / total_frames if total_frames > 0 else 0
    
    task.set_parameter("Всего фреймов", total_frames)
    task.set_parameter("Всего объектов", total_objects)
    task.set_parameter("Уникальных треков", len(unique_track_ids))
    task.set_parameter("В среднем объектов на фрейм", avg_objects_per_frame:.2f)
    
    # Логируем гистограмму объектов на фрейм
    Logger.current_logger().report_histogram(
        title="Распределение объектов",
        series="Количество объектов на фрейме",
        values=objects_per_frame,
        xaxis="Количество объектов",
        yaxis="Количество фреймов"
    )
    
    # Логируем гистограмму уникальных треков
    Logger.current_logger().report_histogram(
        title="Трекирование объектов",
        series="Трекируемые объекты",
        values=unique_tracks_cumulative,
        xaxis="Номер фрейма",
        yaxis="Трекируемые объекты"
    )
    
    # Анализ распределения по категориям
    category_names = []
    category_counts = []
    for cat_id, cat_info in category_distribution.items():
        if cat_info['count'] > 0:
            category_names.append(cat_info['name'])
            category_counts.append(cat_info['count'])
            print(f"Категория {cat_info['name']} (ID: {cat_id}): {cat_info['count']} объектов")
    
    # Логируем распределение по категориям
    if category_names and category_counts:
        Logger.current_logger().report_histogram(
            title="Аннотации - распределение по категориям",
            series="Категории объектов",
            iteration=0,
            xlabels=category_names,
            values=category_counts,
            yaxis="Количество объектов",
            xaxis="Категории"
        )
    
    # Сохраняем итоговую статистику
    task.get_logger().report_single_value("Всего фреймов", total_frames)
    task.get_logger().report_single_value("Всего объектов на всех фреймах", total_objects)
    task.get_logger().report_single_value("Трекируемых объектов", len(unique_track_ids))
    # task.get_logger().report_single_value("В среднем объектов на фрейм", avg_objects_per_frame)
    
    # Статистика по категориям
    for cat_id, cat_info in category_distribution.items():
        if cat_info['count'] > 0:
            task.get_logger().report_single_value(
                f"Объекты категории {cat_info['name']}", 
                cat_info['count']
            )
    
    # Загружаем файл аннотаций как артефакт
    task.upload_artifact("annotations_file", input_file)
    
    print(f"\nСтатистика успешно загружена в ClearML")
    task.close()


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    path_input = current_file.parent.parent / "data" / "input" / "annotations"
    # файлы из папки 
    all_files = os.listdir(path_input)
    # убираем расширение из имен
    fnames = [os.path.splitext(f)[0] for f in all_files if f.endswith('.json')]

    for fname in fnames:
        cml_project_name = "Результаты разметки эксперта"
        cml_task_name = fname
        model_name = "yolo12x"
        input_name = f"{path_input}/{fname}.json"
        time_start = datetime.now()
        
        main(cml_project_name, cml_task_name, input_name )
        
        print(f'\nВремя обработки: {datetime.now() - time_start} сек.')