import cv2
from ultralytics import YOLO
import json
import os

PATH_MODELS = "models/"


class YOLOv8PersonCarDetector:
    def __init__(self, model_path=f'{PATH_MODELS}yolov8n.pt'):
        """Детектор для людей и машин"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель {model_path} не найдена")
        
        self.model = YOLO(model_path)
        self.results = []
        
        # Mapping классов COCO dataset
        self.class_names = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
    
    def process_video(self, video_path, output_json='detections.json', conf_threshold=0.5):
        """Обработка видео с детекцией людей и машин"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Детекция: люди + машины")
        frame_count = 0
        self.results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # ВАЖНО: classes=[0, 2] - люди (0) и машины (2)
            results = self.model(frame, conf=conf_threshold, classes=[0, 2])
            
            frame_detections = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'detections': []
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Получаем название класса
                        class_name = self.class_names.get(class_id, f'class_{class_id}')
                        
                        detection = {
                            'bbox': {
                                'x1': float(x1),
                                'y1': float(y1),
                                'x2': float(x2), 
                                'y2': float(y2),
                                'width': float(x2 - x1),
                                'height': float(y2 - y1)
                            },
                            'confidence': float(confidence),
                            'class': class_name,  # 'person' или 'car'
                            'class_id': class_id
                        }
                        frame_detections['detections'].append(detection)
            
            self.results.append(frame_detections)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Обработано: {frame_count}/{total_frames}")
        
        cap.release()
        self.save_results(output_json)
        print(f"Обработка завершена! Файл: {output_json}")
    
    def save_results(self, output_path):
        """Сохранение результатов"""
        output_data = {
            'detected_classes': ['person', 'car'],
            'detections': self.results,
            'statistics': self.get_detection_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def get_detection_statistics(self):
        """Статистика по детекциям"""
        stats = {
            'total_frames': len(self.results),
            'person_detections': 0,
            'car_detections': 0,
            'frames_with_person': 0,
            'frames_with_car': 0
        }
        
        for frame in self.results:
            has_person = False
            has_car = False
            
            for detection in frame['detections']:
                if detection['class'] == 'person':
                    stats['person_detections'] += 1
                    has_person = True
                elif detection['class'] == 'car':
                    stats['car_detections'] += 1
                    has_car = True
            
            if has_person:
                stats['frames_with_person'] += 1
            if has_car:
                stats['frames_with_car'] += 1
        
        return stats

# Пример использования
def main():
    detector = YOLOv8PersonCarDetector('models/yolov8n.pt')
    
    detector.process_video(
        video_path='data/input/input_video.mp4',
        output_json='data/output/person_car_detections.json',
        conf_threshold=0.5
    )
    
    # Вывод статистики
    stats = detector.get_detection_statistics()
    print(f"\n=== Статистика ===")
    print(f"Всего кадров: {stats['total_frames']}")
    print(f"Детекций людей: {stats['person_detections']}")
    print(f"Детекций машин: {stats['car_detections']}")
    print(f"Кадров с людьми: {stats['frames_with_person']}")
    print(f"Кадров с машинами: {stats['frames_with_car']}")

if __name__ == "__main__":
    main()