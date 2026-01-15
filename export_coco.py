import json
import os
from datetime import datetime
from typing import Optional

import cv2
from ultralytics import YOLO


MODEL_TAG = "yolo12x"
INPUT_FILE_TAG = "spb_zagorodny_proezd_001"
INPUT_FILENAME = f"{INPUT_FILE_TAG}.mp4"
CONFIDENCE = 0.5
DATE_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_FILE_PATH = f"models/{MODEL_TAG}.pt"
ANNOT_FILENAME = f"annotation_{INPUT_FILE_TAG}_{DATE_STAMP}.json"
INPUT_VIDEO_FILE_PATH = f"data/input/{INPUT_FILENAME}"
# OUTPUT_FILENAME = f"out-{INPUT_FILE_TAG}-{MODEL_TAG}-conf-{CONFIDENCE}_{DATE_STAMP}"
# OUTPUT_VIDEO_FILE_PATH = f"data/output/{OUTPUT_FILENAME}.mp4"

def _category_id(yolo_class_id: int) -> Optional[int]:
    """
    Приводим классы YOLO к категориям COCO (car=1, person=2).
    Если класс не из поддерживаемых, вернем None.
    """
    if yolo_class_id == 0:  # person
        return 2
    if yolo_class_id in {2, 3, 5, 6, 7, 8}:  # автомобили и крупный транспорт
        return 1
    return None


def main():
    model = YOLO(MODEL_FILE_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO_FILE_PATH)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {INPUT_FILENAME}")

    export_json_path = os.path.join("data", "output", ANNOT_FILENAME)
    os.makedirs("data/output", exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    color_yellow = (0, 255, 255)

    allowed_indices = {0, 2, 3, 5, 6, 7, 8}  # Фильтрация классов автомобилей

    # out = cv2.VideoWriter(
    #     OUTPUT_VIDEO_FILE_PATH,
    #     cv2.VideoWriter_fourcc(*"mp4v"),
    #     fps,
    #     (frame_width, frame_height),
    # )

    # Заготовка COCO (копируем шапку из instances_default.json)
    coco_data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {"id": 1, "name": "car", "supercategory": ""},
            {"id": 2, "name": "person", "supercategory": ""},
        ],
        "images": [],
        "annotations": [],
    }

    frame_id = 0
    annotation_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_id = frame_id + 1
        file_name = f"frame_{frame_id:06d}.png"

        # Добавляем запись о кадре в images
        coco_data["images"].append(
            {
                "id": image_id,
                "width": frame_width,
                "height": frame_height,
                "file_name": file_name,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        # Печатаем прогресс по кадрам (номер/всего)
        print(f"{frame_id + 1}/{total_frames}: processing")

        # Используем модель для анализа текущего кадра с отслеживанием
        results = model.track(frame, persist=True, iou=0.4, verbose=False)

        if results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                yolo_cls = int(box.cls[0])
                if yolo_cls in allowed_indices and conf > CONFIDENCE:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    category_id = _category_id(yolo_cls)
                    if category_id is None:
                        continue

                    track_id = int(results[0].boxes.id[i])

                    coco_data["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [],
                            "area": area,
                            "bbox": [x1, y1, width, height],
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                "rotation": 0.0,
                                "track_id": track_id,
                                "keyframe": True,
                            },
                        }
                    )
                    annotation_id += 1

                    # # Рисуем bounding box и ID на кадре
                    # x1_int, y1_int, x2_int, y2_int = map(int, (x1, y1, x2, y2))
                    # label = f"{results[0].names[yolo_cls]} {track_id}"
                    # cv2.rectangle(frame, (x1_int, y1_int), (x2_int, y2_int), color_yellow, 1)
                    # cv2.putText(
                    #     frame,
                    #     label,
                    #     (x1_int, y1_int - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.5,
                    #     color_yellow,
                    #     1,
                    # )

        # Запись обработанного кадра в выходное видео
        # out.write(frame)
        frame_id += 1

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    # Сохраняем COCO-аннотации
    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=4)

    # print(f"Обработанное видео с трекингом сохранено в {OUTPUT_VIDEO_FILE_PATH}")
    print(f"COCO-аннотации сохранены в {export_json_path}")


if __name__ == "__main__":
    time_start = datetime.now()
    main()
    print(f"Время работы: {datetime.now() - time_start} сек.")