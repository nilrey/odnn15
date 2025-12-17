import cv2
import numpy as np

def add_gaussian_blur_and_noise(input_video, output_video, blur_kernel=(5,5), noise_intensity=25):
    """
    Добавляет гауссовское размытие и шум к видео

    """
    cap = cv2.VideoCapture(input_video)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"Обработка видео: {total_frames} кадров, {fps} FPS, {width}x{height}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # гауссовское размытие
        blurred_frame = cv2.GaussianBlur(frame, blur_kernel, 0)
        
        # гауссовский шум
        noise = np.random.normal(0, noise_intensity, blurred_frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(blurred_frame, noise)
        
        out.write(noisy_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Обработано: {frame_count}/{total_frames} кадров")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Обработка завершена! Результат сохранен в: {output_video}")

if __name__ == "__main__":
    VIDEO_FILE_NAME = "cars_1"
    NOISE = 0
    BLUR_SIZE = 15
    input_file = f'data/input/{VIDEO_FILE_NAME}.mp4'  
    output_file = f'data/output/noise-gauss-{VIDEO_FILE_NAME}-n{NOISE}-b{BLUR_SIZE}.mp4'
    
    add_gaussian_blur_and_noise(
        input_video=input_file,
        output_video=output_file,
        blur_kernel=(BLUR_SIZE, BLUR_SIZE),  
        noise_intensity=NOISE  
    )