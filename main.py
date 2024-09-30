import torch
from PIL import Image
import numpy as np
import os

def add_noise_to_image(image, mean, stddev):
    image_array = np.array(image)
    noise = np.random.normal(mean, stddev, image_array.shape)
    noisy_image_array = image_array + noise
    noisy_image_array = np.clip(noisy_image_array, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image_array)
    return noisy_image

def main(image_path, std):
    image = Image.open(image_path).convert("RGB")
    noisy_image = add_noise_to_image(image, mean=0, stddev=std)
    
    os.makedirs('noise_images', exist_ok=True) 
    
    imgname = os.path.join('noise_images','noise_' + str(std) + '.png')
    noisy_image.save(imgname, format='PNG')
    print(f"Noisy image saved as {imgname}")
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    results = model([imgname]) 
    os.makedirs('predictions', exist_ok=True) 
    save_dir = os.path.join('predictions/noise' + str(std))
    
    results.print()
    results.save(save_dir=save_dir)
    
    df_results = results.pandas().xyxy[0]
    
    confidence_scores = df_results['confidence'].tolist()
    
    if confidence_scores:
        print("Confidence Score:")
        for score in confidence_scores:
            print(f"{score:.2f}")
        average_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"Average Confidence: {average_confidence:.2f}")
    else:
        print("No objects detected.")

if __name__ == "__main__":
    image_path = "bmw.png"
    stddev=0
    main(image_path, stddev)
