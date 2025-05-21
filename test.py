import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
 
cap = cv2.VideoCapture("bus1.mp4")
success, frame = cap.read()
count = 0
 
while success:
    cv2.imwrite(f"frame_{count}.jpg", frame)
    success, frame = cap.read()
    count += 1

processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to("cuda")
 
image = Image.open("frame_0.jpg").convert("RGB")
prompt = "Describe the objects in this image and identify any buses and their license plate numbers."
 
inputs = processor(image, text=prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(out[0], skip_special_tokens=True))