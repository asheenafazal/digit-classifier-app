import os
import cv2

image_folder = "images"
label_folder = "labels"
output_folder = "cropped_digits"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for label_file in os.listdir(label_folder):
    if not label_file.endswith(".txt"):
        continue

    image_name = label_file.replace(".txt", ".jpg")
    image_path = os.path.join(image_folder, image_name)
    label_path = os.path.join(label_folder, label_file)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_path, "r") as file:
        for idx, line in enumerate(file):
            class_id, x_center, y_center, w, h = map(float, line.strip().split())
            x = int((x_center - w / 2) * width)
            y = int((y_center - h / 2) * height)
            w = int(w * width)
            h = int(h * height)
            cropped = image[y:y+h, x:x+w]
            if cropped.size == 0:
                continue
            output_path = os.path.join(output_folder, f"{label_file[:-4]}_{idx}.jpg")
            cv2.imwrite(output_path, cropped)

print("✅ Cropping complete. Check the 'cropped_digits' folder.")
