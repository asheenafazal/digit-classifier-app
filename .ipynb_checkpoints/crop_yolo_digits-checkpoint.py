import os
import cv2

label_folder = 'labels'
image_folder = 'images'
output_folder = 'cropped_digits'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(label_folder):
    if not filename.endswith('.txt'):
        continue

    base_name = filename[:-4]
    image_path = os.path.join(image_folder, base_name + '.jpg')

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    with open(os.path.join(label_folder, filename), 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Invalid format in {filename}: {line}")
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            print(f"⚠️ Cropped image empty: {filename} idx {idx}")
            continue

        out_path = os.path.join(output_folder, f"{base_name}_{idx}.jpg")
        cv2.imwrite(out_path, cropped)
        print(f"✅ Saved: {out_path}")
