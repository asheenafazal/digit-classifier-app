import os
import cv2

# Paths
images_folder = 'images'
labels_folder = 'labels'
output_base_folder = 'prepared_digits'

# Ensure output folders 1–9 then 0
digit_folders = [str(i) for i in range(1, 10)] + ['0']
for digit in digit_folders:
    os.makedirs(os.path.join(output_base_folder, digit), exist_ok=True)

# Helper to keep count of images per digit
digit_counts = {digit: 0 for digit in digit_folders}

# Loop through label files
for label_file in os.listdir(labels_folder):
    if not label_file.endswith('.txt'):
        continue

    image_name = os.path.splitext(label_file)[0] + '.jpeg'
    image_path = os.path.join(images_folder, image_name)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Read YOLO label file
    with open(os.path.join(labels_folder, label_file), 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"⚠️ Invalid label in {label_file}: {line}")
            continue

        cls, x_center, y_center, width, height = map(float, parts)
        cls = int(cls)

        # Convert YOLO to pixel coordinates
        x = int((x_center - width / 2) * w)
        y = int((y_center - height / 2) * h)
        w_box = int(width * w)
        h_box = int(height * h)

        cropped = img[y:y + h_box, x:x + w_box]

        cls_str = str(cls)
        if cls_str not in digit_folders:
            print(f"⚠️ Unknown class {cls} in {label_file}")
            continue

        count = digit_counts[cls_str]
        output_path = os.path.join(output_base_folder, cls_str, f"img_{cls_str}_{count}.jpg")
        cv2.imwrite(output_path, cropped)
        digit_counts[cls_str] += 1

print("\n✅ All digit images cropped and saved.")
