import os
import shutil
import random

SOURCE_DIR = 'prepared_digits'
TARGET_DIR = 'custom_mnist'
SPLIT_RATIO = 0.8  # 80% train, 20% test

os.makedirs(os.path.join(TARGET_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(TARGET_DIR, 'test'), exist_ok=True)

for digit in sorted(os.listdir(SOURCE_DIR), key=lambda x: int(x) if x != "0" else 10):
    source_path = os.path.join(SOURCE_DIR, digit)
    images = os.listdir(source_path)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    os.makedirs(os.path.join(TARGET_DIR, 'train', digit), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'test', digit), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(source_path, img), os.path.join(TARGET_DIR, 'train', digit, img))
    for img in test_imgs:
        shutil.copy(os.path.join(source_path, img), os.path.join(TARGET_DIR, 'test', digit, img))

print("✅ Dataset split into train/test successfully.")
