import os
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# === Папки
SAVE_DIR = "segmentation_masks"
MASK_IMG_DIR = os.path.join(SAVE_DIR, "images")
MASK_NPY_DIR = os.path.join(SAVE_DIR, "npy")
CSV_PATH = os.path.join(SAVE_DIR, "masks_info.csv")

# === Ремапинг: ADE20K class_id → твой новый класс (1–9)
remap_dict = {
    2: 1,
    6: 2, 11: 2, 83: 2, 116: 2, 20: 2, 102: 2, 80: 2, 40: 2, 12: 2, 138: 2, 69: 2, 127: 2,
    1: 3, 122: 3, 41: 3,
    4: 4, 72: 4,
    9: 5, 13: 5, 29: 5, 46: 5, 52: 5, 94: 5, 121: 5, 91: 5, 34: 5,
    17: 6, 66: 6,
    0: 7, 32: 7, 38: 7, 140: 7,
    87: 8, 93: 8, 43: 8, 149: 8, 136: 8, 42: 8, 95: 8,
    21: 9, 76: 9, 26: 9, 128: 9, 60: 9, 113: 9,
}

# === Новый словарь классов
new_labels = {
    0: "background",
    1: "sky",
    2: "road",
    3: "buildings",
    4: "trees",
    5: "ground",
    6: "plants",
    7: "walls",
    8: "vertical_obj",
    9: "water_obj"
}

def get_color_map():
    return np.array([
        [0, 0, 0],
        [135, 206, 235],
        [128, 64, 128],
        [70, 70, 70],
        [34, 139, 34],
        [160, 82, 45],
        [107, 142, 35],
        [190, 153, 153],
        [0, 0, 128],
        [0, 191, 255],
    ], dtype=np.uint8)

def save_segmentation_masks(folder_path, max_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device)
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    model.eval()

    os.makedirs(MASK_IMG_DIR, exist_ok=True)
    os.makedirs(MASK_NPY_DIR, exist_ok=True)

    # === Уже обработанные маски
    existing_masks = {
        os.path.splitext(f)[0].replace("mask_", "")
        for f in os.listdir(MASK_IMG_DIR) if f.endswith(".png")
    }

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    image_files = [f for f in image_files if os.path.splitext(f)[0] not in existing_masks]
    image_files = random.sample(image_files, min(max_images, len(image_files)))

    color_map = get_color_map()
    TARGET_SIZE = (3584, 1280)

    for image_file in tqdm(image_files, desc="📷 Генерация масок"):
        pano_id = os.path.splitext(image_file)[0]
        mask_id = f"mask_{pano_id}"

        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted = outputs.logits.argmax(dim=1)[0].cpu().numpy()

        # === Перемаппинг классов
        remapped_mask = np.zeros_like(predicted, dtype=np.uint8)
        for old_id, new_id in remap_dict.items():
            remapped_mask[predicted == old_id] = new_id

        # === Ресайз маски
        remapped_mask_resized = Image.fromarray(remapped_mask).resize(TARGET_SIZE, resample=Image.NEAREST)
        remapped_mask_resized_np = np.array(remapped_mask_resized, dtype=np.uint8)

        # === Сохранение PNG-маски
        color_mask = color_map[remapped_mask_resized_np]
        rgb_mask = Image.fromarray(color_mask)
        rgb_mask.save(os.path.join(MASK_IMG_DIR, f"{mask_id}.png"))

        # === Сохранение .npy маски
        np.savez_compressed(os.path.join(MASK_NPY_DIR, f"{mask_id}.npz"), mask=remapped_mask_resized_np)


        # === Анализ классов
        flat_mask = remapped_mask_resized_np.flatten()
        class_ids_present = np.unique(remapped_mask_resized_np)

        row = {
            "id": mask_id,
            "pano_id": pano_id,
            "n_classes": len(class_ids_present)
        }

        for class_id, class_name in new_labels.items():
            present = int(class_id in class_ids_present)
            area = int(np.sum(flat_mask == class_id))
            row[f"class_{class_id}_{class_name}"] = present
            row[f"area_{class_id}_{class_name}"] = area

        # === Запись CSV сразу
        row_df = pd.DataFrame([row])
        if os.path.exists(CSV_PATH):
            row_df.to_csv(CSV_PATH, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            row_df.to_csv(CSV_PATH, mode='w', header=True, index=False, encoding='utf-8-sig')

    print(f"\n✅ Завершено. Маски PNG: {MASK_IMG_DIR}")
    print(f"💾 Маски NPY: {MASK_NPY_DIR}")
    print(f"📄 CSV: {CSV_PATH}")

# === Пример запуска
save_segmentation_masks("E:\\Projects\\streetlevel_new_xmp\\panoramas\\images", 25000)
