import pandas as pd
import time
import random
import csv
import os
from tqdm import tqdm
from download_single import download_around_point
from datetime import datetime

# Параметры
INPUT_CSV = "panoramas.csv"
BASE_DIR = "satelite"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_CSV = os.path.join(BASE_DIR, "satellite_downloads.csv")
DOWNLOADED_IDS_FILE = os.path.join(BASE_DIR, "downloaded_ids.txt")
LOG_FILE = os.path.join(BASE_DIR, "log.txt")
ZOOM_SATELLITE = 20
RADIUS_METERS = 30

def write_log(log_file, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] {message}\n")
    log_file.flush()

# Создание папок
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# Загрузка уже скачанных ID
downloaded_ids = set()
if os.path.exists(DOWNLOADED_IDS_FILE):
    with open(DOWNLOADED_IDS_FILE, "r") as f:
        downloaded_ids = set(line.strip() for line in f)

# Загрузка всех данных
df = pd.read_csv(INPUT_CSV)
#total_to_download = df[~df["id"].isin(downloaded_ids)].shape[0]
total_to_download = df.shape[0]

# Подготовка логов и выходного CSV
log = open(LOG_FILE, "a", encoding="utf-8")
file_exists = os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline='', encoding='utf-8')
csv_writer = csv.DictWriter(csv_file, fieldnames=[
    "id", "pano_id", "latitude", "longitude", "zoom_satelite", "radius_meters", "download_time_sec"
])
if not file_exists:
    csv_writer.writeheader()

# Счётчик успешных загрузок
download_count = len(downloaded_ids)

write_log(log, "🚀 Начало парсинга спутниковых снимков")
# Прогресс-бар
for index, row in tqdm(df.iterrows(), total=len(df), desc="    Downloading"):
    pano_id = row["id"]
    if pano_id in downloaded_ids:
        continue  # пропустить

    lat = row["latitude"]
    lon = row["longitude"]
    filename = f"sat_{pano_id}.png"

    try:
        start_time = time.time()
        success = download_around_point(
            lat=lat,
            lon=lon,
            radius_meters=RADIUS_METERS,
            zoom=ZOOM_SATELLITE,
            save_dir=IMAGE_DIR,
            filename=filename,
            message=False
        )
        elapsed = time.time() - start_time

        if success:
            download_count += 1
            print(f"([{download_count}]✅ Успешно {download_count}/{total_to_download}: {filename} за {elapsed:.2f} сек")
            csv_writer.writerow({
                "id": filename,
                "pano_id": pano_id,
                "latitude": lat,
                "longitude": lon,
                "zoom_satelite": ZOOM_SATELLITE,
                "radius_meters": RADIUS_METERS,
                "download_time_sec": round(elapsed, 2)
            })
            csv_file.flush()
            with open(DOWNLOADED_IDS_FILE, "a", encoding="utf-8") as f:
                f.write(pano_id + "\n")
            #log.write(f"[OK] {filename} | {lat},{lon} | {elapsed:.2f} sec\n")
            write_log(log, f"[OK] {filename} | {lat},{lon} | {elapsed:.2f} sec")

        else:
            print(f"❌ Ошибка {download_count + 1}/{total_to_download}: Не удалось сохранить {filename}")
            #log.write(f"[ERROR] Failed to save {filename} | {lat},{lon}\n")
            write_log(log, f"[ERROR] Failed to save {filename} | {lat},{lon}")

    except Exception as e:
        print(f"❌ Ошибка {download_count + 1}/{total_to_download}: {pano_id} — {str(e)}")
        #log.write(f"[EXCEPTION] {pano_id} | {lat},{lon} | {str(e)}\n")
        write_log(log, f"[EXCEPTION] {pano_id} | {lat},{lon} | {str(e)}")

    log.flush()
    time.sleep(random.uniform(0.5, 1))  # антимаска

log.close()
csv_file.close()
write_log(open(LOG_FILE, "a", encoding="utf-8"), "✅ Парсинг завершён (возможно, не полностью)")
