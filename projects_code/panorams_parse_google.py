import json
import csv
import time
import random
from pathlib import Path
from streetlevel import streetview
from tqdm import tqdm

from datetime import datetime

# === Пути ===
DATA_FILE = Path("all_city_coordinates.json")
UPDATED_FILE = Path("panoramas/all_city_coordinates_updated.json")
PANORAMA_DIR = Path("panoramas")
IMAGE_DIR = PANORAMA_DIR / "images"
CSV_PATH = PANORAMA_DIR / "panoramas.csv"
SEEN_IDS_PATH = PANORAMA_DIR / "seen_ids.txt"
LOG_PATH = PANORAMA_DIR / "log.txt"

# === Настройки ===
TARGET_COUNT = 40000  # общее число нужных панорам (можно больше)
DELAY_BASE = 1.5
DELAY_JITTER = 1.0

ZOOM_LEVEL = 2  # или любое значение, которое ты используешь
SOURCE = "google"  # имя сервиса, можно будет менять для Google, Mapillary и др.


# === Создание директорий ===
PANORAMA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)

# === Загрузка координат и фильтрация pending ===
# with open(DATA_FILE, "r", encoding="utf-8") as f:
#     coords = json.load(f)

def log(msg: str, print_also: bool = False):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f_log:
        f_log.write(line + "\n")
    if print_also:
        print(line)

# === Загрузка seen_ids.txt ===
seen_ids = set()
if SEEN_IDS_PATH.exists():
    with open(SEEN_IDS_PATH, "r", encoding="utf-8") as f:
        seen_ids = set(line.strip() for line in f if line.strip())

# === Загрузка координат
if UPDATED_FILE.exists():
    print("📄 Загружаем предыдущий прогресс из all_city_coordinates_updated.json")
    with open(UPDATED_FILE, "r", encoding="utf-8") as f:
        coords = json.load(f)
else:
    print("📄 Загружаем оригинальный all_city_coordinates.json")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        coords = json.load(f)

# === Восстановление статусов из seen_ids
restored = 0
for pt in coords:
    pano_id = pt.get("pano_id")
    if pano_id and pano_id in seen_ids and pt.get("status") != "done":
        pt["status"] = "done"
        restored += 1

if restored > 0:
    print(f"🔁 Восстановлено {restored} координат со статусом 'done' по seen_ids.txt")
# === Фильтрация координат по статусу ===


#pending_coords = [pt for pt in coords if pt["status"] == "pending"]
pending_coords = list(filter(lambda pt: pt.get("status") == "pending", coords))

random.shuffle(pending_coords)

# === Загрузка seen_ids.txt ===
seen_ids = set()
if SEEN_IDS_PATH.exists():
    with open(SEEN_IDS_PATH, "r", encoding="utf-8") as f:
        seen_ids = set(line.strip() for line in f if line.strip())

# === Подготовка CSV ===
csv_exists = CSV_PATH.exists()
f_csv = open(CSV_PATH, "a", newline="", encoding="utf-8")
writer = csv.writer(f_csv)
if not csv_exists:
    writer.writerow(["id", "latitude", "longitude", "city", "date", "zoom", "source","request_lat", "request_lon"])


    f_csv.flush()

# === Основной цикл с прогресс-баром ===
count = len(seen_ids)
progress = tqdm(pending_coords, desc="🔍 Парсинг панорам", unit="точек")

save_every = 1
processed_since_last_save = 0


for point in progress:
    if count >= TARGET_COUNT:
        break

    lat, lon = point["lat"], point["lon"]
    city = point["city"]
    point_id = f"{lat}_{lon}"

    try_attempts = 0
    success = False

    while try_attempts < 2 and not success:
        try:
            pano = streetview.find_panorama(lat, lon)
            if pano and pano.id not in seen_ids:
                image_path = IMAGE_DIR / f"{pano.id}.jpg"
                streetview.download_panorama(pano, str(image_path), zoom=ZOOM_LEVEL)

                writer.writerow([pano.id, pano.lat, pano.lon, city, pano.date, ZOOM_LEVEL, SOURCE, lat, lon])
                f_csv.flush()
                with open(SEEN_IDS_PATH, "a", encoding="utf-8") as f_seen:
                    f_seen.write(pano.id + "\n")

                seen_ids.add(pano.id)
                point["status"] = "done"
                point["pano_id"] = pano.id
                point["zoom"] = ZOOM_LEVEL
                point["source"] = SOURCE
                count += 1
                success = True

                print(f"✅ [{count}] Сохранена панорама {pano.id} — {city} ({pano.lat}, {pano.lon}) — {pano.date}")
                log(
                    f"✅ [{count}] Панорама {pano.id} — {city} "
                    f"(Найдено: {pano.lat:.6f}, {pano.lon:.6f}; Запрошено: {lat:.6f}, {lon:.6f}) — {pano.date}"
                )


                processed_since_last_save += 1
                if processed_since_last_save >= save_every:
                    with open(UPDATED_FILE, "w", encoding="utf-8") as f:
                        json.dump(coords, f, indent=2, ensure_ascii=False)
                    processed_since_last_save = 0


                time.sleep(DELAY_BASE + random.uniform(0, DELAY_JITTER))
            elif pano:
                point["status"] = "duplicate"
                point["source"] = SOURCE
                point["zoom"] = ZOOM_LEVEL
                point["duplicate_of"] = pano.id  # <– только для информации, не для логики
                success = True
                log(f"🔄 Дубликат: pano_id {pano.id} — найден по координате ({lat:.6f}, {lon:.6f}), уже скачан ранее")

            else:
                point["status"] = "no_pano"
                point["zoom"] = ZOOM_LEVEL
                point["source"] = SOURCE
                success = True
                log(f"🚫 Панорама не найдена по координате ({lat:.6f}, {lon:.6f}) — {city}")

                time.sleep(0.5 + random.uniform(0, 0.5))
        except Exception as e:
            try_attempts += 1
            if try_attempts >= 2:
                point["status"] = "error"
                print(f"⚠️ Ошибка в ({lat}, {lon}) после 2 попыток: {e}")
                #log(f"⚠️ Ошибка в ({lat}, {lon}) после 2 попыток: {e}")
                log(f"❌ Ошибка в координате ({lat:.6f}, {lon:.6f}): {type(e).__name__} — {e}")

            else:
                time.sleep(2)

# === Сохраняем обновлённый JSON со статусами ===
with open(UPDATED_FILE, "w", encoding="utf-8") as f:
    json.dump(coords, f, indent=2, ensure_ascii=False)

f_csv.close()
print(f"\n✅ Завершено. Всего скачано панорам: {count}")
log(f"🎯 Завершено. Всего скачано панорам: {count}")

