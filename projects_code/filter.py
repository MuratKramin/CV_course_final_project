import pandas as pd

# === Шаг 1: Загрузка CSV
df_masks = pd.read_csv('masks_info.csv')
df_panos = pd.read_csv('panoramas_with_size.csv')
df_sats = pd.read_csv('satellite_downloads_with_size.csv')

# Всего строк в файлах: 23217

# === Шаг 2: Объединение разрешений
df = df_masks.merge(df_panos[['id', 'width', 'height']], left_on='pano_id', right_on='id', how='inner')

# === Шаг 3: Вычисляем общую площадь и долю background
df['area_background'] = df['area_0_background']
df['total_area'] = df[[col for col in df.columns if col.startswith('area_')]].sum(axis=1)
df['background_ratio'] = df['area_background'] / df['total_area']

# === Шаг 4: Применяем фильтры
filtered = df[
    (df['class_1_sky'] == 1) &
    (df['width'] == 3584) &
    (df['height'].between(1100, 1400)) &
    (df['background_ratio'] < 0.01)
].copy()

# === Шаг 5: Подготовка итоговой таблицы
filtered['id'] = filtered['pano_id']
filtered['pano_file'] = filtered['pano_id'] + '.jpg'
filtered['sat_file'] = 'sat_' + filtered['pano_id'] + '.png'
filtered['mask_file'] = 'mask_' + filtered['pano_id'] + '.png'

# === Шаг 6: Оставляем только нужные колонки
result = filtered[['id', 'pano_file', 'sat_file', 'mask_file']]

# === Шаг 7: Сохранение результата
result.to_csv('filtered_ids.csv', index=False)

# === Шаг 8: Вывод
print(f"Итог: найдено {len(result)} подходящих записей.")
# Итого отфильтровано -  18474 подходящих записей.
