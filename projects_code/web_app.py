import streamlit as st

if "generated_pano" not in st.session_state:
    st.session_state.generated_pano = None

from streamlit_folium import st_folium
import folium
import os
from PIL import Image
import torch
from torchvision import transforms
from my_options import DummyOpt
from models import create_model
from data.base_dataset import get_transform
import tempfile


import sys
import os

# === Добавляем путь к download_single.py
DOWNLOAD_UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'data_download', 'satelite'))
sys.path.append(DOWNLOAD_UTILS_PATH)

from download_single import download_around_point


# === Настройка модели один раз ===
@st.cache_resource
def load_model():
    opt = DummyOpt()
    opt.isTrain = False
    opt.serial_batches = True
    opt.batch_size = 1
    opt.num_threads = 0
    opt.phase = "test"
    opt.epoch = "iter_40"
    opt.name = "panogan_experiment"
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model, opt

model, opt = load_model()

# === UI ===
st.title("🌍 Генерация панорамы по координатам")
st.markdown("Выберите точку на карте. Система загрузит спутниковое изображение и сгенерирует панораму.")

# === Карта ===
# === Карта с поддержкой кликов ===
m = folium.Map(location=[55.751244, 37.618423], zoom_start=5)
folium.LatLngPopup().add_to(m)
map_data = st_folium(m, width=700, height=500)
if st.session_state.generated_pano is not None:
    st.image(st.session_state.generated_pano, caption="Сгенерированная панорама", use_column_width=True)



# === Получение координат
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.write(f"Вы выбрали точку: ({lat:.6f}, {lon:.6f})")

    if st.button("🔮 Сгенерировать панораму"):
        with st.spinner("Загружаем спутниковое изображение..."):
            # === Загружаем и сохраняем спутник (заглушка)
            #from ../data/data_download_single import download_around_point
            temp_dir = tempfile.mkdtemp()
            sat_path = os.path.join(temp_dir, "input.png")
            download_around_point(lat, lon, radius_meters=30, zoom=20, save_dir=temp_dir, filename="input.png")

        with st.spinner("Генерация изображения..."):
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            img = Image.open(sat_path).convert("RGB")
            A = transform(img)
            A_rot = [A] + [transforms.functional.rotate(A, 90 * i) for i in range(1, 4)]
            A_concat = torch.cat(A_rot, dim=2).unsqueeze(0).to(opt.device)

            # Подготовка input
            data = {'A': A_concat, 'B': A_concat, 'D': A_concat, 'A_paths': "user_input", 'B_paths': "user_input"}
            model.set_input(data)
            model.test()
            model.compute_visuals()
            fake_pano = model.get_current_visuals()["fake_B_final"][0]
            pano_img = (fake_pano.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
            pano_img = transforms.ToPILImage()(pano_img)

            #st.image(pano_img, caption="Сгенерированная панорама", use_column_width=True)
            st.session_state.generated_pano = pano_img
            st.image(pano_img, caption="Сгенерированная панорама", use_column_width=True)
            st.success("✅ Панорама успешно сгенерирована!")


