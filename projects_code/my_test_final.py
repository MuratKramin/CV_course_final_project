import os, sys, time
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from PIL import Image
import numpy as np

from data import create_dataset
from models import create_model
from my_options import DummyOpt

class Logger:
    def __init__(self, path):
        self.terminal, self.log = sys.stdout, open(path, "a", encoding="utf-8")
    def write(self, msg): self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()

def save_tensor_image(tensor, path):
    img = (tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)  # денормализация
    save_image(img, path)

if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/test_log_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    sys.stdout = sys.stderr = Logger(log_file)

    opt = DummyOpt()
    opt.isTrain = False
    opt.serial_batches = True
    opt.batch_size = 1
    opt.num_threads = 0
    opt.name = "experiment"
    opt.phase = "test"
    opt.epoch = "latest"
    opt.num_test = 1000  # Количество тестовых изображений
    #opt.device = torch.device("cpu")

    print(" Запуск тестирования модели...")

    dataset = create_dataset(opt).dataset
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    result_root = f"test_results_{opt.name}_{opt.num_test}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    real_dir = os.path.join(result_root, "real")
    synth_dir = os.path.join(result_root, "synth")
    real_seg_dir = os.path.join(result_root, "real_seg")
    synth_seg_dir = os.path.join(result_root, "synth_seg")

    for d in [real_dir, synth_dir, real_seg_dir, synth_seg_dir]:
        os.makedirs(d, exist_ok=True)

    for i, data in enumerate(tqdm(dataloader, desc=" Тестирование")):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()
        model.compute_visuals()
        visuals = model.get_current_visuals()
        fname_img = f"{i+1:04d}.jpg"
        fname_seg = f"{i+1:04d}.png"

        # Сохраняем реальные и синтезированные изображения
        if "real_B" in visuals:
            save_tensor_image(visuals["real_B"][0], os.path.join(real_dir, fname_img))
        if "fake_B_final" in visuals:
            save_tensor_image(visuals["fake_B_final"][0], os.path.join(synth_dir, fname_img))
        # Сохраняем сегментации в цвете
        if "real_D" in visuals:
            save_tensor_image(visuals["real_D"][0], os.path.join(real_seg_dir, fname_seg))
        if "fake_D_final" in visuals:
            save_tensor_image(visuals["fake_D_final"][0], os.path.join(synth_seg_dir, fname_seg))


    print(f"\n Сохранено {i+1} изображений в:\n {real_dir}\n {synth_dir}\n {real_seg_dir}\n {synth_seg_dir}")
