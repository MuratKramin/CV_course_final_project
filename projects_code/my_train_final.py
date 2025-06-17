import os, sys, time, socket, subprocess
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from data import create_dataset
from models import create_model
from my_options import DummyOpt
from util.visualizer import Visualizer
import multiprocessing

class Logger:
    def __init__(self, path):
        self.terminal, self.log = sys.stdout, open(path, "a", encoding="utf-8")
    def write(self, msg): self.terminal.write(msg); self.log.write(msg)
    def flush(self): self.terminal.flush(); self.log.flush()

def save_opt_params(opt, out_dir, name="opt_params.txt"):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        for attr in dir(opt):
            if not attr.startswith("_") and not callable(getattr(opt, attr)):
                f.write(f"{attr} = {getattr(opt, attr)}\n")

def save_epoch_output(model, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in ['fake_B_final', 'fake_D_final']:
        img = model.get_current_visuals().get(name)
        if img is not None:
            img = (img[0] * 0.5 + 0.5).clamp(0, 1)
            save_image(img, os.path.join(out_dir, f"{epoch:03d}_{name}.png"))

def is_visdom_running(port=8097):
    with socket.socket() as s: return s.connect_ex(("localhost", port)) == 0

if __name__ == "__main__":
    #multiprocessing.freeze_support()

    log_file = f"logs/train_log_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    os.makedirs("logs", exist_ok=True)
    sys.stdout = sys.stderr = Logger(log_file)

    opt = DummyOpt()
    start_epoch = opt.load_iter + 1 if os.path.exists("last_epoch.txt") else opt.epoch_count
    start_epoch = 1
    opt.continue_train = False
    print(f" Стартовая эпоха: {start_epoch}")

    loader = create_dataset(opt)
    dataset = loader.dataset
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                            num_workers=opt.num_threads)

    print(f" Данных: {len(dataset)} | display_freq: {opt.display_freq}")

    model = create_model(opt)
    model.setup(opt)

    if os.path.exists("current_lr.txt"):
        lr = float(open("current_lr.txt").read().strip())
        for i, optz in enumerate(model.optimizers):
            for g in optz.param_groups: g['lr'] = lr
            print(f" LR[{i}]: {lr:.7f}")
    else:
        print(" current_lr.txt не найден")

    if opt.continue_train:
        model.load_optimizers(start_epoch - 1)

    for i, optz in enumerate(model.optimizers):
        lrs = [g['lr'] for g in optz.param_groups]
        print(f" Optimizer[{i}] LR: {lrs}")
        for j, g in enumerate(optz.param_groups):
            print(f"{'✅' if 'initial_lr' in g else '⚠️'} Optimizer[{i}][{j}] initial_lr: {g.get('initial_lr')}")

    for i, sch in enumerate(model.schedulers):
        print(f" Scheduler[{i}] last_epoch: {sch.last_epoch}")

    if not is_visdom_running(opt.display_port):
        print(" Запуск Visdom сервера...")
        subprocess.Popen([sys.executable, "-m", "visdom.server"])
        time.sleep(3)

    visualizer = Visualizer(opt)
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir_root = f"saved_fake_panoramas_{run_id}"
    os.makedirs(save_dir_root, exist_ok=True)
    save_opt_params(opt, save_dir_root)

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        print(f"\n=== Эпоха {epoch} ===")
        epoch_start = time.time()

        for i, data in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch}")):
            model.set_input(data)
            model.optimize_parameters()
            total_iters = epoch * len(dataloader) + i

            if total_iters % opt.save_picture == 0:
                save_dir = os.path.join(save_dir_root, f"epoch_{epoch:03d}")
                os.makedirs(save_dir, exist_ok=True)
                for name in ['fake_B_final', 'fake_D_final']:
                    img = model.get_current_visuals().get(name)
                    if img is not None:
                        img = (img[0].detach().cpu() * 0.5 + 0.5).clamp(0, 1)
                        save_image(img, os.path.join(save_dir, f"{name}_iter_{total_iters:06d}.png"))

            if total_iters % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result=True)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                visualizer.plot_current_losses(epoch, i / len(dataloader), losses)
                print(f"[Epoch {epoch}][Iter {i}] " + " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()]))

        model.update_learning_rate()
        save_epoch_output(model, epoch, f"{save_dir_root}/epoch_outputs")

        os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
        model.save_networks(epoch)
        model.save_optimizers(epoch)
        with open("last_epoch.txt", "w") as f: f.write(str(epoch))
        with open("current_lr.txt", "w") as f: f.write(f"{model.optimizers[0].param_groups[0]['lr']:.7f}")

        print(f" Эпоха {epoch} завершена за {time.time() - epoch_start:.1f} сек.")

        if epoch == opt.niter + opt.niter_decay:
            try: os.remove("last_epoch.txt")
            except FileNotFoundError: pass
