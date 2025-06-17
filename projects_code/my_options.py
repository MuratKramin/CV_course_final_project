import torch
from datetime import datetime
import os

def calc_display_freq(train_size, batch_size, displays_per_epoch=5):
    iters_per_epoch = train_size // batch_size
    display_freq = max(1, iters_per_epoch // displays_per_epoch)
    return display_freq

class DummyOpt:
    def __init__(self):
        # === Общие параметры ===
        self.isTrain = True
        self.no_dropout = False

        #self.init_type = 'normal'
        self.init_type = 'xavier'

        self.init_gain = 0.02

        # === Пути и названия ===
        self.dataroot = "."  # не используется с CSV
        self.dataset_mode = "pano_from_csv"
        self.model = "panogan"
        self.name = "experiment_1"

        self.checkpoints_dir = "./checkpoints"

        # === Устройство ===
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # === Архитектура ===
        self.input_nc = 3
        self.output_nc = 3
        self.netG = "unet_afl_v5"      # генератор с feedback
        self.netD = "fpd"              # Feature Pyramid Discriminator

        self.ngf = 64 
        self.ndf = 64

        self.norm = "instance"
        self.direction = "AtoB"
        self.preprocess = "none"
        self.n_layers_D = 3

        # === Обучение ===
        self.batch_size = 1
        
        self.serial_batches = False
        self.num_threads = 4
        self.max_dataset_size = float("inf")
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lr_policy = "linear"

        # Сколько эпох обучать
        self.niter = 15
        self.niter_decay = 30
        #self.lr_policy = 'constant' 

        self.epoch_count = 1

        # === Потери ===
        self.lambda_L1 = 100.0
        self.lambda_L1_seg = 100.0
        self.gan_mode = "vanilla"
        #self.gan_mode = "lsgan"

        self.pool_size = 0
        #self.pool_size = 50

        # === PanoGAN-specific ===
        self.loop_count = 3
        self.alpha = [0.5, 0.5, 0.5, 0.5, 0.5]

        # === Визуализация / логирование ===
        self.display_id = 1
        self.display_port = 8097
        self.display_ncols = 0

        # Сколько раз во времени обновлять дисплей
        #self.display_freq = 100
        self.display_freq = 500
        #self.display_freq = int(600/self.batch_size)

        self.display_winsize = 1024
        self.display_server = "http://localhost"
        self.display_env = "main"

        # Сколько раз во времени сохранять модель
        self.save_latest_freq = 500

        self.save_epoch_freq = 3

        self.print_freq = 100

        self.update_html_freq = 250
        self.continue_train = False
        self.verbose = True

        # === Размеры изображений (на всякий случай, если где-то используются) ===
        #self.load_size = 352  # уменьшенные размеры
        #self.crop_size = 352
        self.load_size = 256  # уменьшенные размеры
        self.crop_size = 256

        self.no_flip = True   # не флипать по умолчанию

        self.no_html = False

        self.save_picture = 250

        #new params
        #self.no_vgg_loss = False
        #self.no_vgg_loss = True
        self.lambda_feat = 10.0
        #self.no_flip = False

        self.train_size = 10000
        self.display_freq = calc_display_freq(self.train_size, self.batch_size, displays_per_epoch=100)
        self.print_freq = calc_display_freq(self.train_size, self.batch_size, displays_per_epoch=100)

        #self.no_dropout = False
        #self.no_dropout = True

        self.which_epoch = '5'
        self.load_iter = 29