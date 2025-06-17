import torch
from .base_model import BaseModel
from . import networks_old

class panoganModel(BaseModel):

    """
    Этот класс реализует модель, которая использует генеративно-состязательные сети (GAN) для генерации панорамных изображений 
    на основе спутниковых снимков и сегментационных масок.
    Он наследуется от BaseModel и реализует методы для настройки параметров командной строки,
    инициализации модели, установки входных данных, прямого прохода, обратного прохода для дискриминатора и генератора,
    а также оптимизации параметров модели.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='aligned4')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1_seg', type=float, default=100.0, help='weight for L1 loss seg')
            #parser.add_argument('--loop_count', type=int, default=2, help='# feedback loop')
            #parser.add_argument('--epoch_count_afl', type=int, default=20, help='# feedback loop')
            #parser.add_argument('--afl_count', type=int, default=10, help='# feedback loop')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # Укажите потери обучения, которые вы хотите выводить. Скрипты обучения/тестирования будут вызывать <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D', 'GAN_img', 'GAN_seg', 'L1_img', 'L1_seg', 'D_real_img', 'D_fake_img', 'D_real_seg', 'D_fake_seg']
        # Укажите изображения, которые вы хотите сохранять/отображать. 
        # Скрипты обучения/тестирования будут вызывать <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['img_A', 'img_B', 'fake_B_final', 'img_D', 'fake_D_final']
        else:  # during test time, only load G
            #self.visual_names = ['fake_B_final']
            self.visual_names = ['fake_B_final', 'real_B', 'fake_D_final', 'real_D', 'input_A_vis'] 


        # Количество итераций обратной связи
        self.loop_count = opt.loop_count
        self.alpha = opt.alpha

        # Укажите модели, которые вы хотите сохранять на диск. Скрипты обучения/тестирования будут вызывать <BaseModel.save_networks> и <BaseModel.load_networks>
        # В данном случае мы сохраняем генератор и дискриминаторы для изображений и сегментаций.
        self.model_names = ['G', 'D_img', 'D_seg'] #F_img/F_seg: Feedback network w.r.t. image / segmentation

        # Определяем сети (как генератор, так и дискриминаторы)
        self.netG = networks_old.define_G(opt.input_nc, opt.output_nc * 2, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.alpha, opt.loop_count, opt.ndf)
        self.netD_img = networks_old.define_D(2 * opt.input_nc, opt.ndf, opt.netD,
                                       opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.netD_seg = networks_old.define_D(2 * opt.input_nc, opt.ndf, opt.netD,
                                       opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # Определяем функции потерь для GAN и L1.
            self.criterionGAN = networks_old.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # инициализируем оптимизаторы; планировщики будут автоматически созданы функцией <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_img = torch.optim.Adam(self.netD_img.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_seg = torch.optim.Adam(self.netD_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F_img = torch.optim.Adam(self.netF_img.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_F_seg = torch.optim.Adam(self.netF_seg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_img)
            self.optimizers.append(self.optimizer_D_seg)
            # self.optimizers.append(self.optimizer_F_img)
            # self.optimizers.append(self.optimizer_F_seg)

            self.iteration_counter = 0


    def compute_visuals(self):
        """ В этом методе мы генерируем финальные изображения для визуализации."""
        self.fake_D_final = self.fake_D[-1]   # последняя сгенерированная сегментация
        self.real_D = self.img_D              # ground truth сегментация
        self.real_B = self.img_B              # ground truth панорама

        self.input_A_vis = self.img_A # входное спутниковое изображение


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.img_A = input['A' if AtoB else 'B'].to(self.device)
        self.img_B = input['B' if AtoB else 'A'].to(self.device)
        self.img_D = input['D'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_label = 0.9
        self.false_label = 0.0

    def set_loop_count(self, loop_count):
        self.netG.module.loop_count = loop_count

    def forward(self):
        """ В этом методе мы выполняем прямой проход через генератор и дискриминаторы."""

        self.fake_B = []
        self.fake_D = []

        for i in range(0, self.loop_count):
            if i == 0:
                gene_in = self.img_A
                disc_out = None
                alpha = None
            else:
                if isinstance(fea_inner, list):
                    gene_in = fea_inner
                else:
                    gene_in[0] = fea_inner
                disc_out = disc_out_img
                alpha = self.alpha

            # print("========loop_{}============".format(i))
            fake_B, fake_D, fea_inner= self.netG(gene_input=gene_in, disc_out=disc_out, alpha=alpha)
            self.fake_B.extend([fake_B])
            self.fake_D.extend([fake_D])

            #---------img loss ------------------------
            # fake
            fake_AB = torch.cat((self.img_A, fake_B), 1)  # мы используем условные GAN; нам нужно подать как вход, так и выход в дискриминатор
            _, _, disc_out_img = self.netD_img(fake_AB)  
            # Фейковое изображение (fake_AB) конкатенируется с img_A и подается в дискриминатор изображений (netD_img).

            # ---------seg loss ------------------------
            # fake
            fake_AD = torch.cat((self.img_A, fake_D), 1)  

            # Фейковое сегментированное изображение (fake_AD) конкатенируется с img_A и подается в дискриминатор сегментации (netD_seg).
            _, _, disc_out_seg = self.netD_seg(fake_AD)  
            # Фейковое сегментированное изображение (fake_AD) конкатенируется с img_A и подается в дискриминатор сегментации (netD_seg).

            for j in range(len(disc_out_img)):
                disc_out_img[j] = torch.cat((disc_out_img[j], disc_out_seg[j]), 1)

        self.fake_B_final = self.fake_B[-1]
        self.fake_D_final = self.fake_D[-1]

    def backward_D(self):
        self.loss_D_fake_img = 0
        self.loss_D_real_img = 0
        self.loss_D_fake_seg = 0
        self.loss_D_real_seg = 0

        for j in range(self.loop_count):
        # ---------D loss ------------------------
            # fake image
            fake_AB = torch.cat((self.img_A, self.fake_B[j]),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake_img, embed_fake_img, _ = self.netD_img(
                fake_AB.detach())  # Fake; stop backprop to the generator by detaching fake_B
            # real image
            real_AB = torch.cat((self.img_A, self.img_B), 1)
            pred_real_img, embed_real_img, _ = self.netD_img(real_AB)
            # fake seg
            fake_AD = torch.cat((self.img_A, self.fake_D[j]),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake_seg, embed_fake_seg, _ = self.netD_seg(
                fake_AD.detach())  # Fake; stop backprop to the generator by detaching fake_B
            # real seg
            real_AD = torch.cat((self.img_A, self.img_D), 1)
            pred_real_seg, embed_real_seg, _ = self.netD_seg(real_AD)
            # Слияние предсказанных карт: мы объединяем предсказания дискриминаторов изображений и сегментаций
            pred_embed_fake = [pred_fake_img, embed_fake_img, pred_fake_seg, embed_fake_seg]
            pred_embed_real = [pred_real_img, embed_real_img, pred_real_seg, embed_real_seg]
            pred_fake_img, pred_fake_seg, \
            pred_real_img, pred_real_seg = self.confuse_pred_embed(pred_embed_fake, pred_embed_real, True)
            # Вычисляем потери дискриминатора для фейковых и реальных изображений и сегментаций
            num_pred = len(pred_fake_img)
            for i in range(num_pred):
                self.loss_D_fake_img += (self.criterionGAN(pred_fake_img[i], self.false_label) / num_pred)
                self.loss_D_real_img += (self.criterionGAN(pred_real_img[i], self.real_label) / num_pred)
                self.loss_D_fake_seg += (self.criterionGAN(pred_fake_seg[i], self.false_label) / num_pred)
                self.loss_D_real_seg += (self.criterionGAN(pred_real_seg[i], self.real_label) / num_pred)

        # Общая потеря дискриминатора для всех итераций обратной связи
        self.loss_D = (self.loss_D_fake_img + self.loss_D_real_img +
                           self.loss_D_fake_seg + self.loss_D_real_seg) / self.loop_count * 0.5

        self.loss_D.backward()

    def backward_G(self):
        """ В этом методе мы вычисляем потери для генератора, включая потери GAN и L1."""
        # GAN loss
        self.loss_GAN_img = 0
        self.loss_GAN_seg = 0
        self.loss_L1_img = 0
        self.loss_L1_seg = 0

        for j in range(self.loop_count):
            #--------L1 loss--------------------------
            self.loss_L1_img += self.criterionL1(self.fake_B[j], self.img_B) * self.opt.lambda_L1
            self.loss_L1_seg += self.criterionL1(self.fake_D[j], self.img_D) * self.opt.lambda_L1_seg

            # ---------GAN loss ------------------------
            # fake img
            fake_AB = torch.cat((self.img_A, self.fake_B[j]),
                                1)  # мы используем условные GAN; нам нужно подать как вход, так и выход в дискриминатор
            pred_fake_img, embed_fake_img, _ = self.netD_img(
                fake_AB)  #  Фейковое изображение (fake_AB) конкатенируется с img_A и подается в дискриминатор изображений (netD_img).
            # real image
            real_AB = torch.cat((self.img_A, self.img_B), 1)
            pred_real_img, embed_real_img, _ = self.netD_img(real_AB)
            # fake seg
            fake_AD = torch.cat((self.img_A, self.fake_D[j]),
                                1)  # мы используем условные GAN; нам нужно подать как вход, так и выход в дискриминатор
            pred_fake_seg, embed_fake_seg, _ = self.netD_seg(
                fake_AD)  # Фейковое сегментированное изображение (fake_AD) конкатенируется с img_A и подается в дискриминатор сегментации (netD_seg).
            # real seg
            real_AD = torch.cat((self.img_A, self.img_D), 1)
            pred_real_seg, embed_real_seg, _ = self.netD_seg(real_AD)
            #fuse predict map
            pred_embed_fake = [pred_fake_img, embed_fake_img, pred_fake_seg, embed_fake_seg]
            pred_embed_real = [pred_real_img, embed_real_img, pred_real_seg, embed_real_seg]
            pred_fake_img, pred_fake_seg = self.confuse_pred_embed(pred_embed_fake, pred_embed_real, False)
            # compute GAN loss
            num_pred = len(pred_fake_img)
            for i in range(num_pred):
                self.loss_GAN_img += (self.criterionGAN(pred_fake_img[i], self.real_label) / num_pred)
                self.loss_GAN_seg += (self.criterionGAN(pred_fake_seg[i], self.real_label) / num_pred)

        self.loss_GAN = self.loss_GAN_img + self.loss_GAN_seg
        self.loss_L1 = self.loss_L1_img + self.loss_L1_seg
        self.loss_G = (self.loss_GAN + self.loss_L1) / self.loop_count

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # вычисляем фейковые изображения: G(A)
        # update D
        # self.set_requires_grad(self.netD_img, True)  # enable backprop for D
        # self.set_requires_grad(self.netD_seg, True)  # enable backprop for D
        # self.optimizer_D_img.zero_grad()  # set D's gradients to zero
        # self.optimizer_D_seg.zero_grad()  # set D's gradients to zero
        # self.backward_D()  # calculate gradients for D
        # self.optimizer_D_img.step()  # update D's weights
        # self.optimizer_D_seg.step()  # update D's weights
        # === Шаг итерации
        self.iteration_counter += 1

        # === Обновляем D только раз в N итераций
        UPDATE_D_EVERY = 1
        if self.iteration_counter % UPDATE_D_EVERY == 0:
            self.set_requires_grad(self.netD_img, True)
            self.set_requires_grad(self.netD_seg, True)
            self.optimizer_D_img.zero_grad()
            self.optimizer_D_seg.zero_grad()
            self.backward_D()
            self.optimizer_D_img.step()
            self.optimizer_D_seg.step()
        else:
            # Не обновляем дискриминатор — замораживаем
            self.set_requires_grad(self.netD_img, False)
            self.set_requires_grad(self.netD_seg, False)

        # update G
        self.set_requires_grad(self.netD_img, False)  # D не требует градиентов при оптимизации G
        self.set_requires_grad(self.netD_seg, False)  # D не требует градиентов при оптимизации G

        self.optimizer_G.zero_grad()  #  устанавливаем градиенты G в ноль
        self.backward_G()  # вычисляем градиенты для G
        self.optimizer_G.step() #  обновляем веса G


    def confuse_pred_embed(self, pred_embed_fake, pred_embed_real=None, used_for_disc=True):
        # """
        [pred_fake_img, embed_fake_img, pred_fake_seg, embed_fake_seg] = pred_embed_fake
        [pred_real_img, embed_real_img, pred_real_seg, embed_real_seg] = pred_embed_real
        # """
        for i in range(len(pred_fake_img)):
            pred_fake_img[i] += torch.mul(embed_fake_img[i], embed_real_seg[i]).sum(dim=1, keepdim=True)
            pred_fake_seg[i] += torch.mul(embed_real_img[i], embed_fake_seg[i]).sum(dim=1, keepdim=True)

        if used_for_disc:
            for i in range(len(pred_fake_img)):
                pred_real_i = torch.mul(embed_real_img[i], embed_real_seg[i]).sum(dim=1, keepdim=True)
                pred_real_img[i] += pred_real_i
                pred_real_seg[i] += pred_real_i

            return pred_fake_img, pred_fake_seg, pred_real_img, pred_real_seg
        else:
            return pred_fake_img, pred_fake_seg




