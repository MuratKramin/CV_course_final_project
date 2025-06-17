import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F


def get_norm_layer(norm_type='instance'):

    """ Возвращает слой нормализации
    Параметры:
        norm_type (str) -- имя слоя нормализации: batch | instance | none
    Для BatchNorm мы используем обучаемые параметры смещения и масштаба, а также отслеживаем статистику (среднее/стандартное отклонение).
    Для InstanceNorm мы не используем обучаемые параметры смещения и масштаба. Мы не отслеживаем статистику.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(f'Слой нормализации с именем [{norm_type}] не найден')
    return norm_layer


# def get_scheduler(optimizer, opt):
#     if opt.lr_policy == 'linear':
#         def lambda_rule(epoch):
#             lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)

#             #новые строки
#             if lr_l < 0:
#                 print(f" WARNING: computed learning rate multiplier is negative ({lr_l:.6f}) at epoch {epoch}. Setting to 0.")
#             lr_l = max(0.0, lr_l)
#             #конец новых строк

#             return lr_l

#         scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
#     elif opt.lr_policy == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
#     elif opt.lr_policy == 'plateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
#     elif opt.lr_policy == 'cosine':
#         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
#     return scheduler

import os  # в начале файла, если не импортирован
def get_scheduler(optimizer, opt):
    """ Возвращает планировщик скорости обучения на основе параметров обучения."""

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            if lr_l < 0:
                print(f"ПРЕДУПРЕЖДЕНИЕ: вычисленный множитель скорости обучения отрицательный ({lr_l:.6f}) на эпохе {epoch}. Устанавливаем в 0.")
            return max(0.0, lr_l)

        # === Используем opt.load_iter или opt.epoch вместо файла
        if opt.continue_train:
            if opt.load_iter > 0:
                last_epoch = opt.load_iter
            else:
                try:
                    last_epoch = int(opt.which_epoch)
                except ValueError:
                    last_epoch = 0
            last_epoch -= 1  # PyTorch scheduler expects (N - 1)
            print(f" Восстанавливаем scheduler с last_epoch={last_epoch}")
        else:
            last_epoch = -1  # fresh training

        # Установим initial_lr вручную, если он не задан (иначе PyTorch упадёт при last_epoch > -1)
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']


        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        raise NotImplementedError(f'Политика обучения с именем [{opt.lr_policy}] не реализована')

    return scheduler




def init_weights(net, init_type='normal', init_gain=0.02):
    """ Инициализация весов сети.

    Параметры:
        net (network)   -- сеть для инициализации
        init_type (str) -- имя метода инициализации: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- коэффициент масштабирования для normal, xavier и orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Инициализация сети с методом {init_type}')
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    """ Инициализация сети: 1. регистрация устройства CPU/GPU (с поддержкой нескольких GPU); 2. инициализация весов сети
    Параметры:
        net (network)      -- сеть для инициализации
        init_type (str)    -- имя метода инициализации: normal | xavier | kaiming | orthogonal
        gain (float)       -- коэффициент масштабирования для normal, xavier и orthogonal.
        gpu_ids (int list) -- какие GPU использовать: например, 0,1,2
        Возвращает инициализированную сеть.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_net_afl(net, init_type='normal', init_gain=0.02, gpu_ids=[]):

    """ Инициализация сети: 1. регистрация устройства CPU/GPU (с поддержкой нескольких GPU); 2. инициализация весов сети
    Параметры:
        net (network)      -- сеть для инициализации
        init_type (str)    -- имя метода инициализации: normal | xavier | kaiming | orthogonal
        gain (float)       -- коэффициент масштабирования для normal, xavier и orthogonal.
        gpu_ids (int list) -- какие GPU использовать: например, 0,1,2
        Возвращает инициализированную сеть.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net.module.main, init_type, init_gain=init_gain)
    init_weights(net.module.netGA, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[], netD=None, alpha=[0.5, 0.5, 0.5, 0.5, 0.5], loop_count=2, ndf=64):

    """ Создание генератора

    Параметры:
        input_nc (int) -- количество каналов во входных изображениях
        output_nc (int) -- количество каналов в выходных изображениях
        ngf (int) -- количество фильтров в последнем сверточном слое
        netG (str) -- имя архитектуры: unet_afl_v5
        norm (str) -- тип нормализации, используемый в сети: batch | instance | none
        use_dropout (bool) -- использовать ли слои dropout.
        init_type (str)    -- имя метода инициализации.
        init_gain (float)  -- коэффициент масштабирования для normal, xavier и orthogonal.
        gpu_ids (int list) -- какие GPU использовать: например, 0,1,2
    Возвращает генератор
        Текущая реализация предоставляет одну архитектуру генератора:
        [unet_afl_v5]: модифицированная U-Net архитектура, адаптированная для задачи обратной обратной связи в adversarial learning.
        """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_afl_v5':
        net = UnetAFL_v5(input_nc=input_nc, output_nc=output_nc, ngf=ngf, ndf=ndf,
                         norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Имя модели генератора [%s] не распознано' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """ Создание дискриминатора

    Параметры:
        input_nc (int) -- количество каналов во входных изображениях
        ndf (int) -- количество фильтров в первом сверточном слое
        netD (str) -- имя архитектуры: fpd
        n_layers_D (int) -- количество сверточных слоев в дискриминаторе; эффективно, когда netD=='n_layers'
        norm (str) -- тип нормализации, используемый в сети.
        init_type (str)    -- имя метода инициализации.
        init_gain (float)  -- коэффициент масштабирования для normal, xavier и orthogonal.
        gpu_ids (int list) -- какие GPU использовать: например, 0,1,2
        Возвращает дискриминатор

    Текущая реализация предоставляет одну архитектуру дискриминатора:
        [fpd]: Feature Pyramid Discriminator, который использует пирамидальную структуру для обработки изображений
        с различными масштабами и разрешениями. Он эффективен для задач, где требуется анализировать
        изображения на разных уровнях детализации, что позволяет лучше различать реальные и сгенерированные изображения.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'fpd':
        net = FeaturePyramidDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Имя модели дискриминатора [%s] не распознано' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


#--------------------- UnetAFL_v5----------------------------------------
class UnetAFL_v5(nn.Module):
    """ 
    UnetAFL_v5: модифицированная U-Net архитектура, адаптированная для задачи обратной обратной связи в adversarial learning.
    """
    def __init__(self, input_nc, output_nc, ngf=64, ndf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type='reflect', disc_nc_scale=[1,1,1,1,1]):
        super(UnetAFL_v5, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ngf_upbound = ngf * (2**8)
        encoder_layer_nc = []
        decoder_layer_nc = []
        #---------encoding layer 1---------------------
        # Первый слой энкодера принимает входные данные с количеством каналов input_nc.
        ngf_in = input_nc
        ngf_out = min(ngf, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer1 = nn.Sequential(
            # state size: 64 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                               stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 2---------------------
        # Второй слой энкодера принимает входные данные из первого слоя энкодера.
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer2 = nn.Sequential(
            # state size: ngf --> ngf *2 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 3---------------------
        # Третий слой энкодера принимает входные данные из второго слоя энкодера.
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer3 = nn.Sequential(
            # state size: ngf*2 --> ngf*4 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer 4---------------------
        # Четвертый слой энкодера принимает входные данные из третьего слоя энкодера.
        ngf_in = ngf_out
        ngf_out = ngf_out * 2
        ngf_out = min(ngf_out, ngf_upbound)
        encoder_layer_nc.insert(0, ngf_out)
        self.encoder_layer4 = nn.Sequential(
            # state size: ngf*4 -- ngf*8 channel
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            norm_layer(ngf_out),
            nn.ReLU(inplace=True)
        )

        # ---------encoding layer with resnet block---------------------
        # Слой энкодера с блоками ResNet, который принимает входные данные из четвертого слоя энкодера.
        ngf_in = ngf_out
        encoder_layer_nc.insert(0, ngf_out)
        model = []
        for i in range(4):       # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.encoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer with resnet block ---------------------
        # Слой декодера с блоками ResNet, который принимает входные данные из слоя обратной связи и из четвертого слоя энкодера.
        decoder_layer_nc.extend([ngf_in])
        model = []
        for i in range(5):  # add ResNet blocks
            model += [ResnetBlock(ngf_in, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.decoder_resblock_layer = nn.Sequential(*model)

        # ---------decoding layer 1 ---------------------
        # Первый слой декодера принимает входные данные из слоя обратной связи и из четвертого слоя энкодера.
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2) # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer1 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
            )

        # ---------decoding layer 2 ---------------------
        # Второй слой декодера принимает входные данные из первого слоя декодера и из четвертого слоя энкодера.
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer2 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )

        # ---------decoding layer 3 ---------------------
        # Третий слой декодера принимает входные данные из второго слоя декодера и из третьего слоя энкодера.
        ngf_in = ngf_out
        ngf_out = int(ngf_in / 2)
        ngf_in = int(ngf_in * 2)  # skip connection
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer3 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf_out),
            nn.ReLU(True)
        )
        # ---------decoding layer 4 ---------------------
        # Четвертый слой декодера принимает входные данные из третьего слоя декодера и из четвертого слоя энкодера.
        ngf_in = ngf_out
        ngf_in = int(ngf_in * 2)  # skip connection
        ngf_out = output_nc
        decoder_layer_nc.extend([ngf_in])
        self.decoder_layer4 = nn.Sequential(
            # state size: 256 channel
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf_in, ngf_out, kernel_size=3, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )

        # ---------feedback layers ---------------------
        # Слои обратной связи, которые будут использоваться для трансформации выходов дискриминатора.
        # disc_layer_nc = np.array([8, 8, 4, 2, 1]) * ndf
        disc_layer_nc = ndf * np.array(disc_nc_scale)
        gene_layer_nc = decoder_layer_nc + disc_layer_nc*2 # disc_layer_nc*2: disc_out of image and segmentation

        self.trans_block0 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[0], output_nc=decoder_layer_nc[0])
        self.trans_block1 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[1], output_nc=decoder_layer_nc[1])
        self.trans_block2 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[2], output_nc=decoder_layer_nc[2])
        self.trans_block3 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[3], output_nc=decoder_layer_nc[3])
        self.trans_block4 = TransBlockDual(afl_type=1, input_nc=gene_layer_nc[4], output_nc=decoder_layer_nc[4])

        # ----------normalization of feedback output ------------
        # Слои нормализации для обратной связи, которые применяются к выходам из трансформационных блоков.
        # Эти слои помогают стабилизировать обучение, нормализуя выходы после применения трансформационных блоков.
        # Они используются для управления масштабом и распределением значений в выходных данных, что может улучшить качество генерации.
        self.norm_layer0 = norm_layer(decoder_layer_nc[0])
        self.norm_layer1 = norm_layer(decoder_layer_nc[1])
        self.norm_layer2 = norm_layer(decoder_layer_nc[2])
        self.norm_layer3 = norm_layer(decoder_layer_nc[3])
        self.norm_layer4 = norm_layer(decoder_layer_nc[4])

    def forward(self, gene_input, disc_out=None, alpha=None):
        if disc_out is None:
            e1_out = output = self.encoder_layer1(gene_input)
            e2_out = output = self.encoder_layer2(output)
            e3_out = output = self.encoder_layer3(output)
            e4_out = output = self.encoder_layer4(output)
            res_encoder_out = output = self.encoder_resblock_layer(output)
            res_decoder_out = output = self.decoder_resblock_layer(output)
            output = torch.cat((output, e4_out),1)
            output = self.decoder_layer1(output)
            output = torch.cat((output, e3_out), 1)
            output = self.decoder_layer2(output)
            output = torch.cat((output, e2_out), 1)
            output = self.decoder_layer3(output)
            output = torch.cat((output, e1_out), 1)
            output = self.decoder_layer4(output)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]
            encoder_out = [res_encoder_out, e4_out, e3_out, e2_out, e1_out]

            return gen_img, gen_seg, encoder_out

        else:
            # input_fea = torch.cat((decoder_inner_fea, fea_list[0]), 1)
            feedback_input = gene_input[0]
            feedback_input_new = torch.cat((disc_out[0], feedback_input), 1)
            encoder_out = feedback_out = feedback_input + alpha[0] * self.norm_layer0(self.trans_block0(feedback_input_new))
            feedback_input = self.decoder_resblock_layer(feedback_out)


            feedback_input = torch.cat((feedback_input, gene_input[1]), 1)
            feedback_input_new = torch.cat((disc_out[1], feedback_input), 1)
            feedback_out = feedback_input + alpha[1] * self.norm_layer1(
                self.trans_block1(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[1]), 1)
            feedback_input = self.decoder_layer1(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[2]), 1)
            feedback_input_new = torch.cat((disc_out[2], feedback_input), 1)
            feedback_out = feedback_input + alpha[2] * self.norm_layer2(
                self.trans_block2(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[2]), 1)
            feedback_input = self.decoder_layer2(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[3]), 1)
            feedback_input_new = torch.cat((disc_out[3], feedback_input), 1)
            feedback_out = feedback_input + alpha[3] * self.norm_layer3(
                self.trans_block3(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, gene_input[3]), 1)
            feedback_input = self.decoder_layer3(feedback_out)

            feedback_input = torch.cat((feedback_input, gene_input[4]), 1)
            feedback_input_new = torch.cat((disc_out[4], feedback_input), 1)
            feedback_out = feedback_input + alpha[4] * self.norm_layer4(
                self.trans_block4(feedback_input_new))
            # feedback_out = torch.cat((feedback_out, fea_list[4]), 1)
            output = self.decoder_layer4(feedback_out)
            gen_img = output[:, 0:3, ::]
            gen_seg = output[:, 3:6, ::]

            return gen_img, gen_seg, encoder_out

    def get_main_layer_result(self):
        return [self.res_encoder_out, self.e1_out, self.e2_out, self.e3_out, self.e4_out]

class TransBlockDual(nn.Module):
    def __init__(self, afl_type=1, input_nc=6, output_nc=3):
        super(TransBlockDual, self).__init__()
        if afl_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(True),
                nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(output_nc),
                nn.ReLU(True))
        elif afl_type == 2:
            pass

    def forward(self, input):
        return self.main(input)


class GeneratorAFL(nn.Module):
    def __init__(self, alf_type=1, inner_nc_list=None, outer_nc_list=None):
        super(GeneratorAFL, self).__init__()

        self.trans_block0 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[0], output_nc=outer_nc_list[0])
        self.trans_block1 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[1], output_nc=outer_nc_list[1])
        self.trans_block2 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[2], output_nc=outer_nc_list[2])
        self.trans_block3 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[3], output_nc=outer_nc_list[3])
        self.trans_block4 = TransBlockDual(afl_type=1, input_nc=inner_nc_list[4], output_nc=outer_nc_list[4])

    def set_input_disc(self, layers_input):
        self.feedback0 = layers_input[0]
        self.feedback1 = layers_input[1]
        self.feedback2 = layers_input[2]
        self.feedback3 = layers_input[3]
        self.feedback4 = layers_input[4]


class ResnetBlock(nn.Module):
    """ ResnetBlock: Определяет блок ResNet, который состоит из двух сверточных слоев с пропусками (skip connections)."""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """ Инициализация блока ResNet

        Блок ResNet - это сверточный блок с пропусками (skip connections).
        Мы строим сверточный блок с помощью функции build_conv_block,
        и реализуем пропуски в функции <forward>.
        Оригинальная статья Resnet: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        """ Конструирует сверточный блок.
        Параметры:
            dim (int)           -- количество каналов в сверточном слое.
            padding_type (str)  -- имя слоя паддинга: reflect | replicate | zero
            norm_layer          -- слой нормализации
            use_dropout (bool)  -- использовать ли слои dropout.
            use_bias (bool)     -- использовать ли смещение в сверточном слое
        Возвращает сверточный блок (сверточный слой, слой нормализации и слой нелинейности (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('тип паддинга [%s] не реализован' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # "пропуски" (skip connections) позволяют передавать информацию из входа в выход блока
        return out


class FeaturePyramidDiscriminator(nn.Module):
    """ Дискриминатор пирамиды признаков (Feature Pyramid Discriminator) - это архитектура дискриминатора,"""
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, disc_nc_scale=[1,1,1,1,1]):
            super(FeaturePyramidDiscriminator, self).__init__()
            if type(norm_layer) == functools.partial: # не нужно использовать смещение, так как BatchNorm2d имеет аффинные параметры
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            # нижняя часть пути: сверточные слои для извлечения признаков и боковые соединения
            # Здесь мы используем боковые соединения для объединения признаков из нижней части пути с признаками из верхней части пути.
            # Это позволяет сохранить информацию о высоком разрешении и улучшить качество сегментации.
            ndf_in = input_nc
            ndf_out = ndf
            self.l1 = nn.Sequential(  # input size is 256
                nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, inplace=True))

            ndf_in = ndf_out
            ndf_out = ndf_in * 2
            self.l2 = nn.Sequential(  # input size is 256
                nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, inplace=True))

            ndf_in = ndf_out
            ndf_out = ndf_in * 2
            self.l3 = nn.Sequential(  # input size is 256
                nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, inplace=True))

            ndf_in = ndf_out
            ndf_out = ndf_in * 2
            self.l4 = nn.Sequential(  # input size is 256
                nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, inplace=True))

            ndf_in = ndf_out
            ndf_out = ndf_in
            self.l5 = nn.Sequential(  # input size is 256
                nn.Conv2d(ndf_in, ndf_out, kernel_size=3, stride=1, padding=1, bias=use_bias),
                # nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, inplace=True))

            # верхняя часть пути: сверточные слои для извлечения признаков и боковые соединения
            # Здесь мы используем боковые соединения для объединения признаков из нижней части пути с признаками из верхней части пути.
            # Это позволяет сохранить информацию о высоком разрешении и улучшить качество сегментации.

            ndf_out = ndf * 2
            self.lat1 = nn.Sequential(
                nn.Conv2d(ndf, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            self.lat2 = nn.Sequential(
                nn.Conv2d(ndf * 2, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            self.lat3 = nn.Sequential(
                nn.Conv2d(ndf * 4, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            self.lat4 = nn.Sequential(
                nn.Conv2d(ndf * 8, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            self.lat5 = nn.Sequential(
                nn.Conv2d(ndf * 8, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))

            # Upsample используется для увеличения разрешения признаков, полученных из нижней части пути.
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

            # финальные слои: сверточные слои для предсказания истинности/ложности на основе патчей и предсказания сегментации
            ndf_in = ndf_out
            ndf_out = ndf * disc_nc_scale[0]
            self.final1 = nn.Sequential(
                nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            ndf_out = ndf * disc_nc_scale[1]
            self.final2 = nn.Sequential(
                nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            ndf_out = ndf * disc_nc_scale[2]
            self.final3 = nn.Sequential(
                nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            ndf_out = ndf * disc_nc_scale[3]
            self.final4 = nn.Sequential(
                nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))
            ndf_out = ndf * disc_nc_scale[4]
            self.final5 = nn.Sequential(
                nn.Conv2d(ndf_in, ndf_out, kernel_size=1),
                norm_layer(ndf_out),
                nn.LeakyReLU(0.2, True))

            # Предсказание истинности/ложности на основе патчей и предсказание сегментации
            ndf_in = ndf_out
            ndf_out = ndf_in
            self.tf = nn.Conv2d(ndf_in, 1, kernel_size=1)
            self.seg = nn.Conv2d(ndf_in, ndf_out, kernel_size=1)

    def forward(self, input):
        """ Стандартная функция прямого распространения."""
        # нижняя часть пути: сверточные слои для извлечения признаков

        l1out = out = self.l1(input)
        l2out = out = self.l2(out)
        l3out = out = self.l3(out)
        l4out = out = self.l4(out)
        l5out = out = self.l5(out)
        # верхняя часть пути и боковые соединения
        feat25 = self.lat5(l5out)
        feat24 = feat25 + self.lat4(l4out)
        feat23 = self.up(feat24) + self.lat3(l3out)
        feat22 = self.up(feat23) + self.lat2(l2out)
        feat21 = self.up(feat22) + self.lat1(l1out)

        # финальные слои предсказания
        feat31 = self.final1(feat21)
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)
        feat35 = self.final5(feat25)

        # Предсказание истинности/ложности на основе патчей
        pred1 = self.tf(feat31)
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        pred5 = self.tf(feat35)

        # проекция признаков для сопоставления признаков

        seg1 = self.seg(feat31)
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)
        seg5 = self.seg(feat35)

        pred = [pred1, pred2, pred3, pred4, pred5]
        seg = [seg1, seg2, seg3, seg4, seg5]
        feat = [feat35.detach(), feat34.detach(), feat33.detach(), feat32.detach(), feat31.detach()]
        return[pred, seg, feat]