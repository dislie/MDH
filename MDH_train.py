import csv
import math

import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
from thop import profile
from tqdm import tqdm
from loguru import logger
from models.ADSH_Loss import ADSH_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.MDH as MDH
import torch.nn as nn
import time
from torch.optim.lr_scheduler import LambdaLR

from contextlib import suppress
from torch.cuda.amp import autocast, GradScaler

from types import SimpleNamespace


class ConvNextDistillDiffPruningLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, teacher_model=None, ratio_weight=10.0, distill_weight=0.5, keep_ratio=0.5):
        super().__init__()
        self.teacher_model = teacher_model
        self.keep_ratio = keep_ratio
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight


    def forward(self, inputs, outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
        """
        token_feature, decision_mask = outputs

        # ratio loss
        ratio_loss = torch.tensor(0.0)
        ratio = self.keep_ratio

        pos_ratio = decision_mask.mean(dim=1)
        ratio_loss = ratio_loss + ((pos_ratio - ratio) ** 2).mean()  # 让生成的mask的比例接近keep_ratio

        if self.teacher_model is None:
            loss = self.ratio_weight * ratio_loss
            return loss

        # distillation loss
        with torch.no_grad():
            cls_t = self.teacher_model.forward_features(inputs.to(inputs.device))
        cls_kl_loss = torch.pow(token_feature - cls_t, 2).mean()

        loss = self.ratio_weight * ratio_loss  + self.distill_weight * cls_kl_loss
        return loss


def train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, feat_size = args.num_classes, 2048

    model = MDH.mdh(code_length=code_length, num_classes=num_classes, feat_size=feat_size,
                          device=args.device, pretrained=True)

    logger.info(model)

    model = model.cuda()

    logger.info("Calculate Params & FLOPs ...")
    inputs = torch.randn((1, 3, 224, 224)).cuda()
    macs, num_params = profile(model, (inputs,), verbose=False)  # type: ignore
    logger.info(
        "Params(M):{:.2f}M, FLOPs(G):~{:.2f}G".format(num_params / (1000 ** 2), 2 * macs / (1000 ** 3)))


    [backbone_params, other_params] = model.get_param_groups()

    ### optimizers, loss functions
    optimizers = [

        optim.SGD(backbone_params, lr=args.lr, weight_decay=1e-4, momentum=0.91, nesterov=True),
        optim.SGD(other_params, lr=args.lr, weight_decay=1e-4, momentum=0.91, nesterov=True)
    ]

    schedulers = [

        optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_step, gamma=0.1)
        for optimizer in optimizers
    ]



    cross = nn.CrossEntropyLoss()

    # loss
    criterion = ADSH_Loss(code_length, args.gamma)

    dynamic_criterion = ConvNextDistillDiffPruningLoss(teacher_model=None, ratio_weight=10.0, distill_weight=0.5, keep_ratio=0.3)

    num_retrieval = len(retrieval_dataloader.dataset)  # len = train data

    U = torch.zeros(args.num_samples, code_length).cuda()
    B = torch.randn(num_retrieval, code_length).cuda()
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().cuda()  # len = train data
    # print(len(retrieval_targets))
    cnn_losses, hash_losses, quan_losses = AverageMeter(), AverageMeter(), AverageMeter()
    cross_loss = AverageMeter()
    dynamic_losses = AverageMeter()

    keep_ratio_ = AverageMeter()
    fore_weight_ = AverageMeter()

    keep_ratio_iter = AverageMeter()
    fore_weight_iter = AverageMeter()

    start = time.time()
    best_mAP = 0
    best_iter = 0

    # 使用amp训练
    args.use_amp = True
    amp_autocast = suppress  # do nothing
    Grad_scaler = None
    if args.use_amp:
        # 初始化梯度缩放器
        Grad_scaler = GradScaler()
        amp_autocast = autocast

    for it in range(args.max_iter):

        keep_ratio_iter.reset()
        fore_weight_iter.reset()

        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset)
        sample_index = sample_index.cuda()
        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().cuda()  # len = num samples
        S = (train_targets @ retrieval_targets.t() > 0).float()  # num samples * train num

        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        # print(r)
        S = S * (1 + r) - r  #
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            cross_loss.reset()

            dynamic_losses.reset()
            keep_ratio_.reset()
            fore_weight_.reset()


            epoch_start = time.time()
            # pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            # print((len(train_dataloader)))

            for i, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.cuda(), targets.cuda(), index.cuda()

                # optimizer.zero_grad()
                [optimizer.zero_grad() for optimizer in optimizers]
                with amp_autocast():
                    F, cls, cls1, fore_weight,decision_mask = model(data)

                U[index, :] = F.data.float()
                # print('shape of F:', F.shape)
                # print('shape of B:', B.shape)
                # print('shape of S:', S.shape)
                # print('shape of S[index, :]:', S[index, :].shape)
                # print('shape of sample_index:', sample_index.shape)
                # print('shape of sample_index[index]:', sample_index[index].shape)
                # exit()
                # F：batch_size * code_length
                # B：whole_retrieval * code_length 全部样本的哈希码
                # S：num_samples * whole_retrieval 当前迭代样本与全部样本的相似度矩阵
                # sample_index：num_samples  当前迭代样本的索引
                with amp_autocast():
                    cnn_loss, hash_loss, quan_loss = criterion(F, B, S[index, :], sample_index[index])

                targets = torch.argmax(targets, dim=1)  # one-hot to index
                with amp_autocast():
                    cls_loss = (1.0 / 2.0) * cross(cls, targets) + (1.0 / 6.0) * cross(cls1, targets) #+ (1.0 / 6.0) * cross(cls2, targets)
                               #(1.0 / 6.0) * (cross(cls1, targets) + cross(cls2, targets))# + cross(cls3, targets))
                    dynamic_loss = dynamic_criterion(data, (F, decision_mask))
                cnn_loss = cnn_loss + cls_loss #+ dynamic_loss
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())
                cross_loss.update(cls_loss.item())

                decision_mask = decision_mask.mean()
                keep_ratio_.update(decision_mask.item())
                fore_weight_.update(fore_weight.mean().item())

                dynamic_losses.update(dynamic_loss.item())
                if Grad_scaler:
                    Grad_scaler.scale(cnn_loss).backward()
                    # --- 参数更新 ---
                    Grad_scaler.step(optimizers[0])
                    #
                    Grad_scaler.step(optimizers[1])
                    Grad_scaler.update()
                else:
                    cnn_loss.backward()
                    [optimizer.step() for optimizer in optimizers]
                    # optimizer.step()

            logger.info(
                '[epoch:{}/{}][cnn_loss:{:.3f}][hash_loss:{:.3f}][q_loss:{:.3f}][cls_loss:{:.3f}][keep_ratio:{:.2f}][fore_weight:{:.2f}][lr_backbone:{:.1e}][lr_other:{:.1e}][epoch_time:{:.1f}s]'.format(
                    epoch + 1, args.max_epoch,
                    cnn_losses.avg,
                    hash_losses.avg,
                    quan_losses.avg,
                    cross_loss.avg,
                    keep_ratio_.avg,
                    fore_weight_.avg,
                    optimizers[0].param_groups[0]['lr'],
                    optimizers[1].param_groups[0]['lr'],
                    time.time() - epoch_start,
                    ))

            keep_ratio_iter.update(keep_ratio_.avg)
            fore_weight_iter.update(fore_weight_.avg)
        # scheduler.step()

        # Update B
        expand_U = torch.zeros(B.shape).cuda()
        expand_U[sample_index, :] = U
        B = solve_dcc(B, U, expand_U, S, code_length, args.gamma)

        logger.info(
            '[iter:{}/{}][iter_time:{:.1f}s][keep_ratio_iter:{:.2f}][fore_weight_iter:{:.2f}][lr_backbone:{:.1e}][lr_other:{:.1e}] [code_length:{}]'.format(
                it + 1, args.max_iter,
                time.time() - iter_start,
                keep_ratio_iter.avg,
                fore_weight_iter.avg,
                optimizers[0].param_groups[0]['lr'],
                optimizers[1].param_groups[0]['lr'],
                args.code_length,))

        if it % 1 == 0:
            query_code = generate_code(model, query_dataloader, code_length, args.device)
            # print(len(query_dataloader))
            mAP = evaluate.mean_average_precision(
                query_code.cuda(),
                B,
                query_dataloader.dataset.get_onehot_targets().cuda(),
                retrieval_targets,
                args.device,
                args.topk,
            )
            # logger.info(
            #     '[iter:{}/{}][code_length:{}][mAP:{:.4f}]'.format(it + 1, args.max_iter, code_length,
            #                                                       mAP))

            if mAP > best_mAP:
                best_mAP = mAP
                best_iter = it
                ret_path = os.path.join(args.base_path, 'checkpoints')
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.pth'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.pth'))
                torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.pth'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.pth'))
                torch.save(model.state_dict(), os.path.join(ret_path, 'model.pth'))

            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.4f}][best_mAP:{:.4f}]'.format(it + 1, args.max_iter, code_length,
                                                                                   mAP, best_mAP))
            logger.info('best_iter:{}'.format(best_iter + 1))

            ####
            csv_path = os.path.join(args.base_path, 'summary.csv')
            row = {
                'iter': it + 1,
                'code_length': code_length,
                'mAP': mAP.item(),
                'lr_backbone': optimizers[0].param_groups[0]['lr'],
                'lr_other': optimizers[1].param_groups[0]['lr'],
                'best_mAP': best_mAP.item(),
                'best_iter': best_iter + 1,
            }
            with open(csv_path, mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=row.keys())
                if it == 0:  # first iteration
                    dw.writeheader()
                dw.writerow(row)
            ####
        [scheduler.step() for scheduler in schedulers]
        # scheduler.step()

    logger.info('[Training time:{:.2f}s]'.format(time.time() - start))

    return best_mAP


def solve_dcc(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quantization_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quantization_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).cuda()
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.cuda(), targets.cuda(), index.cuda()
            hash_code = model(data)
            code[index, :] = hash_code.sign()
    model.train()
    return code


# class WarmupCosineSchedule(LambdaLR):
#     """ Linear warmup and then cosine decay.
#         Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
#         Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
#         If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
#     """
#
#     def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.t_total = t_total
#         self.cycles = cycles
#         super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
#
#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return float(step) / float(max(1.0, self.warmup_steps))
#         # progress after warmup
#         progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
#         return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
class WarmupCosineSchedule(LambdaLR):
    """
    Warmup + Cosine annealing scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        total_epochs (int): Total number of epochs for training.
        min_lr (float): Minimum learning rate after decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_epoch: int) -> float:
        # Linear warmup phase
        if current_epoch < self.warmup_epochs:
            return float(current_epoch) / float(max(1, self.warmup_epochs))
        # After warmup, apply cosine annealing
        progress = float(current_epoch - self.warmup_epochs) / \
                   float(max(1, self.total_epochs - self.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_decay * (1 - self.min_lr / self.base_lrs[0]) + (self.min_lr / self.base_lrs[0])