import argparse
import datetime
import glob
import logging
import numpy as np
import os
import shutil
import sys
import time
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from backbone import EfficientDetBackbone
from coco_eval import evaluate
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from prepare_data import prepare_annotations
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, init_weights

_DEFAULT_BATCH_SIZE = 64
_INPUT_SIZES = (512, 640, 768, 896, 1024, 1280, 1280, 1536)
_STRFTIME_FORMAT = "%Y-%m-%d-%H-%M-%S"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

num_cpus = int(os.environ["SM_NUM_CPUS"])
num_gpus = int(os.environ["SM_NUM_GPUS"])
logger.info(f"Number of CPUs available: {num_cpus}")
logger.info(f"Number of GPUs available: {num_gpus}")


def _parse_args():
    parser = argparse.ArgumentParser("EfficientDet training script.")
    parser.add_argument(
        "--classes",
        type=lambda x: sorted(set([s.strip().upper() for s in eval(x)])),
        metavar="N",
        help="List of classes with format \"['CLASS_NAME_1', ..., 'CLASS_NAME_N']\".",
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        metavar="N",
        help="Path to the local directory where the trained model will be saved. "
        "The model is copied from the local folder to S3 before the SageMaker "
        "container is deleted. Default: '/opt/ml/model'.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/opt/ml/checkpoints",
        metavar="N",
        help="Path to the local directory where checkpoints are saved during training. "
        "Checkpoints are copied from the local folder to S3 during training. "
        "Default: '/opt/ml/checkpoints'.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="/opt/ml/output/tensorboard",
        metavar="N",
        help="Path to the local directory where tensorboard logs are saved during "
        "training. Logs are copied from the local folder to S3 during training. "
        "Default: '/opt/ml/output/tensorboard'.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_DATA"],
        metavar="N",
        help="Path to the local dir where the data are downloaded. "
        "Default: '/opt/ml/input/data/data'.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=os.environ["SM_CHANNEL_MODEL"],
        metavar="N",
        help="Path to the local dir where the weights used to initialize the model "
        "are downloaded. Default: '/opt/ml/input/data/weights'.",
    )
    parser.add_argument(
        "--freeze-backbone",
        type=eval,
        default=False,
        metavar="N",
        help="Whether to fine-tune only the regressor and the classifier, "
        "useful in early stage convergence or small/easy dataset. Default: False.",
    )
    parser.add_argument(
        "--compound-coef",
        type=int,
        default=0,
        metavar="N",
        help="EfficientDet compound coefficient (phi in the paper). Default: 0.",
    )
    parser.add_argument(
        "--mean",
        type=eval,
        default=[0.485, 0.456, 0.406],
        metavar="N",
        help="List of mean values in RGB order used to normalize the images. "
        "Default: '[0.485, 0.456, 0.406]' (ImageNet mean).",
    )
    parser.add_argument(
        "--std",
        type=eval,
        default=[0.229, 0.224, 0.225],
        metavar="N",
        help="List of std values in RGB order used to normalize the images. "
        "Default: '[0.229, 0.224, 0.225]', (ImageNet std).",
    )
    parser.add_argument(
        "--anchors-scales",
        type=eval,
        default=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        metavar="N",
        help="List of anchors scales, they should be adapted to the specific dataset. "
        "Default: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]' (typical anchors "
        "scales for COCO).",
    )
    parser.add_argument(
        "--anchors-ratios",
        type=eval,
        default=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
        metavar="N",
        help="List of anchors ratios, they should be adapted to the specific dataset. "
        "Default: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]' (typical anchors "
        "ratios for COCO).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        metavar="N",
        help="Number of images per batch per GPU, e.g. if there are 8 GPUs and "
        "batch_size=4, the total batch size is 32. Default: 4.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.08,
        metavar="N",
        help="Unadjusted learning rate, the adjusted learning rate is "
        "learning_rate * total_batch_size / _DEFAULT_BATCH_SIZE. Default: 0.08.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=6.4e-6,
        metavar="N",
        help="Weight decay. Notice that this is the real weight decay, "
        "not the L2 regularization. Default: 6.4e-6.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="N",
        help="Optimizer momentum. Default: 0.9.",
    )
    parser.add_argument(
        "--lr-warmup-epoch",
        type=float,
        default=1.0,
        metavar="N",
        help="Number of warm-up epochs for the learning rate, does not have to be an "
        "int. Default: 1.0.",
    )
    parser.add_argument(
        "--lr-warmup-init",
        type=float,
        default=0.008,
        metavar="N",
        help="Initial value of the learning rate at the beginning of the warm-up "
        "phase. Default: 0.008.",
    )
    parser.add_argument(
        "--anneal-strategy",
        type=str,
        default="cos",
        metavar="N",
        help="Specifies the annealing strategy: 'cos' for cosine annealing, "
        "'linear' for linear annealing. Default: 'cos'.",
    )
    parser.add_argument(
        "--cycle-momentum",
        type=eval,
        default=False,
        metavar="N",
        help="If True, momentum is cycled inversely to learning rate between "
        "'base_momentum' and 'max_momentum'. Default: False.",
    )
    parser.add_argument(
        "--base-momentum",
        type=float,
        default=0.85,
        metavar="N",
        help="Lower momentum boundaries in the cycle for each parameter group. "
        "Note that momentum is cycled inversely to learning rate; at the peak of a "
        "cycle, momentum is 'base_momentum' and learning rate is 'max_lr'. "
        "Default: 0.85.",
    )
    parser.add_argument(
        "--max-momentum",
        type=float,
        default=0.95,
        metavar="N",
        help="Upper momentum boundaries in the cycle for each parameter group. "
        "Functionally, it defines the cycle amplitude (max_momentum - base_momentum). "
        "Note that momentum is cycled inversely to learning rate; at the start of a "
        "cycle, momentum is 'max_momentum' and learning rate is 'base_lr'. "
        "Default: 0.95.",
    )
    parser.add_argument(
        "--final-div-factor",
        type=float,
        default=1e4,
        metavar="N",
        help="Determines the minimum learning rate via "
        "min_lr = initial_lr / final_div_factor. Default: 1e4.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="N",
        help="Number of epochs to train. Default: 300.",
    )
    parser.add_argument(
        "--clip-gradients-norm",
        type=float,
        default=10.0,
        metavar="N",
        help="Max norm of the gradients, set to 0.0 for no gradient clipping. "
        "Default: 10.0.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        metavar="N",
        help="Number of workers used by data loaders. If --num-workers is a negative "
        "value, batch_size * num_gpus workers are used. Default: -1.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N", help="Random seed. Default: 42."
    )
    parser.add_argument(
        "--weights",
        type=lambda x: None if x.strip().lower() == "none" else x,
        default=None,
        metavar="N",
        help="File name of the weights to be loaded or 'best' to select the best "
        "weights in the weights dir. None to initialize the weights from scratch. "
        "Default: None.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        metavar="N",
        help="Number of epochs between validation phases. Default: 1.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="sgd",
        metavar="N",
        help="Optimizer, 'sgd' or 'adamw'. Default: 'sgd'.",
    )
    parser.add_argument(
        "--scheduler",
        type=lambda x: None if x.strip().lower() == "none" else x,
        default="onecyclelr",
        metavar="N",
        help="Learning rate scheduler, 'onecyclelr' or None. Default: 'onecyclelr'.",
    )
    parser.add_argument(
        "--milestones",
        type=eval,
        default=[200, 250],
        metavar="N",
        help="List of epoch indices. Must be increasing. Default: '[200, 250]'.",
    )
    parser.add_argument(
        "--multisteplr-gamma",
        type=float,
        default=0.1,
        metavar="N",
        help="Multiplicative factor of learning rate decay. Default: 0.1.",
    )
    parser.add_argument(
        "--max-images",
        type=eval,
        default=None,
        metavar="N",
        help="Tuple like (max_images_train, max_images_val) of the maximum number of "
        "images to include in the training and validation sets or None to set no "
        "limits. Default: None.",
    )
    parser.add_argument(
        "--es-min-delta",
        type=float,
        default=0.0,
        metavar="N",
        help="Minimum change in the validation loss to qualify as an improvement, "
        "i.e. an absolute change of less than min_delta, will count as no improvement. "
        "Default: 0.0.",
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=0,
        metavar="N",
        help="Number of epochs with no improvement after which training will be "
        "stopped. Default: 0.",
    )
    parser.add_argument(
        "--es-baseline",
        type=float,
        default=np.inf,
        metavar="N",
        help="Baseline value for the monitored quantity. Training will stop if the "
        "model doesn't show improvement over the baseline. Default: np.inf.",
    )
    parser.add_argument(
        "--resume-training",
        type=eval,
        default=False,
        metavar="N",
        help="If True, resume training using the last step and early stopping baseline "
        "inferred from the weights path. Default: False.",
    )
    parser.add_argument(
        "--use-float16",
        type=lambda x: True if x.lower() == "true" else False,
        default=False,
        metavar="N",
        help="If to use half precision. Default: False.",
    )
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.05,
        metavar="N",
        help="Score threshold. Default: 0.05.",
    )
    parser.add_argument(
        "--eval-nms-threshold",
        type=float,
        default=0.5,
        metavar="N",
        help="Non Maximum Suppression threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--eval-max-imgs",
        type=int,
        default=10000,
        metavar="N",
        help="Maximum number of images considered. Default: 10000.",
    )
    parser.add_argument(
        "--eval-device",
        type=int,
        default=0,
        metavar="N",
        help="Device index to select. Default: 0.",
    )
    return parser.parse_known_args()


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model

    def forward(self, imgs, annotations):
        _, regressions, classifications, anchors = self.model(imgs)
        loss_cls, loss_box_reg = self.criterion(
            classifications, regressions, anchors, annotations
        )
        return loss_cls, loss_box_reg


class EarlyStopping:
    def __init__(self, args, baseline, best_epoch):
        self.args = args
        self.best_loss = baseline
        self.best_epoch = best_epoch
        self.stop_training = False

    def step(self, epoch, loss):
        if not self.args.es_patience > 0:
            pass
        elif loss + self.args.es_min_delta < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
        elif epoch - self.best_epoch >= self.args.es_patience:
            self.stop_training = True
            logger.info(
                f"Early stopping at epoch {epoch}. The lowest loss achieved "
                f"is {self.best_loss} at epoch {self.best_epoch}"
            )

        return self.stop_training


def _get_train_data_loader(args):
    logger.info("Getting train data loader")
    dataset = CocoDataset(
        root_dir=args.data_dir,
        set="train",
        transform=transforms.Compose(
            [
                Normalizer(mean=args.mean, std=args.std),
                Augmenter(),
                Resizer(_INPUT_SIZES[args.compound_coef]),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size * num_gpus,
        shuffle=True,
        drop_last=True,
        collate_fn=collater,
        num_workers=args.num_workers
        if args.num_workers >= 0
        else args.batch_size * num_gpus,
    )


def _get_val_data_loader(args):
    logger.info("Getting val data loader")
    dataset = CocoDataset(
        root_dir=args.data_dir,
        set="val",
        transform=transforms.Compose(
            [
                Normalizer(mean=args.mean, std=args.std),
                Resizer(_INPUT_SIZES[args.compound_coef]),
            ]
        ),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size * num_gpus,
        shuffle=False,
        drop_last=True,
        collate_fn=collater,
        num_workers=args.num_workers
        if args.num_workers >= 0
        else args.batch_size * num_gpus,
    )


def _get_last_step_and_es_baseline(weights_path, resume_training):
    if weights_path is None or not resume_training:
        return 0, args.es_baseline

    weights_path_split = weights_path.split("_")
    last_step = int(weights_path_split[-2])
    es_baseline = float(weights_path_split[-1].split(".pt")[0])
    return last_step, es_baseline


def _get_best_weights_path(weights_dir):
    # warning: don't write *.pt*, sagemaker generate files like
    # filename.pth.sagemaker-uploaded
    weights_paths = glob.glob(os.path.join(weights_dir, "*.pth"))
    logger.info(
        f"Selecting the best weights in {weights_dir} ({len(weights_paths)} .pth files "
        "found)"
    )
    if len(weights_paths) > 0:
        return sorted(weights_paths, key=lambda x: x.split("_")[-1].split(".pth")[0])[0]
    else:
        return None


def _get_weights_path(weights_dir, weights):
    if weights is None:
        weights_path = None
    elif weights.endswith(".pth"):
        weights_path = os.path.join(weights_dir, weights)
    elif weights == "best":
        weights_path = _get_best_weights_path(weights_dir)
    else:
        raise ValueError(f"{weights} is not a valid weights path.")

    return weights_path


def _init_weights(model, weights_path):
    logger.info("Initializing weights")
    if weights_path is None:
        init_weights(model)
    elif weights_path.endswith(".pth"):
        logger.info(f"Loading weights {os.path.basename(weights_path)}")
        try:
            model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            logger.info(
                f"Ignoring {e} probably caused by loading weights of a model with a "
                "different number of classes. The part of the weights not affected by "
                "this should already be loaded."
            )
    else:
        raise ValueError(f"{weights_path} is not a valid weights path.")


def _freeze_submodule_if_backbone(submodule):
    class_name = submodule.__class__.__name__
    if ("EfficientNet" == class_name) or ("BiFPN" == class_name):
        for param in submodule.parameters():
            param.requires_grad = False


def _get_optimizer(model, args):
    lr = args.learning_rate * args.batch_size * num_gpus / _DEFAULT_BATCH_SIZE
    l2 = args.weight_decay / lr
    if args.optim == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=args.momentum, weight_decay=l2
        )
    elif args.optim == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"{args.optim} is not a supported optimizer.")

    return optimizer


def _get_scheduler(optimizer, steps_per_epoch, args):
    lr = args.learning_rate * args.batch_size * num_gpus / _DEFAULT_BATCH_SIZE
    if args.scheduler == "onecyclelr":
        pct_start = args.lr_warmup_epoch / args.epochs
        div_factor = args.learning_rate / args.lr_warmup_init
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=args.anneal_strategy,
            cycle_momentum=args.cycle_momentum,
            base_momentum=args.base_momentum,
            max_momentum=args.max_momentum,
            div_factor=div_factor,
            final_div_factor=args.final_div_factor,
        )
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError(f"{args.scheduler} is not a supported scheduler.")

    return scheduler


def _save_model(model, model_dir, compound_coef, epoch, last_step, loss):
    date_time = datetime.datetime.now().strftime(_STRFTIME_FORMAT)
    filename = (
        f"{date_time}_efficientdet-d{compound_coef}_{epoch}_{last_step}_{loss:.4f}.pth"
    )
    model_path = os.path.join(model_dir, filename)
    if isinstance(model, CustomDataParallel):
        # multi gpu torch.nn.DataParallel model
        torch.save(model.module.model.state_dict(), model_path)
    else:
        # single gpu model
        torch.save(model.model.state_dict(), model_path)


def validate(model, loader, step, epoch, epochs, writer):
    model.eval()
    val_loss_cls_list = []
    val_loss_box_reg_list = []
    iter_time_list = []
    data_time_list = []
    loader_iter = iter(loader)
    with torch.no_grad():
        for batch_idx in range(len(loader)):
            iter_start_time = time.perf_counter()
            data_start_time = time.perf_counter()
            data = next(loader_iter)
            data_time_list.append(time.perf_counter() - data_start_time)
            imgs = data["img"]
            annotations = data["annot"]
            # if only one gpu, just send it to cuda:0
            # elif multiple gpus, send it to multiple gpus in
            # CustomDataParallel
            if num_gpus == 1:
                imgs = imgs.cuda()
                annotations = annotations.cuda()

            val_loss_cls, val_loss_box_reg = model(imgs, annotations)
            val_loss_cls = val_loss_cls.mean()
            val_loss_box_reg = val_loss_box_reg.mean()
            total_val_loss = val_loss_cls + val_loss_box_reg
            if total_val_loss == 0 or not torch.isfinite(total_val_loss):
                continue

            val_loss_cls_list.append(val_loss_cls.item())
            val_loss_box_reg_list.append(val_loss_box_reg.item())
            iter_time_list.append(time.perf_counter() - iter_start_time)

        val_loss_cls = np.mean(val_loss_cls_list)
        val_loss_box_reg = np.mean(val_loss_box_reg_list)
        total_val_loss = val_loss_cls + val_loss_box_reg
        iter_time = np.mean(iter_time_list)
        data_time = np.mean(data_time_list)
        date_time = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
        logger.info(
            f"[{date_time} valid]:  "
            f"epoch: {epoch + 1}/{epochs}  "
            f"val_loss_cls: {val_loss_cls:.4f}  "
            f"val_loss_box_reg: {val_loss_box_reg:.4f}  "
            f"total_val_loss: {total_val_loss:.4f}  "
            f"time: {iter_time:.4f}  "
            f"data_time {data_time:.4f}"
        )
        writer.add_scalar("loss/total_loss", total_val_loss, step)
        writer.add_scalar("loss/loss_cls", val_loss_cls, step)
        writer.add_scalar("loss/loss_box_reg", val_loss_box_reg, step)
        writer.add_scalar("time/time", iter_time, step)
        writer.add_scalar("time/data_time", data_time, step)
        writer.flush()
    return total_val_loss


def train(args):  # noqa: C901
    train_start_time = time.perf_counter()
    assert num_gpus > 0, "Found 0 cuda devices, CPU training is not supported."
    total_batch_size = args.batch_size * num_gpus
    assert total_batch_size % args.num_workers == 0, (
        f"batch_size * num_gpus ({total_batch_size}) must be divisible by num_workers "
        f"({args.num_workers})."
    )

    with open(os.path.join(args.model_dir, "hyperparameters.yml"), "w") as f:
        yaml.dump(vars(args), f)

    # initialization of tensorboard summary writers
    date_time = datetime.datetime.now().strftime(_STRFTIME_FORMAT)
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, f"logs/{date_time}"))
    train_writer = SummaryWriter(
        os.path.join(args.tensorboard_dir, f"logs/{date_time}/train")
    )
    val_writer = SummaryWriter(
        os.path.join(args.tensorboard_dir, f"logs/{date_time}/val")
    )

    # get weights path, selecting the best weights if weights == "best"
    weights_path = _get_weights_path(args.weights_dir, args.weights)
    # create the correct data structure splitting input data in train and val sets
    prepare_annotations(args.data_dir, args.classes, ["train", "val"])

    torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args)
    val_loader = _get_val_data_loader(args)

    model = EfficientDetBackbone(
        num_classes=len(args.classes),
        compound_coef=args.compound_coef,
        ratios=args.anchors_ratios,
        scales=args.anchors_scales,
    )
    _init_weights(model, weights_path)

    if args.freeze_backbone:
        logger.info("Freezing backbone")
        model.apply(_freeze_submodule_if_backbone)

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # use synchronized batch normalization when the batch size per gpu is too small
    if args.batch_size < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
        logger.info("Using Synchronized Batch Normalization")
    else:
        use_sync_bn = False

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model)
    model = model.cuda()

    if num_gpus > 1:
        # TODO: see if there are better way to parallelize
        model = CustomDataParallel(model, num_gpus)
        if use_sync_bn:
            patch_replication_callback(model)

    steps_per_epoch = len(train_loader)
    last_step, es_baseline = _get_last_step_and_es_baseline(
        weights_path, args.resume_training
    )
    es = EarlyStopping(
        args, baseline=es_baseline, best_epoch=last_step // steps_per_epoch - 1
    )
    optimizer = _get_optimizer(model, args)
    scheduler = _get_scheduler(optimizer, steps_per_epoch, args)
    model.train()
    logger.info(f"Starting training from step {last_step}")

    for epoch in range(args.epochs):
        if epoch in args.milestones:
            for group in optimizer.param_groups:
                if args.scheduler == "onecyclelr":
                    group["max_lr"] *= args.multisteplr_gamma
                    group["min_lr"] *= args.multisteplr_gamma
                else:
                    group["lr"] *= args.multisteplr_gamma

        last_epoch = last_step // steps_per_epoch
        if epoch < last_epoch:
            if scheduler is not None:
                for _ in range(steps_per_epoch):
                    scheduler.step()

            continue

        train_loader_iter = iter(train_loader)
        for batch_idx in range(steps_per_epoch):
            iter_start_time = time.perf_counter()
            data_start_time = time.perf_counter()
            data = next(train_loader_iter)
            data_time = time.perf_counter() - data_start_time
            if batch_idx < (last_step - last_epoch * steps_per_epoch):
                if scheduler is not None:
                    scheduler.step()

                continue

            imgs = data["img"]
            annotations = data["annot"]
            # if only one gpu, just send it to cuda:0 elif multiple gpus,
            # send it to multiple gpus in CustomDataParallel
            if num_gpus == 1:
                imgs = imgs.cuda()
                annotations = annotations.cuda()

            optimizer.zero_grad()
            loss_cls, loss_box_reg = model(imgs, annotations)
            loss_cls = loss_cls.mean()
            loss_box_reg = loss_box_reg.mean()
            total_loss = loss_cls + loss_box_reg
            if total_loss == 0 or not torch.isfinite(total_loss):
                continue

            total_loss.backward()
            if args.clip_gradients_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_gradients_norm
                )

            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            date_time = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
            eta = datetime.timedelta(
                seconds=round(time.perf_counter() - train_start_time)
            )
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            iter_time = time.perf_counter() - iter_start_time
            logger.info(
                f"[{date_time} train]:  "
                f"eta: {eta}  "
                f"epoch: {epoch + 1}/{args.epochs}  "
                f"batch: {batch_idx + 1}/{steps_per_epoch}  "
                f"loss_cls: {loss_cls.item():.4f}  "
                f"loss_box_reg: {loss_box_reg.item():.4f}  "
                f"total_loss: {total_loss.item():.4f}  "
                f"time: {iter_time:.4f}  "
                f"data_time: {data_time:.4f}  "
                f"lr: {lr:.6f}  "
                f"max_mem: {max_mem_mb:.0f}M"
            )
            writer.add_scalar("hp/lr", lr, last_step)
            if args.cycle_momentum:
                momentum = optimizer.param_groups[0]["momentum"]
                writer.add_scalar("hp/momentum", momentum, last_step)

            writer.add_scalar("usage/max_mem", max_mem_mb, last_step)
            writer.flush()

            train_writer.add_scalar("loss/total_loss", total_loss.item(), last_step)
            train_writer.add_scalar("loss/loss_cls", loss_cls.item(), last_step)
            train_writer.add_scalar("loss/loss_box_reg", loss_box_reg.item(), last_step)
            train_writer.add_scalar("time/time", iter_time, last_step)
            train_writer.add_scalar("time/data_time", data_time, last_step)
            train_writer.flush()

            last_step += 1

        if epoch % args.val_interval == 0 or epoch + 1 == args.epochs:
            total_val_loss = validate(
                model, val_loader, last_step - 1, epoch, args.epochs, val_writer
            )
            _save_model(
                model,
                args.checkpoints_dir,
                args.compound_coef,
                epoch,
                last_step,
                total_val_loss,
            )
            if es.step(epoch, total_val_loss):
                break

            model.train()

    model_params = {
        "classes": args.classes,
        "compound_coef": args.compound_coef,
        "anchors_scales": args.anchors_scales,
        "anchors_ratios": args.anchors_ratios,
    }
    with open(os.path.join(args.model_dir, "model_params.yml"), "w") as f:
        yaml.dump(model_params, f)

    writer.close()
    train_writer.close()
    val_writer.close()

    best_weights_path = _get_best_weights_path(args.checkpoints_dir)
    shutil.copyfile(best_weights_path, os.path.join(args.model_dir, "model.pth"))

    evaluate(
        args.model_dir,
        args.data_dir,
        eval_set="val",
        threshold=args.eval_threshold,
        nms_threshold=args.eval_nms_threshold,
        max_imgs=args.eval_max_imgs,
        use_float16=args.use_float16,
        device=args.eval_device,
    )


if __name__ == "__main__":
    args, _ = _parse_args()
    train(args)
