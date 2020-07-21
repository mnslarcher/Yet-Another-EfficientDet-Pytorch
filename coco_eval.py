import argparse
from contextlib import redirect_stdout
import io
import json
import logging
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
from tqdm import tqdm
import yaml

import torch

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

from prepare_data import prepare_annotations

_INPUT_SIZES = (512, 640, 768, 896, 1024, 1280, 1280, 1536)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _parse_args():
    parser = argparse.ArgumentParser("EfficientDet evaluation script.")
    parser.add_argument(
        "--model-dir",
        type=str,
        metavar="N",
        help="Path to the local directory of the trained model.",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        metavar="N",
        required=True,
        help="Path to the local dir where the data are downloaded.",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        metavar="N",
        default="test",
        help="Name of the evaluation set, must match the name of the folder where the "
        "images are saved. Default: 'test'.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        metavar="N",
        help="Score threshold. Default: 0.05.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        metavar="N",
        help="Non Maximum Suppression threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--use-float16",
        type=lambda x: True if x.lower() == "true" else False,
        default=False,
        metavar="N",
        help="If to use half precision. Default: False.",
    )
    parser.add_argument(
        "--max-imgs",
        type=int,
        default=10000,
        metavar="N",
        help="Maximum number of images considered. Default: 10000.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="N",
        help="Device index to select. Default: 0.",
    )
    return parser.parse_known_args()


def generate_res_file(
    imgs_dir,
    model_dir,
    annotation_file,
    res_file,
    max_size,
    eval_set="test",
    threshold=0.05,
    nms_threshold=0.5,
    max_imgs=10000,
    use_float16=False,
    device=0,
):
    use_cuda = torch.cuda.is_available()

    coco_gt = COCO(annotation_file)

    model = model_fn(model_dir)
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(device)
        if use_float16:
            model.half()
        else:
            model.float()
    else:
        model.float()

    results = []

    regress_boxes = BBoxTransform()
    clip_boxes = ClipBoxes()
    img_ids = coco_gt.getImgIds()[:max_imgs]

    for img_id in tqdm(img_ids):
        img = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(imgs_dir, img["file_name"])

        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=max_size)
        x = torch.from_numpy(framed_imgs[0])
        if use_cuda:
            x = x.cuda(device)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        _, regression, classification, anchors = model(x)

        predictions = postprocess(
            x,
            anchors,
            regression,
            classification,
            regress_boxes,
            clip_boxes,
            threshold,
            nms_threshold,
        )

        if not predictions:
            continue

        predictions = invert_affine(framed_metas, predictions)[0]

        scores = predictions["scores"]
        class_ids = predictions["class_ids"]
        rois = predictions["rois"]

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            for roi_id in range(rois.shape[0]):
                score = float(scores[roi_id])
                category_id = int(class_ids[roi_id]) + 1
                bbox = rois[roi_id, :]
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": category_id,
                        "score": score,
                        "bbox": bbox.tolist(),
                    }
                )

    if not len(results):
        raise Exception(
            "The model does not provide any valid output, check model architecture and "
            "input data."
        )

    # write output
    with open(res_file, "w") as f:
        json.dump(results, f)

    return img_ids


def model_fn(model_dir):
    with open(os.path.join(model_dir, "hyperparameters.yml")) as f:
        hps = yaml.load(f, yaml.FullLoader)

    model = EfficientDetBackbone(
        compound_coef=hps["compound_coef"],
        num_classes=len(hps["classes"]),
        ratios=hps["anchors_ratios"],
        scales=hps["anchors_scales"],
    )

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        # without map_location=torch.device('cpu'), the weights are loaded to
        # default tensor device which was assigned when the weights were saved, such as
        # 'cuda:0', instead of forcing to be loaded into cpu memory first.
        # most of the time it works without map_location=torch.device('cpu'),
        # but in case one run coco_eval on a cpu-only server, it might fail
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

    return model


def coco_eval(annotation_file, res_file, img_ids=None):
    # load coco ground truth
    coco_gt = COCO(annotation_file)
    # load coco detection results
    coco_dt = coco_gt.loadRes(res_file)
    # run coco evaluation
    ce = COCOeval(coco_gt, coco_dt, "bbox")
    if img_ids is not None:
        ce.params.imgIds = img_ids
    ce.evaluate()
    ce.accumulate()
    ce.summarize()


def evaluate(
    model_dir,
    data_dir,
    eval_set="test",
    threshold=0.05,
    nms_threshold=0.5,
    max_imgs=10000,
    use_float16=False,
    device=0,
):
    with open(os.path.join(model_dir, "hyperparameters.yml")) as f:
        hps = yaml.load(f, yaml.FullLoader)

    imgs_dir = os.path.join(data_dir, eval_set)
    annotation_file = os.path.join(
        data_dir, "annotations", f"instances_{eval_set}.json"
    )
    res_file = os.path.join(
        model_dir,
        f"coco_res_{eval_set}_th_{threshold}_nms_th_{nms_threshold}"
        f"_f16_{str(use_float16).lower()}.json",
    )
    eval_file = os.path.join(
        model_dir,
        f"coco_eval_{eval_set}_th_{threshold}_nms_th_{nms_threshold}"
        f"_f16_{str(use_float16).lower()}.txt",
    )
    prepare_annotations(data_dir, hps["classes"], [eval_set])
    img_ids = generate_res_file(
        imgs_dir,
        model_dir,
        annotation_file,
        res_file,
        max_size=_INPUT_SIZES[hps["compound_coef"]],
        eval_set=eval_set,
        threshold=threshold,
        nms_threshold=nms_threshold,
        max_imgs=max_imgs,
        use_float16=use_float16,
        device=device,
    )

    f = io.StringIO()
    with redirect_stdout(f):
        coco_eval(annotation_file, res_file, img_ids)

    out = "\n".join(f.getvalue().split("\n")[13:])
    logger.info("\n" + out)

    with open(eval_file, "w") as f:
        f.write(out)


if __name__ == "__main__":
    args, _ = _parse_args()
    evaluate(**vars(args))
