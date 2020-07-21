import json
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def prepare_annotations(data_dir, classes, parts):
    """Prepare the annotations filtering unneeded classes, removing image entries
    without associated bounding boxes or image files.
    """
    # annotations dir
    anns_dir = os.path.join(data_dir, "annotations")
    for part in parts:
        json_path = os.path.join(anns_dir, f"instances_{part}.json")

        with open(json_path) as f:
            instances = json.load(f)

        images = instances["images"]
        annotations = instances["annotations"]
        categories = instances["categories"]

        # discard all unwanted classes and replace category ids with incremental numbers
        categories, mapping_id_categories = _filter_categories(categories, classes)
        annotations = _filter_annotations(annotations, mapping_id_categories)
        images, annotations = _clean_annotations(
            images, annotations, os.path.join(data_dir, part)
        )

        with open(json_path, "w") as f:
            json.dump(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                f,
            )


def _clean_annotations(images, annotations, images_dir):
    present_images = os.listdir(images_dir)
    clean_images = []
    clean_image_ids = []
    for img in images:
        if img["file_name"] in present_images:
            clean_images.append(img)
            clean_image_ids.append(img["id"])

    clean_annotations = [
        ann for ann in annotations if ann["image_id"] in clean_image_ids
    ]

    return clean_images, clean_annotations


def _filter_categories(categories, keep_list):
    """Convert categories id and keep only those specified in keep_list.

    Arguments:
        categories (list): categories section of COCO json.
        keep_list (list): list of categories to be kept.

    Returns:
        The list of modified categories and a dict mapping old to new category id.
    """

    # map: name - id_new
    mapping = {c: n + 1 for n, c in enumerate(sorted(keep_list))}
    # inverse_mapping = {n + 1: c for n, c in enumerate(sorted(keep_list))}

    # map: name - id_old
    old_mapping = {
        c["name"].upper(): c["id"] for c in categories if c["name"].upper() in mapping
    }
    # map: id_old - id_new
    mapping_id = {old_mapping[k]: v for k, v in mapping.items()}

    categories_mod = []
    for category in categories:
        category_mod = category.copy()
        category_mod["name"] = category_mod["name"].upper()
        if category_mod["name"] in mapping:
            category_mod["id"] = mapping[category_mod["name"]]
            categories_mod.append(category_mod)
    categories_mod = sorted(categories_mod, key=lambda c: c["id"])

    return categories_mod, mapping_id


def _filter_annotations(annotations, mapping):
    """Convert category ids in annotations and keep only the specified classes.

    Arguments:
        annotations (list): annotations section of COCO json.
        mapping (dict): mapping from old to new category id.

    Returns:
        The filtered annotations.
    """

    annotations_mod = []
    counter = 1
    for annotation in annotations:
        annotation_mod = annotation.copy()
        if annotation["category_id"] in mapping:
            annotation_mod["id"] = counter
            annotation_mod["category_id"] = mapping[annotation["category_id"]]
            annotations_mod.append(annotation_mod)
            counter += 1

    return annotations_mod
