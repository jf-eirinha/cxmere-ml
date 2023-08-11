from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import cv2
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(
    description=("Run mask generation for an image using SAM.")
)

parser.add_argument(
    "--model-type",
    type=str,
    required=False,
    default="vit_b",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b'].",
)

parser.add_argument(
    "--image", type=str, required=True, help="The path to the image to segment."
)


CHECKPOINT_PATH = {
    "vit_h": "./checkpoints/sam_vit_h_4b8939.pth",
    "vit_l": "./checkpoints/sam_vit_l_0b3195.pth",
    "vit_b": "./checkpoints/sam_vit_b_01ec64.pth",
}


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    checkpoint = CHECKPOINT_PATH[args.model_type]
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Loading image...")
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Generating masks...")
    masks = mask_generator.generate(image)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    show_anns(masks)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
