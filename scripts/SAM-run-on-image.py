from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import cv2

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


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    checkpoint = CHECKPOINT_PATH[args.model_type]
    sam = sam_model_registry[args.model_type](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Loading image...")
    image = cv2.imread(args.image)

    print("Generating masks...")
    masks = mask_generator.generate(image)

    print("Masks generated!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
