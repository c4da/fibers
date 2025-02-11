import cv2

import common.tools as common


def run(image_path) -> cv2.UMat:

    img = common.read_image(image_path)

    img = common.crop_image(img)

    img = common.filter_noise(img)

    # img = common.contrast_enhancement(img)
    img = common.contrast_enhancement_adapthist(img)

    return img