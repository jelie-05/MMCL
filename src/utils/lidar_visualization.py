import cv2
import numpy as np


def print_projection_plt(points, image):
    """ project converted velodyne points into camera image """

    # Convert PyTorch tensor to NumPy array
    image_np = image.numpy()

    # Convert from CHW (Channel, Height, Width) to HWC (Height, Width, Channel) format
    image_np = np.transpose(image_np, (1, 2, 0))

    # Convert from float to uint8 (necessary for OpenCV)
    image_np = (image_np * 255).astype(np.uint8)

    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    # for i in range(points.shape[1]):
    #     cv2.circle(hsv_image, (np.int32(points[0][i]) ,np.int32(points[1][i])) ,2, (int(color[i]) ,255 ,255) ,-1)

    for i, x in enumerate(points[0]):
        for j, y in enumerate(x):
            if y == 0:
                pass
            else:
                hsv_image = cv2.circle(hsv_image, (j, i), 2, (int(y*100), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

