from io import BytesIO
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
import cv2
from PIL import Image as PILImage


class Noise:
    def __init__(self):
        pass

    @staticmethod
    def pixelate(x, severity=2):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

        x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
        x = x.resize((224, 224), PILImage.BOX)

        return x

    @staticmethod
    def contrast(x, severity=2):
        c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

        x = np.array(x) / 255
        means = np.mean(x, axis=(0, 1), keepdims=True)

        return PILImage.fromarray(
            np.uint8(np.clip((x - means) * c + means, 0, 1) * 255)
        )

    @staticmethod
    def brightness(x, severity=2):
        c = [.1, .2, .3, .4, .5][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return PILImage.fromarray(
            np.uint8(np.clip(x, 0, 1) * 255)
        )

    @staticmethod
    def saturate(x, severity=2):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        return PILImage.fromarray(
            np.uint8(np.clip(x, 0, 1) * 255)
        )

    @staticmethod
    def impulse_noise(x, severity=2):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return PILImage.fromarray(
            np.uint8(np.clip(x, 0, 1) * 255)
        )

    @staticmethod
    def shot_noise(x, severity=2):
        c = [60, 25, 12, 5, 3][severity - 1]

        x = np.array(x) / 255.
        return PILImage.fromarray(
            np.uint8(np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255)
        )

    @staticmethod
    def gaussian_noise(x, severity=2):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(x) / 255.
        return PILImage.fromarray(
            np.uint8(np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255)
        )

    @staticmethod
    def defocus_blur(x, severity=2):
        def disk(radius, alias_blur=0.1, dtype=np.float32):
            if radius <= 8:
                L = np.arange(-8, 8 + 1)
                ksize = (3, 3)
            else:
                L = np.arange(-radius, radius + 1)
                ksize = (5, 5)
            X, Y = np.meshgrid(L, L)
            aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
            aliased_disk /= np.sum(aliased_disk)

            # super sample disk to anti-alias
            return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return PILImage.fromarray(
            np.uint8(np.clip(channels, 0, 1) * 255)
        )

    @staticmethod
    def speckle_noise(x, severity=2):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

        x = np.array(x) / 255
        return PILImage.fromarray(
            np.uint8(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255)
        )

    @staticmethod
    def gaussian_blur(x, severity=2):
        c = [1, 2, 3, 4, 6][severity - 1]

        x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
        return PILImage.fromarray(
            np.uint8(np.clip(x, 0, 1) * 255)
        )
