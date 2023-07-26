from .noise_utils import Noise as n

class_dict = {
    'airplane': 0, 'car': 1, 'bird': 2, 'cat': 3,
    'elephant': 4, 'dog': 5, 'bottle': 6, 'knife': 7,
    'truck': 8, 'boat': 9
}

style_dict = {
    'candy': 0, 'mosaic_ducks_massimo': 1, 'pencil': 2, 'seated-nude': 3,
    'shipwreck': 4, 'starry_night': 5, 'stars2': 6, 'strip': 7,
    'the_scream': 8, 'wave': 9
}

noise_dict = {
    n.pixelate: 0, n.gaussian_blur: 1, n.contrast: 2, n.speckle_noise: 3,
    n.brightness: 4, n.defocus_blur: 5, n.saturate: 6, n.gaussian_noise: 7,
    n.impulse_noise: 8, n.shot_noise: 9
}

novel_dict = {
    'fa1': 0, 'fa2': 1, 'fb1': 2, 'fb3': 3,
    'fc1': 4
}

seed = 1235
alt_seed = 1234

data_dir = '/home/user/Documents/data'
weights_dir = '/home/user/Documents/models/weights'

inet_classes = [i for i in range(900)]
