import numpy as np
import cv2
from PIL import Image
from typing import Optional, Callable, List

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Tint(object):
	"""
	Tint images with a given color; if not defined, turn to grayscale

	Args:
		color (str): Given color
	"""

	def __init__(self, color):
		assert isinstance(color, str)
		self.color = color

	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be tinted

		Returns:
			PIL Image: Tinted image
		"""

		if self.color is not 'grayscale':
			np_img = np.asarray(img.convert('RGB'))
			cv2_img = np_img[:, :, ::-1].copy()

			b, g, r = cv2.split(cv2_img)

			if self.color == 'red':
				val_b, val_g, val_r = 0, 0, 1  # binary - 1
			elif self.color == 'blue':
				val_b, val_g, val_r = 1, 0, 0  # binary - 4
			elif self.color == 'green':
				val_b, val_g, val_r = 0, 1, 0  # binary - 2
			elif self.color == 'yellow':
				val_b, val_g, val_r = 0, 1, 1  # binary - 3
			elif self.color == 'violet':
				val_b, val_g, val_r = 1, 0, 1  # binary - 5

			np.multiply(b, val_b, out=b, casting='unsafe')
			np.multiply(g, val_g, out=g, casting='unsafe')
			np.multiply(r, val_r, out=r, casting='unsafe')
			cv2_img = cv2.merge([b, g, r])

			cv2_img = cv2_img[:, :, ::-1].copy()

			return Image.fromarray(cv2_img)
		else:
			np_img = np.asarray(img.convert('L'))
			np_img = np.stack((np_img,)*3, axis=-1)

			return Image.fromarray(np_img)


class Grayscale(object):
	def __init__(self):
		pass

	@staticmethod
	def mnist_transform():
		transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize((0.1307,), (0.3081,)),
				transforms.Lambda(lambda x: x.expand(3, -1, -1))
			]
		)

		return transform

	@staticmethod
	def fmnist_transform():
		transform = transforms.Compose(
			[
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,)),
				transforms.Lambda(lambda x: x.expand(3, -1, -1))
			]
		)

		return transform


