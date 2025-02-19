# Import the required libraries
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

# Read the image
image = plt.imread('penguins.jpg')
#image = image.copy()

# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
tensor = transform(image)

# Print the converted image tensor
print(tensor)