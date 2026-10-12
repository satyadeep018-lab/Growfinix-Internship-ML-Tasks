import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

CLASS_NAMES = ["Defective", "Perfect"]
