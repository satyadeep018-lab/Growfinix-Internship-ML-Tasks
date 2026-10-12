import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_data_loaders, CLASS_NAMES
from model import DefectResNet
