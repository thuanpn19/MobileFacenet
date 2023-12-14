import os
import pickle
import tarfile
import time

import cv2 as cv
import cv2
import numpy as np
import scipy.stats
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import device
from data_gen import data_transforms
# from utils import align_face, get_central_face_attributes, get_all_face_attributes, draw_bboxes, ensure_folder

transformer = data_transforms['val']

def transform(img, flip=False):
    if flip:
        img = cv.flip(img, 1)
    img = img[..., ::-1]  # RGB
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    img = img.to(device)
    return img


def get_feature(model, img):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float, device=device)
    imgs[0] = transform(img.copy(), False)
    imgs[1] = transform(img.copy(), True)
    with torch.no_grad():
        output = model(imgs)
    feature_0 = output[0].cpu().numpy()
    feature_1 = output[1].cpu().numpy()
    feature = feature_0 + feature_1
    return feature / np.linalg.norm(feature)

# def get_feature(model, img):
#     imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float, device=device)
#     imgs[0] = transform(img.copy(), False)
#     with torch.no_grad():
#         output = model(imgs)
#     feature_0 = output[0].cpu().numpy()
#     feature = feature_0
#     return feature / np.linalg.norm(feature)

def adjust_bounding_box(box):
    x, y, w, h = box

    if h > w:
        diff = h - w
        w = h
        x -= int(diff // 2)  
        x = max(x, 0)
        w = h

    return x, y, w, h

def align_face(image, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Tính góc nghiêng giữa 2 mắt
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Lấy kích thước ảnh
    height, width = image.shape[:2]

    # Tính ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Áp dụng phép biến đổi xoay cho ảnh
    aligned_face = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

    return aligned_face

def load_model():
    scripted_model_file = 'mobilefacenet_scripted.pt'
    model = torch.jit.load(scripted_model_file)
    model = model.to(device)
    model.eval()
    return model


def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity