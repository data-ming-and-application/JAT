import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

def euclidean(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real - distribution_predict) ** 2, 1))) / height


def sorensen(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = np.sum(np.abs(distribution_real - distribution_predict), 1)
    denominator = np.sum(distribution_real + distribution_predict, 1)
    return np.sum(numerator / denominator) / height


def squared_chi2(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (distribution_real - distribution_predict) ** 2
    denominator = distribution_real + distribution_predict
    return np.sum(numerator / denominator) / height


def kl(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(distribution_real * np.log(distribution_real / distribution_predict)) / height


def intersection(distribution_real, distribution_predict):
    height, width = distribution_real.shape
    inter = 0.
    for i in range(height):
        for j in range(width):
            inter += np.min([distribution_real[i][j], distribution_predict[i][j]])
    return inter / height


def fidelity(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(distribution_real * distribution_predict)) / height


def chebyshev(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.max(np.abs(distribution_real-distribution_predict), 1)) / height


def clark(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real-distribution_predict)**2 / (distribution_real+distribution_predict)**2, 1))) / height


def canberra(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.abs(distribution_real-distribution_predict) / (distribution_real+distribution_predict)) / height


def cosine(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sum(distribution_real*distribution_predict, 1) / (np.sqrt(np.sum(distribution_real**2, 1)) *\
           np.sqrt(np.sum(distribution_predict**2, 1)))) / height


def squared_chord(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    numerator = (np.sqrt(distribution_real) - np.sqrt(distribution_predict)) ** 2
    denominator = np.sum(numerator)
    return denominator / height

def euclidean_t_single(distribution_real, distribution_predict):
            
    height = distribution_real.shape[0]
    return torch.sqrt(torch.sum((distribution_real - distribution_predict) ** 2, 1))

def kl_t_single(distribution_real, distribution_predict):
    
    height = distribution_real.shape[0]
    #print("1 : ", distribution_real[0]);  print("2 : ", distribution_predict[0])
    return torch.sum(distribution_real * torch.log(distribution_real / distribution_predict), 1)


def euclidean_t(distribution_real, distribution_predict):
            
    height = distribution_real.shape[0]
    return torch.sum(torch.sqrt(torch.sum((distribution_real - distribution_predict) ** 2, 1))) / height

def kl_t(distribution_real, distribution_predict):
    
    height = distribution_real.shape[0]
    #print("1 : ", distribution_real[0]);  print("2 : ", distribution_predict[0])
    return torch.sum(distribution_real * torch.log(distribution_real / distribution_predict)) / height

def chebyshev_t(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    #print(torch.abs(distribution_real-distribution_predict))
    #print(torch.max(torch.abs(distribution_real-distribution_predict),dim=1).values,type(torch.max(torch.abs(distribution_real-distribution_predict))))
    return torch.sum(torch.max(torch.abs(distribution_real-distribution_predict), 1).values) / height


def intersection_t(distribution_real, distribution_predict):
    height, width = distribution_real.shape
    inter = 0.
    for i in range(height):
        for j in range(width):
            inter += torch.min(distribution_real[i][j], distribution_predict[i][j])
    return inter / height

def clark_t(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return torch.sum(torch.sqrt(torch.sum((distribution_real-distribution_predict)**2 / (distribution_real+distribution_predict)**2, 1))) / height


def canberra_t(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return torch.sum(torch.abs(distribution_real-distribution_predict) / (distribution_real+distribution_predict)) / height


def cosine_t(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return torch.sum(torch.sum(distribution_real*distribution_predict, 1) / (torch.sqrt(torch.sum(distribution_real**2, 1)) *\
           torch.sqrt(torch.sum(distribution_predict**2, 1)))) / height