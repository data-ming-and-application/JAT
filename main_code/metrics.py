import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn

def cal_mrr(y_true, click_rates):
    combined_data = list(zip(y_true, click_rates))
    sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

    reciprocal_ranks = []
    for i, (label, _) in enumerate(sorted_data):
        if label == 1:
            reciprocal_ranks.append(1 / (i + 1))

    mrr = np.mean(reciprocal_ranks)
    return mrr

def cal_auc1(y_true, y_pred):
    n_bins = 10
    postive_len = sum(y_true)  # M正样本个数
    negative_len = len(y_true) - postive_len  # N负样本个数
    total_case = postive_len * negative_len  # M * N样本对数
    pos_histogram = [0 for _ in range(n_bins)]  # 保存每一个概率值下的正样本个数
    neg_histogram = [0 for _ in range(n_bins)]  # 保存每一个概率值下的负样本个数
    bin_width = 1.0 / n_bins
    for i in range(len(y_true)):
        nth_bin = int(y_pred[i] / bin_width)  # 概率值转化为整数下标
        if y_true[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    # print(pos_histogram)
    # print(neg_histogram)
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        # print(pos_histogram[i], neg_histogram[i], accumulated_neg, satisfied_pair)
        accumulated_neg += neg_histogram[i]

    return satisfied_pair / float(total_case)

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


def compute_dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)
def compute_ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int):
    best = compute_dcg_score(y_true, y_true, k)
    actual = compute_dcg_score(y_true, y_score, k)
    return actual / best
def ndcg(score, label, a, b, c, d, e):
    num_df = len(score)
    ranks = np.zeros(num_df)
    scores = []
    recall_k = [a, b, c, d, e]
    # recall_k = [6]
    for i in range(num_df):
        target_line = np.array(label[i])
        yscore_line = np.array(score[i])
        true_index = np.where(target_line == 1)[0][0]
        inds = np.argsort(yscore_line)[::-1]
        ranks[i] = np.where(inds == true_index)[0][0]
    for k in recall_k:
        list_score = [compute_ndcg_score(y_true=np.array(label[i]), y_score=np.array(score[i]), k=k)for i in range(num_df)]
        scores.append(np.mean(list_score))
    return scores
import random
def recall_ndcg(user, jd):
    score = np.dot(user, jd.T)
    score_diag = np.diagonal(score)
    random.seed(100)

    index = []
    for i in range(len(user)):
        temp = []
        for j in score[i]:
            if j not in temp and j != score_diag[i]:
                temp.append(j)
        temp = random.sample(temp, 200)
        temp.append(score_diag[i])
        temp = sorted(temp, reverse=True)
        ids = temp.index(score_diag[i])
        index.append(ids)

    r20 = len(np.where(np.array(index) < 20)[0]) / len(index)
    r40 = len(np.where(np.array(index) < 40)[0]) / len(index)
    r60 = len(np.where(np.array(index) < 60)[0]) / len(index)
    r80 = len(np.where(np.array(index) < 80)[0]) / len(index)
    r100 = len(np.where(np.array(index) < 100)[0]) / len(index)

    mask_ = list(0 for i in range(200))
    mask_.append(1)
    results_ = []
    masks_ = []

    for i in range(len(user)):
        temp = []
        for j in score[i]:
            if j not in temp and j != score_diag[i]:
                temp.append(j)
        temp = random.sample(temp, 200)
        temp.append(score_diag[i])
        mask = mask_
        results_.append(temp)
        masks_.append(mask)

    results = np.array(results_)
    masks = np.array(masks_)
    NDCG = ndcg(results, masks, 20, 40, 60, 80, 100)

    return r20,r40,r60,r80,r100,NDCG