import sys

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn1
from JAT1 import CoKEModel
from metrics import *
import os
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import time
from sklearn.metrics import roc_auc_score


# 预热学习
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# --------------Loda Datasets-------------
class TextDataset(Data.Dataset):
    def __init__(self, split_dataset):
        self.train_captions = []
        self.train_lengths = []
        self.phrase_lengths = []
        self.all_captions = []

        # with open(r"./jingpai/{}_lengths_.txt".format(split_dataset), 'r') as f:
        with open(r"./new_jp/{}_lengths_new.txt".format(split_dataset), 'r') as f:
            for line in f:
                # record = line.strip().split('\t')
                record = line.strip().split('    ')
                # 定义每条句子包含100条短语，每个短语长度100
                # 定义一个JD有100个item，每个item长度是50？
                # 新版数据定义一个JD有50个item，每个item长度是100
                phrase = np.zeros((100, 50))
                # all_phrase = np.zeros((300))
                amount = 0
                phrase_len = []
                for rows in range(len(record)):
                    if record[rows] == [''] or len(record[rows]) == 0:
                        continue
                    record_split = record[rows].split(',')
                    phrase_len.append(len(record_split))
                    for columns in range(len(record_split)):
                        a = int(record_split[columns])
                        if a > 15000:
                            a = 0
                        phrase[rows][columns] = a
                        # all_phrase[amount] = int(record_split[columns])
                        # amount += 1
                self.train_captions.append(phrase)
                self.train_lengths.append(phrase_len)
                # self.all_captions.append(all_phrase)

        # self.train_label = np.load(r"./jingpai/match_matrix_{}.npy".format(split_dataset))
        self.train_label = np.load(r"./new_jp/match_matrix_{}_new.npy".format(split_dataset))
        # self.user_feature = np.load(r"./jingpai/jingpai_user_{}.npy".format(split_dataset))
        self.user_feature = np.load(r"./new_jp/jingpai_user_{}_new.npy".format(split_dataset))

        # if split_dataset == 'train':
        #     self.label_similarity = 0.7*self.train_label + 0.3*1.0/204

        self.length = len(self.train_captions)
        # if split_dataset == "test":
        # self.length = 500
        print("Data Description : ", len(self.train_captions), len(self.train_label), len(self.user_feature))

    def __getitem__(self, index):
        # handle the image redundancy
        caption = self.train_captions[index]

        '''if index + 3 >= self.length:
            all_caption = np.vstack((self.train_captions[index -3],self.train_captions[index -2],self.train_captions[index -1]))
        else:
            all_caption = np.vstack((self.train_captions[index +1],self.train_captions[index +2],self.train_captions[index +3]))'''

        '''if index + 2 >= self.length:
            all_caption = np.vstack((self.train_captions[index -2],self.train_captions[index -1]))
        else:
            all_caption = np.vstack((self.train_captions[index +1],self.train_captions[index +2]))'''

        if index + 1 >= self.length:
            all_caption = self.train_captions[index - 1]
        else:
            all_caption = self.train_captions[index + 1]
        captions = np.vstack((caption, all_caption))
        target = torch.LongTensor(captions)
        label = torch.tensor(self.train_label[index])
        # length = torch.Tensor(self.train_lengths[index])
        user_embed = torch.Tensor(self.user_feature[index])

        # target：2个样本
        # index：当前样本索引
        # label：当前样本标签，是否点击
        # user_embed：用户特征
        return target, index, label, user_embed

    def __len__(self):
        return self.length - 1


def collate_fn(data):
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    # captions, ids, labels = zip(*data)
    # labels = torch.stack(labels, 0)
    # targets = torch.stack(captions, 0)
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, ids, labels, users = zip(*data)
    labels = torch.stack(labels, 0)
    targets = torch.stack(captions, 0)
    users = torch.stack(users, 0)
    '''# Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), 300).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]'''

    # 2个JD,是否点击，用户特征
    return targets, labels, ids, users


# definition super parameters
batch_size = 32
num_workers = 0
shuffle = True
voc_size = 12525
emb_size = 512
nhead = 8
nhid = 1024
nlayers = 4
dropout = 0.1
lr = 0.00001
weight_decay = 0.1
warmup_steps = 10
t_total = 50
sava_period = 2
save_dir = 'logss'

train_dataset = TextDataset("train")
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               collate_fn=collate_fn, drop_last=True)
test_dataset = TextDataset("test")
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              collate_fn=collate_fn, drop_last=True)

model = CoKEModel(voc_size, emb_size, nhead, nhid, nlayers, dropout, batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=t_total)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)
model = model.to(device)
print(model.load_state_dict(torch.load('./logs/ep038.pth', map_location=device), strict=False))

freeze = False
if freeze:
    print("冻结训练！")
    model.freeze_train()

alpha = 0.3
max_auc = 0
print("-----------------------------------------------开始训练！--------------------------------------------------")

start_time_total = time.time()
for epoch in range(t_total):
    # print("***"*5,"train","***"*5)
    for i, (train_text, train_label, ids, train_user_feature) in enumerate(train_loader):
        # -----------------------------------------------------------------------------------
        # train_text:(32,200,50)  32个样本，每个样本（200，50）包含当前的样本和它之后的样本
        # train_label:(32,)
        # train_user_feature:(32,204)
        # -----------------------------------------------------------------------------------
        train_text = train_text.to(device)
        train_label = train_label.to(device)
        train_user_feature = train_user_feature.to(device)
        # 对用户特征分布做了一次线性变换
        train_user_feature = (1 - alpha) * train_user_feature + alpha * 1.0 / 204

        # JD和用户特征
        all_feature, attention = model(train_text, train_user_feature)  # (32,2)

        # -------------------------------------------loss--------------------------------------------
        # 交叉熵损失
        # cross_entropy =nn.CrossEntropyLoss()(all_feature, train_label.long())
        cross_entropy = nn.BCELoss()(all_feature.squeeze(), train_label.float())
        loss = cross_entropy
        # -------------------------------------------------------------------------------------------

        print("打印损失值：")
        print("epoch : ", epoch, "itera : ", i, " loss : ", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler_warmup.step()  # Update learning rate schedule

        # if epoch < 2: # 如果epoch<18的话，后续代码不会执行
        #     continue

        if i % 100 == 0 and i != 0:
            print("开始验证：")
            with torch.no_grad():
                jp_pre = np.zeros((2816, 2)) - 1
                jp_pres = np.zeros((2816,)) - 1
                jp_labels = np.zeros((2816,)) - 1
                for j, (test_text, test_label, ids, test_user_feature) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        test_text = test_text.cuda()
                        test_label = test_label.cuda()
                        test_user_feature = test_user_feature.cuda()
                    test_user_feature = (1 - alpha) * test_user_feature + alpha * 1.0 / 204

                    # all_feature：(32,2)
                    all_feature_, attention_ = model(test_text, test_user_feature)

                    # ------------------------------------------------------
                    # all_feature_ = nn.Softmax(dim=-1)(all_feature_)
                    ids = list(ids)
                    jp_labels[ids] = test_label.detach().cpu().numpy()
                    jp_pre[ids] = all_feature_.detach().cpu().numpy()
                    for k in range(len(jp_pre)):
                        jp_pres[k] = jp_pre[k][1]
                    # ------------------------------------------------------

                split_num = np.load('./new_jp/spilt_num.npy')
                split_num = list(split_num)
                prediction = []
                label = []
                start_index = 0
                for size in split_num:
                    group1 = jp_labels[start_index:start_index + size]
                    label.append(group1)
                    start_index += size
                start_index = 0
                for size in split_num:
                    group2 = jp_pres[start_index:start_index + size]
                    prediction.append(group2)
                    start_index += size

                mrr_list = []  # 用于存储每个用户的MRR值
                auc_list = []  # 用于存储每个用户的AUC值

                # 计算AUC
                for a1 in range(len(prediction)):
                    auc = roc_auc_score(label[a1], prediction[a1])
                    auc_list.append(auc)

                # 计算每个用户的MRR和AUC
                for ii in range(len(prediction)):
                    sorted_indices = sorted(range(len(prediction[ii])), key=lambda kk: prediction[ii][kk], reverse=True)
                    mrr = 0.0
                    for jj, index_ in enumerate(sorted_indices):
                        if label[ii][index_] == 1:
                            mrr = 1 / (jj + 1)
                            # break
                            mrr_list.append(mrr)
                    # mrr_list.append(mrr)

                    # # 计算AUC
                    # auc = roc_auc_score(label[i], prediction[i])
                    # auc_list.append(auc)

                # 计算平均MRR和平均AUC
                avg_mrr = sum(mrr_list) / len(mrr_list)
                avg_auc = sum(auc_list) / len(auc_list)
                # -----------------------------------------------------
                # # auc = cal_auc1(jp_labels, jp_pres)
                # auc = roc_auc_score(jp_labels, jp_pres)
                # # mrr = cal_mrr(jp_labels, jp_pres)
                # mrr = 0.0
                # num_users = 0
                # for i in range(len(jp_labels)):
                #     if jp_labels[i]==1:
                #         mrr += 1/(i+1)
                #         num_users += 1
                # if num_users > 0:
                #     mrr /=num_users
                # -----------------------------------------------------
                print("best_epoch : ", epoch)
                print("iteration : ", i)
                print("auc：", avg_auc)
                print("mrr：", avg_mrr)
                if avg_auc > max_auc:
                    with open("jingpai_721.txt", 'w') as f:
                        strs = ''
                        strs = "auc : " + str(avg_auc) + '\n'
                        strs += "mrr : " + str(avg_mrr) + '\n'
                        strs += "best_epoch : " + str(epoch) + '\n'
                        strs += "iteration : " + str(i) + '\n'
                        f.write(strs)
                    max_auc = avg_auc

        # if (epoch+1)%sava_period==0 or epoch+1==t_total:
        #     torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d.pth" % (epoch + 1)))

end_time_total = time.time()
print("train total cost time : ", end_time_total - start_time_total)
