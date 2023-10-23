import sys

import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn1
from JAT import CoKEModel
from metrics import *
import os
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time


#预热学习
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
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
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



#--------------Loda Datasets-------------
class TextDataset(Data.Dataset):
    def __init__(self,split_dataset):
        self.train_captions = []
        self.train_lengths = []
        self.phrase_lengths = []
        self.all_captions = []

        # with open(r"./split_phrase/{}_lengths.txt".format(split_dataset),'r') as f:
        with open(r"./new_data_/{}_lengths_.txt".format(split_dataset),'r') as f:
            for line in f:
                # record = line.strip().split('\t')
                record = line.strip().split('    ')
                #定义每条句子包含100条短语，每个短语长度100
                # 定义一个JD有100个item，每个item长度是50？
                # 新版数据定义一个JD有50个item，每个item长度是100
                phrase = np.zeros((100, 50))
                #all_phrase = np.zeros((300))
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
                        #all_phrase[amount] = int(record_split[columns])
                        #amount += 1
                self.train_captions.append(phrase)
                self.train_lengths.append(phrase_len)
                #self.all_captions.append(all_phrase)
                
        # self.train_label = np.load(r"./split_id/{}_label.npy".format(split_dataset))
        self.train_label = np.load(r"./new_data_/{}_label.npy".format(split_dataset))
        self.train_user = np.load(r"./new_data_/{}_user.npy".format(split_dataset))

        if split_dataset == 'train':
            self.label_similarity = 0.7*self.train_label + 0.3*1.0/204

        self.length = len(self.train_captions)
        #if split_dataset == "test":
            #self.length = 500
        print("Data Description : ",len(self.train_captions), len(self.train_label), len(self.train_user))
        
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
            all_caption = np.vstack((self.train_captions[index -1]))
        else:
            all_caption = np.vstack((self.train_captions[index +1]))
        captions = np.vstack((caption, all_caption))
        target = torch.LongTensor(captions)
        label = torch.Tensor(self.train_label[index])
        # length = torch.Tensor(self.train_lengths[index])
        user = torch.Tensor(self.train_user[index])

        # target：2个样本
        # index：当前样本索引
        # label：当前样本标签
        # user：当前用户信息
        return  target, index, label, user

    def __len__(self):
        return self.length-1
    
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

    # 2个JD，标签分布，索引，用户信息
    return targets, labels, ids, users

# definition super parameters
batch_size = 32
num_workers=0
shuffle=True
voc_size=12525
emb_size =512
nhead = 8
nhid = 1024
nlayers = 4
dropout = 0.1
lr=0.0001
weight_decay = 0.1
warmup_steps = 10
t_total = 10
sava_period=2
save_dir = 'logs'

train_dataset = TextDataset("train")
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=collate_fn,drop_last=True)
test_dataset = TextDataset("test")
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collate_fn,drop_last=True)


model = CoKEModel(voc_size, emb_size, nhead, nhid, nlayers, dropout,batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=t_total)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)
model = model.to(device)
# print(model.load_state_dict(torch.load('./logs/ep010.pth', map_location=device),strict=False))


alpha= 0.3
ndcg_mean = 0
ndcg_20 = 0
label_similarity = torch.FloatTensor(train_dataset.label_similarity).to(device)
label_similarity = torch.matmul(label_similarity.T,label_similarity) # (204,204)
print("label_similarity : ", label_similarity.shape)
print("-----------------------------------------------开始训练！--------------------------------------------------")

start_time_total = time.time()
for epoch in range(t_total):
    #print("***"*5,"train","***"*5)
    for i, (train_text, train_label, ids, train_user) in enumerate(train_loader):
        # -----------------------------------------------------------------------------------------
        # train_text:(32,200,50)  32个样本，每个样本（200，50）包含当前的样本和它之后的样本
        # train_label:(32,204)
        # train_user:(32,142)
        # -----------------------------------------------------------------------------------------
        train_text = train_text.to(device)
        train_label = train_label.to(device)
        train_user = train_user.to(device)
        # 对标签分布做了一次线性变换
        train_label = (1-alpha)*train_label + alpha*1.0/204

        # lg_emb:(32,204)，cap_emb:(32,1,204)，loss_sims:数值，match_loss:数值
        lg_emb, cap_emb, loss_sims, match_loss = model(train_text, train_user, label_similarity=label_similarity)

        if len(cap_emb.size()) == 3:
            cap_emb = torch.squeeze(cap_emb, dim=1) # (32,204)

        # 就是最普通的KL散度，前者为真实分布，后者为预测分布
        # loss = kl_t(train_label + 10 ** -6, lg_emb + 10 ** -6)\
        #          +kl_t(train_label + 10 ** -6, cap_emb + 10 ** -6)#+loss_sims

        # -------------------------------------------loss--------------------------------------------
        # 标签分布损失
        loss_label_dis = kl_t(train_label+10**-6, cap_emb+10**-6)
        # 关系一致性损失
        loss_relation_con = loss_sims.item()
        # 用户-JD匹配损失
        user_match_loss = match_loss.item()

        mju = 0.2
        lamda = 1
        loss = (loss_label_dis + mju * loss_relation_con) + lamda * user_match_loss
        # -------------------------------------------------------------------------------------------

        print("打印损失值：")
        print("epoch : ", epoch, "itera : ", i, 'label_dis:%.4f'%float(loss_label_dis), '   relation:%.4f'%float(loss_relation_con), '   match:%.4f'%float(user_match_loss), '   total:%.4f'%float(loss))
        # print("epoch : ", epoch, "itera : ", i, 'label_dis:%.4f'%float(loss_label_dis), '     match:%.4f'%float(user_match_loss), '   total:%.4f'%float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler_warmup.step()  # Update learning rate schedule


        # if epoch < 5: # 如果epoch<18的话，后续代码不会执行
        #     continue
        start_time_2 = time.time()
        if i % 100 == 0 and i != 0:
            print("开始验证：")
            with torch.no_grad():
                chebyshevs, intersections, cosines, clarks, canberras, kls = [], [], [], [], [], []
                cap_embs = np.zeros((992, 204))
                test_labels = np.zeros((992, 204))
                for j, (test_text, test_label, ids, test_user) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        test_text = test_text.cuda()
                        test_label = test_label.cuda()
                        test_user = test_user.cuda()
                    test_label = (1-alpha)*test_label + alpha*1.0/204

                    # global_cls:(32,204)，cap_emb:(32,1,204)，loss_sims:数值，_:数值
                    global_cls, cap_emb, loss_sims, _ = model(test_text, test_user, label_similarity=label_similarity)

                    if len(cap_emb.size()) == 3:
                        cap_emb = cap_emb.mean(dim=1)
                        # cap_emb = cap_emb.detach().cpu().numpy()
                        # np.save('./123.npy',cap_emb)
                        # sys.exit(0)

                    chebyshevs.append(chebyshev_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    intersections.append(intersection_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    cosines.append(cosine_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    kls.append(kl_t(test_label + 10 ** -6, cap_emb + 10 ** -6).item())
                    clarks.append(clark_t(cap_emb, test_label ).item())
                    canberras.append(canberra_t(test_label + 10 ** -6, cap_emb + 10 ** -6 ).item())

                    #记录测试文件
                    cap_emb = cap_emb.detach().cpu().numpy()
                    test_label = test_label.detach().cpu().numpy()
                    cap_embs[ids,:] = cap_emb # 将测试集上的数据填入空的张量中
                    test_labels[ids,:] = test_label # 将测试集上的数据填入空的张量中

                end_time_2 = time.time()
                print("times for train:", end_time_2 - start_time_2)
                result = recall_ndcg(test_labels, cap_embs)
                loss4test1 = (result[5][0]+result[5][1]+result[5][2]+result[5][3]+result[5][4])/5 # 在当前iteration上，测试集上的平均loss
                loss4test2 = result[5][0]
                print("chebyshev : ", np.mean(chebyshevs))
                print("intersection : ", np.mean(intersections))
                print("cosine : ", np.mean(cosines))
                print("kls : ", np.mean(kls))
                print("clarks : ", np.mean(clarks))
                print("canberras : ", np.mean(canberras))
                print("best_epoch : ", epoch)
                print("iteration : ", i)
                print("---------------------------------------------")
                print("recall@20 : ", result[0])
                print("recall@40 : ", result[1])
                print("recall@60 : ", result[2])
                print("recall@80 : ", result[3])
                print("recall@100 : ", result[4])
                print("ndcg : ", result[5])
                print("kl1 : ", kl(test_labels + 10 ** -6, cap_embs + 10 ** -6))
                if loss4test1 > ndcg_mean or loss4test2 > ndcg_20:
                    with open("result_709.txt", 'w') as f:
                        strs = ''
                        strs = "chebyshev : " + str(np.mean(chebyshevs)) + '\n'
                        strs += "intersection : "+ str(np.mean(intersections)) + '\n'
                        strs += "cosine : "+ str(np.mean(cosines)) + '\n'
                        strs += "kls : "+ str(np.mean(kls)) + '\n'
                        strs += "clarks : "+ str(np.mean(clarks)) + '\n'
                        strs += "canberras : "+ str(np.mean(canberras)) + '\n'
                        strs += "best_epoch : "+ str(epoch) + '\n'
                        strs += "iteration : " + str(i) + '\n'
                        strs += "---------------------------------------------" + '\n'
                        strs += "recall@20 : " + str(result[0]) + '\n'
                        strs += "recall@40 : " + str(result[1]) + '\n'
                        strs += "recall@60 : " + str(result[2]) + '\n'
                        strs += "recall@80 : " + str(result[3]) + '\n'
                        strs += "recall@100 : " + str(result[4]) + '\n'
                        strs += "ndcg : " + str(result[5]) + '\n'
                        f.write(strs)
                    # 记录当前测试集上的loss和当前的epoch
                    ndcg_mean = loss4test1
                    ndcg_20 = loss4test2
                    # best_epoch = epoch

                    # 保存测试文件
                    # np.save("main_test_embedding.npy", cap_embs)
                    # np.save("main_test_label.npy", test_labels)

        if (epoch+1)%sava_period==0 or epoch+1==t_total:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d.pth" % (epoch + 1)))


end_time_total = time.time()    
print("train total cost time : ", end_time_total-start_time_total)
