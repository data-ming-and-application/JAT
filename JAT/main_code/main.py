import numpy as np
import torch
import torch.utils.data as Data
import torch.nn as nn1
from JAT import CoKEModel
from metrics import *
import os
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
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

        with open(r"../../data/DataSets_jd/split_phrase/{}_lengths.txt".format(split_dataset),'r') as f:
            for line in f:
                record = line.strip().split('\t')
                #定义每条句子包含100条短语，每个短语长度100
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
                        phrase[rows][columns] = int(record_split[columns])
                        #all_phrase[amount] = int(record_split[columns])
                        #amount += 1
                self.train_captions.append(phrase)
                self.train_lengths.append(phrase_len)
                #self.all_captions.append(all_phrase)
                
        self.train_label = np.load(r"../../data/DataSets_jd/split_id/{}_label.npy".format(split_dataset))
        
        if split_dataset == 'train':
            self.label_similarity = 0.7*self.train_label + 0.3*1.0/200
        
        self.length = len(self.train_captions)
        #if split_dataset == "test":
            #self.length = 500
        print("Data Description : ",len(self.train_captions), len(self.train_label))
        
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
            all_caption = self.train_captions[index -1]
        else:
            all_caption = self.train_captions[index +1]
        captions = np.vstack((caption, all_caption))
        target = torch.LongTensor(captions)
        label = torch.Tensor(self.train_label[index])
        length = torch.Tensor(self.train_lengths[index])

        return  target, index, label

    def __len__(self):
        return self.length-1
    
def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, ids, labels = zip(*data)
    labels = torch.stack(labels, 0)
    targets = torch.stack(captions, 0)

    '''# Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), 300).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]'''
       
    return targets, labels, ids

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
t_total = 15

train_dataset = TextDataset("train")
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True,num_workers=num_workers,collate_fn=collate_fn,drop_last=True)
test_dataset = TextDataset("test")
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=collate_fn,drop_last=True)


model = CoKEModel(voc_size, emb_size, nhead, nhid, nlayers, dropout,batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=t_total)
loss_cls = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)
model = model.to(device)
    
alpha= 0.3
max_kls = 1000
best_epoch = 0
label_similarity = torch.FloatTensor(train_dataset.label_similarity).to(device)
label_similarity = torch.matmul(label_similarity.T,label_similarity)
print("label_similarity : ", label_similarity.shape)

start_time_total = time.time()
for epoch in range(t_total):
    #print("***"*5,"train","***"*5)
    for i, (train_text,train_label,ids) in enumerate(train_loader):
        #print("ids : ", ids)
        start_time = time.time()
        train_text = train_text.to(device)
        train_label = train_label.to(device)
        train_label = (1-alpha)*train_label + alpha*1.0/200
        
        lg_emb, cap_emb, loss_sims = model(train_text, epoch=epoch, label_similarity=label_similarity) 
        lg_emb = lg_emb.squeeze()
        if len(cap_emb.size()) == 3:
            cap_emb = cap_emb.mean(dim=1)
        if len(lg_emb.size()) == 3:
            lg_emb = lg_emb.mean(dim=1)
        
        loss = kl_t(train_label + 10 ** -6, lg_emb + 10 ** -6)\
                 +kl_t(train_label + 10 ** -6, cap_emb + 10 ** -6)#+loss_sims
        print(kl_t(train_label + 10 ** -6, lg_emb + 10 ** -6)\
              ,kl_t(train_label + 10 ** -6, cap_emb + 10 ** -6), loss_sims)
        end_time = time.time()
        print("epoch : ", epoch, "itera : ", i, 'lr : ', scheduler_warmup.get_lr()[0], " loss : ", loss.item())
        print("train cost time : ", end_time-start_time)
        
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler_warmup.step()  # Update learning rate schedule
        
        start_time = time.time()
        if epoch < 18:
            continue
        if epoch % 1 == 0 or i % 20 == 0:
            with torch.no_grad():
                chebyshevs, intersections, cosines, clarks, canberras, kls = [], [], [], [], [], []
                cap_embs = np.zeros((len(test_loader.dataset),200))
                test_labels = np.zeros((len(test_loader.dataset),200))
                for j, (test_text,test_label,ids) in enumerate(test_loader): 
                    if torch.cuda.is_available():
                        test_text = test_text.cuda()
                        test_label = test_label.cuda()
                    test_label = (1-alpha)*test_label + alpha*1.0/200
                    global_cls, cap_emb, loss_sims = model(test_text,epoch=epoch, label_similarity=label_similarity)
                    global_cls = global_cls.squeeze()
                    if len(cap_emb.size()) == 3:
                        cap_emb = cap_emb.mean(dim=1)
                    
                    chebyshevs.append(chebyshev_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    intersections.append(intersection_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    cosines.append(cosine_t(cap_emb + 10 ** -6, test_label + 10 ** -6).item())
                    kls.append(kl_t(test_label + 10 ** -6, cap_emb + 10 ** -6).item())
                    clarks.append(clark_t(cap_emb, test_label ).item())
                    canberras.append(canberra_t(test_label + 10 ** -6, cap_emb + 10 ** -6 ).item())
                    
                    #记录测试文件
                    cap_emb = cap_emb.detach().cpu().numpy()
                    test_label = test_label.detach().cpu().numpy()
                    cap_embs[ids,:] = cap_emb
                    test_labels[ids,:] = test_label
                
                if np.mean(kls) < max_kls:
                    with open("result.txt", 'w') as f:
                        strs = ''
                        strs = "chebyshev : " + str(np.mean(chebyshevs)) + '\n'
                        strs += "intersection : "+ str(np.mean(intersections)) + '\n'
                        strs += "cosine : "+ str(np.mean(cosines)) + '\n'
                        strs += "kls : "+ str(np.mean(kls)) + '\n'
                        strs += "clarks : "+ str(np.mean(clarks)) + '\n'
                        strs += "canberras : "+ str(np.mean(canberras)) + '\n'
                        strs += "best_epoch : "+ str(best_epoch) + '\n'
                        f.write(strs)
                    print("chebyshev : ",np.mean(chebyshevs))
                    print("intersection : ",np.mean(intersections))
                    print("cosine : ",np.mean(cosines))
                    print("kls : ",np.mean(kls))
                    print("clarks : ",np.mean(clarks))
                    print("canberras : ",np.mean(canberras))
                    print("best_epoch : ",best_epoch)
                    max_kls = np.mean(kls)
                    best_epoch = epoch
                    
                    # 保存测试文件
                    #np.save("main_test_embedding_u_0.4.npy",cap_embs)
                    #np.save("main_test_label_u_0.4.npy",test_labels)
                    print("kl : ", kl(test_labels + 10 ** -6, cap_embs + 10 ** -6))
        end_time = time.time()
        print("test cost time : ", end_time-start_time)
        
end_time_total = time.time()    
print("train total cost time : ", end_time_total-start_time_total)
