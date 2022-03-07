import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import jsonlines
import semanticscholar as sch
import os
import re
import torch
import math
import random
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from IPython.display import FileLink,FileLinks



# scicite data loading............
train_main = 'scicite/train.jsonl'
test_main = 'scicite/test.jsonl'
val_main = 'scicite/dev.jsonl'
sec_sc = 'scicite/scaffolds/sections-scaffold-train.jsonl'
cit_sc = 'scicite/scaffolds/cite-worthiness-scaffold-train.jsonl'

## Section scaffold data
p=[]
with jsonlines.open(sec_sc) as f:
    for line in f.iter():
        p.append(line)
data = pd.DataFrame(p)

## Citation Worthiness scaffold data
q=[]
with jsonlines.open(cit_sc) as f:
    for line in f.iter():
        q.append(line)
datac = pd.DataFrame(q)

## SciCite Main task data
tm=[]
with jsonlines.open(train_main) as f:
    for line in f.iter():
        tm.append(line)
data_main = pd.DataFrame(tm)




def remove_idx(l,max_len): # max_len = len of tokens criteria above which to remove
    lent = []
    idx_remove = []
    for i in l:
        lent.append(len(tokenizer.encode(i, padding=True, return_tensors="pt")[0]))
    for i in range(len(lent)):
        if(lent[i]>max_len):
            idx_remove.append(i)
    return idx_remove
    
    
def class_div(X,Y,instances,num_class):
    inst_per_class = instances//num_class
    rem = instances%num_class
    x_out=[]
    y_out=[]
    for i in range(num_class):
        xcl = [x for x,y in zip(X,Y) if y==i]
        ycl = [y for y in Y if y==i]
        print('xcl length : ',len(xcl))
        if(rem):
            x_out += xcl[:inst_per_class+1]
            y_out += ycl[:inst_per_class+1]
            rem -= 1
            print('x_out length : ',len(x_out))
        else:
            x_out += xcl[:inst_per_class]
            y_out += ycl[:inst_per_class]
            print('x_out length : ',len(x_out))
    return (x_out,y_out)
def fun(i):
    pat = ' '.join(re.split(r'\[[^\[\]]*\]' ,i))
    pat = ' '.join(re.split(r'\([^\[\]\(\)]*\)' ,pat))
    return pat
    
    
    
citm = data_main['string'].values.tolist()
citm = [fun(i) for i in citm]
data_main['processed_text'] = citm

yc = data_main['label'].values.tolist()
label={'background':0,'method':1,'result':2}
y = list(map(lambda t : label[t],yc))


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)

citm_idx_remove_300 = remove_idx(citm,300)
for i in sorted(citm_idx_remove_300, reverse = True):  
    del citm[i]
    del y[i]
print(f'removed {len(citm_idx_remove_300)} instances')


x_cit = datac['cleaned_cite_text'].values.tolist()
x_cit = [fun(i) for i in x_cit]
datac['processed_text'] = x_cit

ycit = datac['is_citation'].values.tolist()
ycit_ie = list(map(lambda x : int(x),ycit))




xcit_idx_remove_300 = remove_idx(x_cit,300)
for i in sorted(xcit_idx_remove_300, reverse = True):  
    del x_cit[i]
    del ycit_ie[i]
print(f'removed {len(xcit_idx_remove_300)} instances')


x_sec = data['text'].values.tolist()
x_sec = [fun(i) for i in x_sec]
data['processed_text'] = x_sec

y_sec = data['section_name'].values.tolist()
labels={'introduction':0,'related work':1,'method':2,'experiments':3,'conclusion':4}
ysec_ie = list(map(lambda t : labels[t],y_sec))


xsec_idx_remove_300 = remove_idx(x_sec,300)
for i in sorted(xsec_idx_remove_300, reverse = True):  
    del x_sec[i]
    del ysec_ie[i]
print(f'removed {len(xsec_idx_remove_300)} instances')



#for v-1 model = method - 1 = without balancing the data
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)

x = list(map(lambda t : tokenizer.encode(t,padding='max_length',max_length=300),citm))

y = torch.tensor(y[:-2])
x = torch.tensor(x[:-2])
print('x.shape : ',x.shape)
print('y.shape : ',y.shape)

xcit = list(map(lambda t : tokenizer.encode(t,padding='max_length',max_length=300),x_cit))

xsec = list(map(lambda t : tokenizer.encode(t,padding='max_length',max_length=300),x_sec))

sec = [(x,y) for x,y in zip(xsec,ysec_ie)]
sec = random.sample(sec, 8232)

cit = [(x,y) for x,y in zip(xcit,ycit_ie)]
cit = random.sample(cit, 8232)

xcit = torch.tensor([t[0] for t in cit])
ycit = torch.tensor([t[1] for t in cit])
xsec = torch.tensor([t[0] for t in sec])
ysec = torch.tensor([t[1] for t in sec])
print(xcit.shape)
print(xsec.shape)


# validation data
vm=[]
with jsonlines.open(val_main) as f:
    for line in f.iter():
        vm.append(line)
pdvm = pd.DataFrame(vm)

citvm=pdvm['string'].values.tolist()
citvm = [fun(i) for i in citvm]
pdvm['processed_text'] = citvm
yc = pdvm['label'].values.tolist()


label={'background':0,'method':1,'result':2}
vy = list(map(lambda t : label[t],yc))
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)
vx = list(map(lambda t : tokenizer.encode(t,padding='max_length',max_length=300),citvm))
vy = torch.tensor(vy)
vx = torch.tensor(vx)


# test data
testm=[]
with jsonlines.open(test_main) as f:
    for line in f.iter():
        testm.append(line)
pdtm = pd.DataFrame(testm)


cittem=pdtm['string'].values.tolist()
cittem = [fun(i) for i in cittem]
pdtm['processed_text'] = cittem
cittem = cittem[:-1]
yc = pdtm[:-1]['label'].values.tolist()
label={'background':0,'method':1,'result':2}
ty = list(map(lambda t : label[t],yc))
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)
tx = list(map(lambda t : tokenizer.encode(t,padding='max_length',max_length=300),cittem))
ty = torch.tensor(ty)
tx = torch.tensor(tx)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        self.a = 0
        self.th = 0
        self.eij = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        # print('weight : ',self.weight)
        # print('weight shape : ',weight.shape)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        # print('bias shape : ',self.b.shape)
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim
        # print('x shape : ',x.shape)
        # tem1 = x.contiguous().view(-1, feature_dim)
        # print(tem1.shape)
        # tem2 = torch.mm(tem1, self.weight)
        # print(tem2.shape)
        # print(step_dim)
        self.eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
#         print('eij shape : ',self.eij.shape)
        
        if self.bias:
            self.eij = self.eij + self.b
            
        self.th = torch.tanh(self.eij)
#         print('tanh out shape : ',self.th.shape)
        a = torch.exp(self.th)
#         print('a shape : ',a.shape)
        
        if mask is not None:
            a = a * mask

        self.a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
#         print('a divided by sum shape : ',self.a.shape)

        weighted_input = x * torch.unsqueeze(self.a, -1)
#         print('weighted input : ',weighted_input.shape)
        return torch.sum(weighted_input, 1)
        


#feed forward of main task
class feedforward1(nn.Module):
    def __init__(self,data):
        super().__init__()
        
        n = 3 if data=='sci' else 6
        drop = 0.2 if data == 'sci' else 0.4
            
        self.drop = nn.Dropout(p=drop)
        self.lin = nn.Linear(100,20)
        self.relu = torch.nn.ReLU()
        self.out = nn.Linear(20,n)
    def forward(self,x):
        x = self.drop(x)
        lin_out = self.lin(x)
        x = self.relu(lin_out)
        x = self.out(x)
        return (x,lin_out) 
        
        
        
#feed forward of section scaffold
class feedforward2(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.4)
        self.lin = nn.Linear(100,20)
        self.relu = torch.nn.ReLU()
        self.out = nn.Linear(20,5)
    def forward(self,x):
        x = self.drop(x)
        
        lin_out = self.lin(x)
        x = self.relu(lin_out)
        x = self.out(x)
        return (x,lin_out) 
        
        
        
#feed forward of citation worthiness scaffold
class feedforward3(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.4)
        self.lin = nn.Linear(100,20)
        self.relu = torch.nn.ReLU()
        self.out = nn.Linear(20,2)
    def forward(self,x):
        x = self.drop(x)
        lin_out = self.lin(x)
        x = self.relu(lin_out)
        x = self.out(x)
        return (x,lin_out)
        
        


#feed forward of citance + cited title scaffold
class feedforward4(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.4)
        self.lin = nn.Linear(100,20)
        self.relu = torch.nn.ReLU()
        self.out = nn.Linear(20,6)
    def forward(self,x):
        x = self.drop(x)
        lin_out = self.lin(x)
        x = self.relu(lin_out)
        x = self.out(x)
        return (x,lin_out) 
        
        
        

class modelv3(nn.Module):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.BertModel = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
        self.lstm = nn.LSTM(768,50,num_layers=1,bidirectional=True,batch_first=True)
        self.att = attention(100,300)   ## for attention defination
        self.main_sci = feedforward1(data='sci')
        self.main_pk = feedforward1(data='pk')
        self.sec = feedforward2()
        self.cit = feedforward3()
        self.cited = feedforward4()
    def forward(self,x,n,data=None):
        xbert=self.BertModel(x)
        lstm,_=self.lstm(xbert[0])
        print(lstm.shape)
        at = self.att(lstm).view(-1,100)
        if(n==1 and data=='sci'):
            return self.main_sci(at)[0]
        
        elif(n==1 and data=='pk'):
            return self.main_pk(at)[0]
        
        elif(n==2):
            return self.sec(at)[0]
        
        elif(n==3):
            return self.cit(at)[0]
        elif(n==4):
            return self.cited(at)[0]
        else:
            # predicting == training done!!
            if(data=='sci'):
                z,last = self.main_sci(at)
                _,last_sec = self.sec(at)
                _,last_cit = self.cit(at)
                z = F.softmax(z,dim=1)
                z = torch.argmax(z,dim=1) 
                return (z,last,at,last_cit,last_sec,lstm)
            else:
                z,last = self.main_pk(at)
                _,last_sec = self.sec(at)
                _,last_cit = self.cit(at)
                _,last_cited = self.cited(at)
                z = F.softmax(z,dim=1)
                z = torch.argmax(z,dim=1)
                return (z,last,at,last_cit,last_sec,last_cited,lstm)
            
            
            
            
batchSize = 12
num_of_feedforwards = 3
#model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
main_tr = torch.utils.data.TensorDataset(x, y)
main_val = torch.utils.data.TensorDataset(vx, vy)
main_test = torch.utils.data.TensorDataset(tx, ty)
sec_tr = torch.utils.data.TensorDataset(xsec,ysec)
cit_tr = torch.utils.data.TensorDataset(xcit,ycit)
#tit_tr = torch.utils.data.TensorDataset(x_tit,y_tit)

 
train_sampler = torch.utils.data.RandomSampler(main_tr)
train_data = torch.utils.data.DataLoader(main_tr, sampler=train_sampler, batch_size=batchSize//num_of_feedforwards)

train_sampler = torch.utils.data.RandomSampler(sec_tr)
sec_data = torch.utils.data.DataLoader(sec_tr, sampler=train_sampler, batch_size=batchSize//num_of_feedforwards)

train_sampler = torch.utils.data.RandomSampler(cit_tr)
cit_data = torch.utils.data.DataLoader(cit_tr, sampler=train_sampler, batch_size=batchSize//num_of_feedforwards)

val_sampler = torch.utils.data.RandomSampler(main_val)
val_data = torch.utils.data.DataLoader(main_val, sampler=val_sampler, batch_size=batchSize//num_of_feedforwards)

test_sampler = torch.utils.data.RandomSampler(main_test)
test_data = torch.utils.data.DataLoader(main_test, sampler=test_sampler, batch_size=batchSize//num_of_feedforwards)




## Training models
mod=modelv3(batchSize//num_of_feedforwards)
mod.to(device)

for name,param in mod.named_parameters():
    if(name.split('.')[0] == 'main_pk' or name.split('.')[0] == 'cited'):
        param.requires_grad = False

lambd1 = 0.05   #lambd1 for influence of section scaffold
lambd2 = 0.1    #lambd2 for influence of citation worthiness scaffold
#lambd3 = 0.1    #lambd3 for influence of cited paper title scaffold

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(mod.parameters(), lr = 0.01)

loss_list = []
f1_list = []
best_val_f1 = 0
val_f1 = 0

for epoch in range(1):
    running_loss = 0
    ypr=[]
    ytr=[]
    y_eval=[]
    ypr_eval=[]
    
# training--------------------
    mod = mod.train()
#     print(mod.att.att_weights.requires_grad)
#     print(mod.att.out.requires_grad)
    for i,data in enumerate(zip(train_data,cit_data,sec_data)):
        m = data[0]
        c = data[1]
        s = data[2]
        
        in_main,tar_main = m[0].to(device),m[1].to(device)
        in_cit,tar_cit = c[0].to(device),c[1].to(device)
        in_sec,tar_sec = s[0].to(device),s[1].to(device)
        
        optimizer.zero_grad()
        
        main = mod(in_main,1,'sci')
        sec = mod(in_sec,2)
        cit = mod(in_cit,3)
        
        loss_main = loss(main,tar_main)
        loss_cit = loss(cit,tar_cit)
        loss_sec = loss(sec,tar_sec)

        overall_loss = (loss_main + lambd1*loss_sec + lambd2*loss_cit)/num_of_feedforwards  # becoz initially the summation is avg loss per mini batch(8) but we need avg loss per mini batch(24)
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 5)
#         plot_grad_flow(mod.named_parameters())
        optimizer.step()
        
        running_loss += overall_loss.item()
        if(i%100 == 99):
            loss_list.append({'epoch':epoch+1,'batch':i+1,'loss':round(running_loss / 100,3)})
            print('[epoch : %d,batch : %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0
   
# validation ------------------------
    mod = mod.eval()
    with torch.no_grad():
        
        # calculating f1_score for train data
        for d in train_data:
            x = d[0].to(device)
            y = d[1].to(device)
            for yt in y.cpu():
                ytr.append(yt)
            y_pred = mod(x,0,'sci')[0].cpu()
            for yt in y_pred:
                ypr.append(yt)
                
        f1 = f1_score(ytr,ypr,average='macro')
        
        # calculating f1_score for validation data
        for d in val_data:
            xv = d[0].to(device)
            yv = d[1].to(device)
            for yt in yv.cpu():
                y_eval.append(yt)
            y_pred = mod(xv,0,'sci')[0].cpu()
            for yt in y_pred:
                ypr_eval.append(yt)
                
        val_f1 = f1_score(y_eval,ypr_eval,average='macro')
        
        f1_list.append({'epoch':epoch+1,'train_f1_score':f1,'val_f1_score':val_f1})
        
    print('*'*40)
    print('train confusion matrix : ')
    print(confusion_matrix(ytr, ypr))
    print('*'*40)
    print('val confusion matrix : ')
    print(confusion_matrix(y_eval, ypr_eval))
    print('*'*40)
    print('[epoch : %d] train_f1_macro: %.3f, val_f1_macro: %.3f' %(epoch+1, f1, val_f1))
    print('*'*40)
    if val_f1 > best_val_f1:
        torch.save(mod, f'./cohan_v3_newatt_dropout0.4_ep.pt')
    
#     if((epoch+1)%2==0):
#         torch.save(mod, f'./cohan_modelv3_ep{epoch+1}.pt')

print('Finished Training!!') 

from IPython.display import FileLink, FileLinks
FileLinks('./') #lists all downloadable files on server


## test on SciCite val data
y_val=[]
ypr_val=[]
with torch.no_grad():
    mod.eval()
    for v in val_data:
        xtv = v[0].to(device)
        ytv = v[1].to(device)
        for yv in ytv.cpu():
            y_val.append(yv)
        y_predv = mod(xtv,0,'sci')[0].cpu()
        for yv in y_predv:
            ypr_val.append(yv)
                
val_f1 = f1_score(y_val,ypr_val,average='macro')
print('val_f1_score : ',val_f1)


## test on SciCite test data
y_test=[]
ypr_test=[]
with torch.no_grad():
    mod.eval()
    for d in test_data:
        xte = d[0].to(device)
        yte = d[1].to(device)
        for yt in yte.cpu():
            y_test.append(yt)
        y_pred = mod(xte,0,'sci')[0].cpu()
        for yt in y_pred:
            ypr_test.append(yt)
                
test_f1 = f1_score(y_test,ypr_test,average='macro')
print('test_f1_score : ',test_f1)



                
pdtr_new = pd.read_csv('./iiptrain.csv')
pdte_new = pd.read_csv('./iiptest.csv')



# code for combining citance and cited title
total = int(pdtr_new.shape[0] * 0.8)
prtr_new = list(map(lambda t : t[:t.find('#AUTHOR_TAG')]+t[t.find('#AUTHOR_TAG')+11:],pdtr_new['citation_context']))
prtr_new = [fun(i) for i in prtr_new]
pdtr_new['processed_text'] = prtr_new
yc = pdtr_new['citation_class_label'].values.tolist()
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)
xe = list(map(lambda t : tokenizer.encode(t),prtr_new[:total]))
x_citede = list(map(lambda t : tokenizer.encode(t[0],padding='max_length',pad_to_max_length=True,max_length=301-len(t[1]))[1:],zip(pdtr_new['cited_title'][:total],xe)))
x1 = [x+y for x,y in zip(xe,x_citede)]
ty = torch.tensor(yc)
x1 = torch.tensor(x1)
print(x1.shape)




tx = list(map(lambda t : tokenizer.encode(t,padding='max_length',pad_to_max_length=True,max_length=300),prtr_new))
tx = torch.tensor(tx)
total = int(tx.shape[0] * 0.8)
x = tx[:total]
y = ty[:total]
vx = tx[total:]
vy = ty[total:]
print(x.shape)
print(y.shape)
print(vx.shape)
print(vy.shape)

sampling_strategy = {0:1447,1:600,2:600,3:600,4:600,5:600}  
# use this if u want to undersample using RandomUnderSampler after oversampling by SMOTE
tr_over = SMOTE(sampling_strategy=sampling_strategy)
tr_under = RandomUnderSampler(sampling_strategy={0:600,1:600,2:600,3:600,4:600,5:600})
x_smote,y_smote = tr_over.fit_resample(x,y)
x_smote,y_smote = tr_under.fit_resample(x_smote,y_smote)

x1_smote,y_smote = tr_over.fit_resample(x1,y)
x1_smote,y_smote = tr_under.fit_resample(x1_smote,y_smote)
# print(Counter(y_smote))

sampling_strategy = {0:201,1:100,2:100,3:100,4:100,5:100} # use this if u want to undersample as well after oversampling by SMOTE is done
# while using above dict for sampling strategy of SMOTE, put k_neighbors = 4
val_over = SMOTE(sampling_strategy=sampling_strategy,k_neighbors = 3)
val_under = RandomUnderSampler(sampling_strategy={0:100,1:100,2:100,3:100,4:100,5:100})
# print(vx, vy)
vx_smote,vy_smote = val_over.fit_resample(vx,vy)
vx_smote,vy_smote = val_under.fit_resample(vx_smote,vy_smote)
print(Counter(vy_smote))

x_smote = torch.tensor(x_smote)
y_smote = torch.tensor(y_smote)
vx_smote = torch.tensor(vx_smote)
vy_smote = torch.tensor(vy_smote)
x1_smote = torch.tensor(x1_smote)



batchSize = 8
num_of_feedforwards = 2  # 2 for finetuning on 3C and 3 for training on SciCite.

main_tr = torch.utils.data.TensorDataset(x_smote, y_smote)
main_val = torch.utils.data.TensorDataset(vx_smote, vy_smote)
tit_tr = torch.utils.data.TensorDataset(x1_smote,y_smote)
 
train_sampler = torch.utils.data.RandomSampler(main_tr)
train_data = torch.utils.data.DataLoader(main_tr, sampler=train_sampler, batch_size=batchSize//num_of_feedforwards)

val_sampler = torch.utils.data.RandomSampler(main_val)
val_data = torch.utils.data.DataLoader(main_val, batch_size=batchSize//num_of_feedforwards)

train_sampler = torch.utils.data.RandomSampler(tit_tr)
tit_data = torch.utils.data.DataLoader(tit_tr, sampler=train_sampler, batch_size=batchSize//num_of_feedforwards)



#training cell..................
mod = torch.load('./cohan_v3_newatt_dropout0.4_ep.pt')
# mod=modelv3(batchSize//num_of_feedforwards)
mod.to(device)

for name,param in mod.named_parameters():
    if(name.split('.')[0] == 'main_sci'):
        param.requires_grad = False
    if(name.split('.')[0] == 'main_pk' or name.split('.')[0] == 'cited'):
        param.requires_grad = True

lambd1 = 0.05   #lambd1 for influence of section scaffold
lambd2 = 0.1    #lambd2 for influence of citation worthiness scaffold
lambd3 = 0.1    #lambd3 for influence of cited paper title scaffold

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(mod.parameters(), lr = 0.01)

loss_list = []
f1_list = []
best_val_f1 = 0
val_f1 = 0

for epoch in range(1):
    running_loss = 0
    ypr=[]
    ytr=[]
    y_eval=[]
    ypr_eval=[]
    
# training--------------------
    mod.train()
    
    for i,data in enumerate(zip(train_data,tit_data)):
        m = data[0]
        t = data[1]
        
        in_main,tar_main = m[0].to(device),m[1].to(device)
        in_tit,tar_tit = t[0].to(device),t[1].to(device)
        
        optimizer.zero_grad()
        
        main = mod(in_main,1,'pk')
        tit = mod(in_tit,4)
    
        loss_main = loss(main,tar_main)
        loss_tit = loss(tit,tar_tit)

        overall_loss = (loss_main + lambd3*loss_tit)/num_of_feedforwards  # becoz initially the summation is avg loss per mini batch(8) but we need avg loss per mini batch(24)
        overall_loss.backward()
        torch.nn.utils.clip_grad_norm_(mod.parameters(), 5)
        optimizer.step()
        
        running_loss += overall_loss.item()
        if(i%100 == 99):
            loss_list.append({'epoch':epoch+1,'batch':i+1,'loss':round(running_loss / 100,3)})
            print('[epoch : %d,batch : %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0
   
# validation ------------------------
    with torch.no_grad():
        mod.eval()
        
        # calculating f1_score for train data
        for d in train_data:
            x = d[0].to(device)
            y = d[1].to(device)
            for yt in y.cpu():
                ytr.append(yt)
            y_pred = mod(x,0,'pk')[0].cpu()
            for yt in y_pred:
                ypr.append(yt)
                
        f1 = f1_score(ytr,ypr,average='macro')
        
        # calculating f1_score for validation data
        for d in val_data:
            xv = d[0].to(device)
            yv = d[1].to(device)
            for yt in yv.cpu():
                y_eval.append(yt)
            y_pred = mod(xv,0,'pk')[0].cpu()
            for yt in y_pred:
                ypr_eval.append(yt)
                
        val_f1 = f1_score(y_eval,ypr_eval,average='macro')
        
        f1_list.append({'epoch':epoch+1,'train_f1_score':f1,'val_f1_score':val_f1})
        
    print('*'*40)
    print('train confusion matrix : ')
    print(confusion_matrix(ytr, ypr))
    print('*'*40)
    print('val confusion matrix : ')
    print(confusion_matrix(y_eval, ypr_eval))
    print('*'*40)
    print('[epoch : %d] train_f1_macro: %.3f, val_f1_macro: %.3f' %(epoch+1, f1, val_f1))
    print('*'*40)
    # if((epoch+1)%2==0):
    if val_f1 > best_val_f1: 
        torch.save(mod, f'./cohan_dropout_0.4_3c_smote_ep.pt')

print('Finished Training!!')  
print(ty.shape)


mod = torch.load('./cohan_dropout_0.4_3c_smote_ep.pt')
mod.to(device)

prte = list(map(lambda t : t[:t.find('#AUTHOR_TAG')]+t[t.find('#AUTHOR_TAG')+11:],pdte_new['citation_context']))
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased",do_lower_case=True)
tx = list(map(lambda t : tokenizer.encode(t,padding='max_length',pad_to_max_length=True,max_length=300),prte))
tx = torch.tensor(tx)

ind = list(pdte_new['unique_id'])
act_label = pdte_new['citation_class_label'].tolist()
pred = []
pre_label = []
with torch.no_grad():
    mod.eval()
    for i in range(0,len(tx),4):
        l=[]
        x = tx[i:i+4]
        idx = ind[i:i+4]
        x = x.to(device)
        y_pr = mod(x,0,'pk')[0].cpu()
        for j in range(len(x)):
            l = [idx[j],y_pr[j].item()]
            pred.append(l)
            pre_label.append(y_pr[j].item())
# print(pre_label)
# print(act_label)
output_file_name = 'log.txt'
file = open(output_file_name,'w+')
f1 = f1_score(act_label, pre_label, average='macro')
file.write(f"Macro F1: {f1}\n")
print(f1)
file.close()
        
# df = pd.DataFrame(pred, columns = ['unique_id', 'citation_class_label']) 
# df.set_index('unique_id', inplace = True)

# print(df)
# df.to_csv('./submission.csv') 





