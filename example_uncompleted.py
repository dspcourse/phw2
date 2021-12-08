#%%
import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl

start = time()
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

if torch.cuda.is_available() == True:
    device = "cuda"
    print("Use GPU!")
else:
    device = "cpu"

norm = True
traindata, trainlabel = np.load("traindata.npy"), np.load("trainlabel.npy")
testdata = np.load("testdata.npy")

if norm:
    print("preprocessing data")
    scalerA = StandardScaler()
    scalerA.fit(traindata[:,0,:])
    tmp = scalerA.transform(traindata[:,0,:])
    traindata[:,0,:] = tmp
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    scalerB = StandardScaler()
    scalerB.fit(traindata[:,1,:])
    tmp = scalerB.transform(traindata[:,1,:])
    traindata[:,1,:] = tmp
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp

# define dataset here
class RawDataset(Dataset):
    def __init__(self, traindata, trainlabel):
        self.traindata = torch.from_numpy(np.array(traindata).astype(np.float32))
        if trainlabel is not None:
            self.trainlabel = torch.from_numpy(np.array(trainlabel).astype(np.int64))
        else:
            self.trainlabel = None
    def __len__(self):
        return len(self.traindata)
    def __getitem__(self,idx):
        sample = self.traindata[idx]
        if self.trainlabel is not None:
            target = self.trainlabel[idx]
            return sample, target
        else:
            return sample

trainset = RawDataset(traindata,trainlabel) 
trainloader = DataLoader(trainset,batch_size=48,shuffle=True)
testset = RawDataset(testdata,None) 
testloader = DataLoader(testset,batch_size=1,shuffle=False)

class MyDSPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # define your arch
        self.encoder = ...
        self.clf = ...
    def forward(self, x):
        # define your forward
        return output
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss

# model
model = MyDSPNet()

# training
trainer = pl.Trainer(gpus=0, num_nodes=1, max_epochs=100)
trainer.fit(model, trainloader,)

model = model.to(device)
with open("result.csv","w") as f:
    f.write("id,category\n")
    for i, x in enumerate(testloader):
        x = x.to(device)
        output = model(x)
        pred = output.argmax(dim=1, keepdim=True)
        f.write("%d,%d\n"%(i,pred.item()))