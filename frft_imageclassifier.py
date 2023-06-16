# -*- coding: utf-8 -*-
"""FrFT_ImageClassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1n-IjP-tgRlJV-G7xI58Njd5KMfUmxOZM
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 07:42:48 2022

@author: emirhan
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 03:41:42 2022

@author: emirhan
"""

'''
2D DFRFT  class implementation for images with trainable parameter a and b. You can change the code according to
application.
'''

import torch
import torch.nn as nn
import numpy as np
import math

import matplotlib.image as img
import matplotlib.pyplot as plt

import torchvision.transforms as T
from PIL import Image

class DFrFT2DM(nn.Module):

    def __init__(self):
         super(DFrFT2DM, self).__init__()

         self.a=  nn.Parameter(torch.rand(1),requires_grad=True)
         self.b =  nn.Parameter(torch.rand(1),requires_grad=True)



    def dfrtmtrx2(self,N,order):
        # Approximation order

        app_ord = 2
        Evec = self._dis_s(N,app_ord)
        Evec=Evec.type(torch.complex64)
        even = 1 - (N%2)

        l = torch.tensor(np.array(list(range(0,N-1)) + [N-1+even]))

        f = torch.diag(torch.exp(-1j*math.pi/2*order*l))


        F= N**(1/2)*torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)/np.sqrt(N)

        return F

    def _dis_s(self, N,app_ord):

        S = self._creates(N,app_ord)

        p = N
        r = math.floor(N/2)
        P = torch.zeros(p,p)

        P[0,0] = 1
        even = 1 - (p%2)

        for i in range(1,r-even+1):
            P[i,i] = 1/(2**(1/2))
            P[i,p-i] = 1/(2**(1/2))

        if even:
            P[r,r] = 1

        for i in range(r+1,p):
            P[i,i] = -1/(2**(1/2))
            P[i,p-i] = 1/(2**(1/2))


        CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)

        C2 = CS[0:math.floor(N/2+1), 0:math.floor(N/2+1)]
        S2 = CS[math.floor(N/2+1):N, math.floor(N/2+1):N]


        ec, vc = torch.linalg.eig(C2)
        ec= ec.type(torch.float32)
        vc= vc.type(torch.float32)

        # idx = np.argsort(ec)
        # ec = ec[idx]
        # vc = vc[:,idx]

        es, vs = torch.linalg.eig(S2)
          # idx = np.argsort(es)
        # es = es[idx]
        # vs = vs[:,idx]
        es= es.type(torch.float32)
        vs= vs.type(torch.float32)

        qvc = torch.vstack((vc, torch.zeros([math.ceil(N/2-1), math.floor(N/2+1)])))
        SC2 = P@qvc # Even Eigenvector of S




        qvs = torch.vstack((torch.zeros([math.floor(N/2+1), math.ceil(N/2-1)]),vs))

        SS2 = P@qvs # Odd Eigenvector of S

        idx = torch.argsort(-ec)

        SC2 = SC2[:,idx]

        idx = torch.argsort(-es)
        SS2 = SS2[:,idx]


        if N%2 == 0:
            S2C2 = torch.zeros([N,N+1])
            SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0],1))])
            S2C2[:,range(0,N+1,2)] = SC2;
            S2C2[:,range(1,N,2)] = SS2


            S2C2= torch.cat((S2C2[:,:N-1],torch.unsqueeze(S2C2[:,-1],1)),1)
            #S2C2 = np.delete(S2C2, (N-1), axis=1)


        else:
            S2C2 = torch.zeros([N,N])
            S2C2[:,range(0,N+1,2)] = SC2;
            S2C2[:,range(1,N,2)] = SS2

        Evec = S2C2


        return Evec

    def _creates(self,N,app_ord):
        # Creates S matrix of approximation order ord
        # When ord=1, elementary S matrix is returned

        app_ord = int(app_ord / 2)

        s = torch.cat((torch.tensor([0, 1]), torch.zeros(N-1-2*app_ord), torch.tensor(np.array([1]))))
        S = self._cconvm(N,s) + torch.diag((torch.fft.fft(s)).real)

        return S

    def _cconvm(self,N,s):
        # Generates circular Convm matrix
        M = torch.zeros((N,N))
        dum = s
        for i in range(N):
            M[:,i] = dum
            dum = torch.roll(dum,1)

        return M

    def forward(self,data4D):


        B= data4D.shape[0]
        CH=data4D.shape[1]
        R= data4D.shape[2]
        C= data4D.shape[3]


        dfrftmatrix1 = self.dfrtmtrx2(R,self.a)
        dfrftmatrix1 = torch.unsqueeze(torch.unsqueeze(dfrftmatrix1,0),0)


        dfrftmatrix2 = self.dfrtmtrx2(C,self.b)
        dfrftmatrix2 = torch.unsqueeze(torch.unsqueeze(dfrftmatrix2,0),0)

        C_transform=dfrftmatrix1@data4D.type(torch.complex64)
        R_transform= dfrftmatrix2@torch.transpose(C_transform,2,3)
        out= torch.transpose(R_transform,2,3)
        return out

import torch
import torchvision
import torchvision.transforms as transforms

batch_size=100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.frac= DFrFT2DM()

    def forward(self, x):

        x = torch.real(self.frac(self.pool(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
pytorch_total_params

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

a1=[]
a2=[]
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("**********************Parameter A1***************************",net.frac.a)
        #print("**********************Parameter B1***************************",net.frac.b)
        a1.append(net.frac.a)
        a2.append(net.frac.b)
        # print statistics
        running_loss += loss.item()
        print(f'[{epoch+1},{(i+1)*100}]')

print('Finished Training')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')