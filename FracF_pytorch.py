# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:22:16 2022

@author: emirhan
"""


'''
Direct computation of fractional fourier transform, implemented on pytorch. Signal to be transformed must be in torch.complex64 data type and must be a column vector.

'''
import torch
import math
import numpy as np

def FracF(vec,a):
    
    N= vec.size(0)
    
    if N % 2 !=0:
        raise Exception("N should be even")
        
    
    
    vec=bizinter(vec)
    z= torch.zeros(N,1)
    vec = torch.cat((z,vec,z),dim=0)
    
    flag=0
    
    if a>0 and a <0.5:
        flag=1
        a=a-1
        
    if a > -0.5 and a<0:
        flag=2
        a=a+1
        
    if a >1.5 and a <2:
        flag=3
        a=a-1
    if a>-2 and a<-1.5:
        flag =4
        a=a+1
        
    res= vec
    
    if flag==1 or flag==3:
        res= corefrmod2(vec,1.0)
    if flag==2 or flag==4:
        res= corefrmod2(vec,-1.0)
   
    if a==0:
        res=vec
    else:
        if a==2 or a==-2:
            res=torch.flipud(vec)
        else:
            res= corefrmod2(res, a)
    
    res=res[N:3*N]
    res=bizdec(res)
    res[0]=2*res[0]
    
    return res
    
    
    
    
    
    
    
def corefrmod2(vec,a):
    
    N= vec.size(0)
  
    deltax= np.sqrt(vec.size(0))
    phi= a*np.pi/2
 
    deltax1= deltax
    alpha=1/np.tan(phi)
    beta= 1/np.sin(phi)
 
    
    x= torch.arange(int(-N/2),int(N/2))/deltax1
    f1=torch.exp(-1j*np.pi*np.tan(phi/2)*x*x)
    f1= torch.unsqueeze(f1,dim=1)
    vec= vec*f1
  
    t= torch.arange(-N+1,N)/deltax1
    hlptc=torch.exp(1j*np.pi*beta*t*t)
    hlptc= torch.unsqueeze(hlptc,dim=1)
    N2= hlptc.shape[0]
   
    N3= int(2**np.ceil(np.log(N2+N-1)/np.log(2)))
    
    z1= torch.zeros(N3-N2,1)
    z2= torch.zeros(N3-N,1)
    hlptcz= torch.cat((hlptc,z1),dim=0)
    fcz=torch.cat((vec,z2),dim=0)
    
    Hcfft= torch.fft.ifft(torch.fft.fft(fcz,dim=0)*torch.fft.fft(hlptcz,dim=0),dim=0) # Convolution with chirp
    Hc=Hcfft[N-1:2*N-1]
    print(Hc)
    #Aphi= np.exp(-1j*(np.pi*np.sign(np.sin(phi)/4-phi/2)))/np.sqrt(np.abs(np.sin(phi)))
    Aphi= np.exp(-1j*(np.pi*np.sign(np.sin(phi))/4-phi/2))/np.sqrt(np.abs(np.sin(phi)))
    print(Aphi)
    xx= torch.arange(int(-N/2),int(N/2))/deltax1
    res= Aphi*f1*Hc/deltax1 # Chirp multiplication
    
    
    return res
    
    


def bizinter(vec):
    
    N= vec.size(0)
    im=0
    
    if torch.sum(torch.abs(vec.imag)) >= 0:
        im=1
        imx=vec.imag
        vec=vec.real
     
    vec2=vec
    z=torch.zeros(N,1)
    vec2= torch.cat((vec2,z),dim=1)
    vec2= torch.reshape(vec2,(2*N,1))
  
    vecf=torch.t(torch.fft.fft(torch.t(vec2)))

    
    patch1= vecf[0:int(N/2)]
    patch2= torch.zeros(N,1).type(torch.complex64)
    patch3= vecf[2*N-int(N/2):2*N]
    
    patch_all= torch.cat((patch1,patch2,patch3),dim=0)
    patch_all = torch.t(patch_all)
    
    xint= 2*torch.real(torch.fft.ifft(patch_all))
 
    print(xint)
    
    if im ==1:
        x2= imx
        z= torch.zeros(N,1)
        x2= torch.cat((x2,z),dim=1)
        x2=torch.reshape(x2,(2*N,1))
        xf=torch.t(torch.fft.fft(torch.t(x2)))
        
        patch1= xf[0:int(N/2)]
        patch2= torch.zeros(N,1).type(torch.complex64)
        patch3= xf[2*N-int(N/2):2*N]
        
        patch_all= torch.cat((patch1,patch2,patch3),dim=0)
        patch_all = torch.t(patch_all)
        
        xmint= 2*torch.real(torch.fft.ifft(patch_all))
  
        xint= torch.complex(xint,xmint)
    
    xint= torch.t(xint)
    
    return xint


def bizdec(x):
    
    L= x.size(0)
    k=torch.arange(0,L-1,2)
    xdec= x[k]
    
    return xdec



############################### EXAMPLE #######################################3
x= torch.unsqueeze(torch.tensor(np.arange(32)),dim=1).type(torch.complex64)
X_fractional= FracF(x,1.7)
#############################################################################33