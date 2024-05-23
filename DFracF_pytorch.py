import numpy as np
from scipy import linalg
import math
import torch


'''
This file creates discrete fractional Fourier transform matrix( (NxN) for a signal of length N.
It takes three parametere. N is the sequence length, a is the fraction order.

The vector to be multiplied with dFrFT matrix must be a column vector (Nx1). Also, its
data dtype should be converted to "tensor.complex64". A solution to convert it: 

X.type(tensor.complex64) where is X is the vector we use.

After you obtain the fractional Fourier of a signal with fraction ordeer and would like to
invert it to its original value, the procedure to follow is simple. Just take a glance to example below.
                                                               
In ivnersion, we divide the result with length of the original signal. Instead of dividing forward and backward
conversions of square root of length of signal, we divided end product with length of signal. 

It seems that there is a small difference such as 0,000005 between orijinal and inverted values. This is not considerable.

'''
def dfrtmtrx2(N, a):
    # Approximation order
    app_ord = 2
    Evec = _dis_s(N,app_ord)
    Evec=Evec.type(torch.complex64)
    even = 1 - (N%2)
    
    l = torch.tensor(np.array(list(range(0,N-1)) + [N-1+even]))
    
    f = torch.diag(torch.exp(-1j*math.pi/2*a*l))
    
    F= N**(1/2)*torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)/np.sqrt(N)
    
    return F

def _dis_s(N,app_ord):
    
    S = _creates(N,app_ord)
    
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
    
def _creates(N,app_ord):
    # Creates S matrix of approximation order ord
    # When ord=1, elementary S matrix is returned
    
    app_ord = int(app_ord / 2) 
    
    s = torch.cat((torch.tensor([0, 1]), torch.zeros(N-1-2*app_ord), torch.tensor(np.array([1]))))
    S = _cconvm(N,s) + torch.diag((torch.fft.fft(s)).real)
    
    return S
    
def _cconvm(N,s):
    # Generates circular Convm matrix
    M = torch.zeros((N,N))
    dum = s
    for i in range(N):
        M[:,i] = dum
        dum = torch.roll(dum,1)
        
    return M
    


############################## EXAMPLE ##################################


vec= torch.arange(8).type(torch.complex64)
N=vec.shape[0]

a=1.1

dFrFT= dfrtmtrx2(N,a)
print("Discrete Fractional Fourier matrix is created. Matrix dimension is {}x{} and fraction order is {}".format(N,N,a) )
dFrFT_inv = dfrtmtrx2(N,-a)
print("Discrete Fractional Fourier matrix is created. Matrix dimension is {}x{} and fraction order is -{}".format(N,N,a) )

y= dFrFT@vec
print("*****************************************")
print("DFrFT is computed:",y)
print("*****************************************")
z= dFrFT_inv@y
print("DFrFT is inverted: Original signal is:",z)

