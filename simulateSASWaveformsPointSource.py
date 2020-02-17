import torch
import numpy as np
import ProjData
from timeDelay import *
from utils import *
import time

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def simulateSASWaveformsPointSource(RP, ps):
    shape = ps.shape
    numScat = list(shape)[0]
    RP.projDataArray = []
    psShape = ps.shape
    dim = list(shape)[1]
    count = 0
    #h = ps.register_hook(lambda x: print(x))
    #RP.hooks.append(h)
    for i in range(0, numScat):
        for j in range(0, RP.numProj):
            pData = ProjData.ProjData(projPos=RP.projectors[j, :], Fs=RP.Fs, tDur=RP.tDur)

            if dim == 3:
                t = torch.sqrt(torch.sum(torch.pow(pData.projPos - ps[i, :], 2)))
            else:
                t = torch.sqrt(torch.sum((pData.projPos - ps[i, :])**2) + torch.tensor(RP.zs[0])**2)

            tau = (t * 2) / torch.tensor(RP.c, requires_grad=True)


            pData.wfm = torchTimeDelay(RP.transmitSignal, torch.tensor(RP.Fs, requires_grad=True),
                                           tau, RP)


            pData.RCTorch(RP.transmitSignal)
            RP.projDataArray.append(pData)
