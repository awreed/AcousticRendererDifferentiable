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
    #print(numScat)

    for i in range(0, RP.numProj):
        pData = ProjData.ProjData(projPos=RP.projectors[i, :], Fs=RP.Fs, tDur=RP.tDur)
        for j in range(0, numScat):
            t = torch.sqrt(torch.sum((pData.projPos - ps[j, :]) ** 2) + torch.tensor(RP.zs[0]) ** 2)

            tau = (t * 2) / torch.tensor(RP.c, requires_grad=True)

            pData.wfms.append(torchTimeDelay(RP.transmitSignal, torch.tensor(RP.Fs, requires_grad=True),
                                       tau, RP))

        pData.wfm = torch.sum(torch.stack(pData.wfms), 0)
        #print(pData.wfm.shape)
        pData.RCTorch(RP.transmitSignal)
        RP.projDataArray.append(pData)

