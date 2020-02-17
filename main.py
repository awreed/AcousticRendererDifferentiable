import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.optim
from RenderParameters import RenderParameters
from simulateSASWaveformsPointSource import simulateSASWaveformsPointSource
from Beamformer import *
import pytorch_ssim
import matplotlib.image as mpimg
import time
from VectorDistribution import *
from Complex import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Where you left, need to calc means of each projector waveform, figure out how to loop over all waveforms
# and update the loss with breaking the graph

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    device = torch.device(dev)

    # Data structure for ground truth projector waveforms
    RP_GT = RenderParameters()
    RP_GT.generateTransmitSignal()
    #RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=1, rStop=1, zStart=.3, zStop=.3)
    RP_GT.defineProjectorPosGrid(xStep=0.2, yStep=0.2)

    # Data structure for estimate projector waveforms
    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()
    #RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=3, rStop=3, zStart=.3, zStop=.3)
    RP_EST.defineProjectorPosGrid(xStep=0.2, yStep=0.2)

    # One point source is GT, the other we will try and evolve towards GT
    ps_GT = torch.tensor([[-1.0, -1.0]], requires_grad=True).cuda()
    ps_EST = torch.tensor([[1.0, 1.0]], requires_grad=True).cuda()

    simulateSASWaveformsPointSource(RP_GT, ps_GT)

    # Beamformer to create images from projector waveforms
    BF = Beamformer(sceneDimX=[-2, 2], sceneDimY=[-2, 2], sceneDimZ=[0, 0], nPix=[128, 128, 1], dim=2)
    x_vals = torch.unique(BF.pixPos[:, 0]).numel()
    y_vals = torch.unique(BF.pixPos[:, 1]).numel()

    GT = BF.beamformTest(RP_GT).abs()

    GT_XY = GT.view(x_vals, y_vals)
    #u = torch.mean(GT_XY)
    #sigma = torch.std(GT_XY)
    #k = torch.tensor([18.0])
    #mask = (GT_XY[:, :] >= u + sigma * k).float()
    #filtered_GT = GT_XY * mask

    plt.imshow(GT_XY.detach().cpu().numpy())
    plt.savefig("pics\GT.png")




    val = 0
    thresh = 500
    optimizer = torch.optim.SGD([ps_EST], lr=.00001, momentum=0.9)
    criterion = torch.nn.PairwiseDistance(p=1.0)
    L1_weight = 1
    mean_weight = 1
    loss_val = 10000

    count = 0
    loss = []

    while loss_val > thresh:
        simulateSASWaveformsPointSource(RP_EST, ps_EST)
        for pData_gt, pData_est in zip(RP_GT.projDataArray, RP_EST.projDataArray):
            EX_GT = VectorDistribution(pData_gt.wfmRC.abs())
            EX_EST = VectorDistribution(pData_est.wfmRC.abs())
            #plt.stem(pData_gt.wfmRC.abs().detach().cpu().numpy())
            #plt.show()

            dist = torch.abs(EX_GT.mean - EX_EST.mean)
            loss.append(dist)

        final_loss = torch.sum(torch.stack(loss))
        final_loss.backward(retain_graph=True)
        print(final_loss.data)
        optimizer.step()
        #print(ps_EST.grad)
        optimizer.zero_grad()
        loss.clear()
        EST = BF.beamformTest(RP_EST).abs()

        EST_XY = EST.view(x_vals, y_vals)
        #u = torch.mean(EST_XY)
        #sigma = torch.std(EST_XY)
        #k = torch.tensor([3.0])
        #mask = (EST_XY[:, :] >= u + sigma * k).float()
        #filtered_EST = mask * EST_XY

        plt.imshow(EST_XY.detach().cpu().numpy())
        plt.savefig("pics\est" + str(count)+ ".png")
        count = count + 1

        print(ps_EST)



    # plt.show()
