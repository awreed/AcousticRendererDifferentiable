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

torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        print("Fix ur cuda loser")
        dev = "cpu"

    device = torch.device(dev)

    RP_GT = RenderParameters()

    RP_GT.generateTransmitSignal()

    #RP_GT.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=1, rStop=1, zStart=.3, zStop=.3)
    RP_GT.defineProjectorPosGrid()

    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()
    #RP_EST.defineProjectorPos(thetaStart=0, thetaStop=359, thetaStep=1, rStart=1, rStop=1, zStart=.3, zStop=.3)
    RP_EST.defineProjectorPosGrid()

    # One point source is GT, the other we will try and evolve towards the correct answer
    # psEST = torch.ones((1, 3), requires_grad=True)
    ps_GT = torch.tensor([[-0.7, 0.7]], requires_grad=True).cuda()
    psEST = torch.tensor([[.7, -.7]], requires_grad=True).cuda()

    BF_GT = Beamformer(sceneDimX=[-1, 1], sceneDimY=[-1, 1], sceneDimZ=[0, 0], nPix=[128, 128, 1])
    simulateSASWaveformsPointSource(RP_GT, ps_GT)

    GT = BF_GT.beamformTest(RP_GT)
    realABS = torch.abs(GT[:, 0])
    imagABS = torch.abs(GT[:, 1])
    GT_ABS = compABS(GT)

    GT_abs_norm = GT_ABS / torch.norm(GT_ABS, p=1)

    x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
    y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()

    GT_ABS_XY = GT_abs_norm.view(x_vals, y_vals)
    #with torch.no_grad()
    maxVal = torch.max(GT_ABS_XY)
    meanVal = torch.mean(GT_ABS_XY)
    stdVal = torch.std(GT_ABS_XY)
    k = torch.tensor([1.0])
    mask = (GT_ABS_XY[:, :] > meanVal + stdVal*k).float()
    GT_ABS_XY_OH = (GT_ABS_XY * mask)/torch.norm((GT_ABS_XY * mask), p=1)

    #GT_DRC = DRC_Isaac_Vectorized(GT_ABS_XY, torch.median(GT_ABS_XY), 0.2)

    GT_abs_EX = VectorDistribution(torch.sum(GT_ABS_XY_OH, dim=0))
    GT_abs_EY = VectorDistribution(torch.sum(GT_ABS_XY_OH, dim=1))



    # print(compABS(GT).max())
    # plt.cla()
    plt.scatter(BF_GT.pixPos[:, 0].detach().cpu().numpy(), BF_GT.pixPos[:, 1].detach().cpu().numpy(),
                c=compABS(GT).detach().cpu().numpy())
    #plt.show()
    plt.savefig("pics\GT.png")

    # plt.show()

    val = 0
    thresh = .01
    optimizer = torch.optim.SGD([psEST], lr=.0005, momentum=0.0)
    criterion = torch.nn.PairwiseDistance(p=1.0)
    L1_weight = 0
    mean_weight = 1
    loss_val = 10000

    BF = Beamformer(sceneDimX=[-1, 1], sceneDimY=[-1, 1], sceneDimZ=[0, 0], nPix=[128, 128, 1])
    count = 0

    means_gt = torch.stack((GT_abs_EX.mean, GT_abs_EY.mean))

    while loss_val > thresh:
        plt.cla()
        optimizer.zero_grad()

        simulateSASWaveformsPointSource(RP_EST, psEST)
        EST = BF.beamformTest(RP_EST)

        EST_ABS = compABS(EST)
        EST_ABS_norm = EST_ABS/torch.norm(EST_ABS, p=1)
        EST_XY = EST_ABS_norm.view(x_vals, y_vals)
        maxVal = torch.max(EST_XY)
        meanVal = torch.mean(EST_XY)
        stdVal = torch.std(EST_XY)
        k = torch.tensor([1.0])
        mask = (EST_XY[:, :] > meanVal + k*stdVal).float()
        EST_XY_OH = (EST_XY * mask)/torch.norm((EST_XY * mask), p=1)

        EST_EX = VectorDistribution(torch.sum(EST_XY_OH, dim=0))
        EST_EY = VectorDistribution(torch.sum(EST_XY_OH, dim=1))

        L1_loss = L1_weight * (criterion(EST[:, 0].unsqueeze(0), GT[:,0].unsqueeze(0)) +
                               criterion(EST[:,1].unsqueeze(0), GT[:,1].unsqueeze(0)))

        means_est = torch.stack((EST_EX.mean, EST_EY.mean))
        #print(means_est)

        mean_loss = torch.sqrt((means_gt - means_est)**2)

        #mean_loss = torch.tensor([mean_weight * ((EST_EX.mean - GT_abs_EX.mean) ** 2, mean_weight * (EST_EY.mean - GT_abs_EY.mean) ** 2)])
        print("EST Index")
        print(mean_loss)
        print("GT Index")
        print(GT_abs_EX.mean, GT_abs_EY.mean)

        #loss = L1_loss + mean_loss
        #loss_val = loss.data

        mean_loss.backward(torch.FloatTensor([1.0, 1.0]).cuda(), retain_graph=True)

        optimizer.step()
        print("Mean Loss: " + str(mean_loss.data) + "\t" + "L1_loss: " + str(
            L1_loss.data) + "\t" + "Point estimate: " + str(psEST), "Gradient: " + str(psEST.grad))

        fig = plt.scatter(BF.pixPos[:, 0].detach().cpu().numpy(), BF.pixPos[:, 1].detach().cpu().numpy(),
                          c=compABS(EST).detach().cpu().numpy())
        plt.savefig('pics\est' + str(count) + ".png")
        count += 1
    # plt.show()
