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

    RP_GT.defineProjectorPosGrid(xStep=.1, yStep=.1)

    RP_EST = RenderParameters()
    RP_EST.generateTransmitSignal()

    RP_EST.defineProjectorPosGrid(xStep=.1, yStep=.1)

    # One point source is GT, the other we will try and evolve towards the correct answer
    # psEST = torch.ones((1, 3), requires_grad=True)
    ps_GT = torch.tensor([[-.8, -.8]], requires_grad=True).cuda()
    psEST = torch.tensor([[.8, .8]], requires_grad=True).cuda()

    BF_GT = Beamformer(sceneDimX=[-3, 3], sceneDimY=[-3, 3], sceneDimZ=[0, 0], nPix=[256, 256, 1], dim=2)
    simulateSASWaveformsPointSource(RP_GT, ps_GT)
    RP_GT.freeHooks()
    GT = BF_GT.beamformTest(RP_GT)
    #realABS = torch.abs(GT[:, 0])
    #imagABS = torch.abs(GT[:, 1])
    GT_ABS = compABS(GT)

    #GT_abs_norm = GT_ABS / torch.norm(GT_ABS, p=1)

    x_vals = torch.unique(BF_GT.pixPos[:, 0]).numel()
    y_vals = torch.unique(BF_GT.pixPos[:, 1]).numel()

    GT_ABS_XY = GT_ABS.view(x_vals, y_vals)
    #maxVal = torch.max(GT_ABS_XY)
    #meanVal = torch.mean(GT_ABS_XY)
    #stdVal = torch.std(GT_ABS_XY)
    #k = torch.tensor([2.0])
    #thresh = meanVal + k * stdVal
    power = 4
    GT_ABS_XY_OH = (GT_ABS_XY**power)/(torch.norm(GT_ABS_XY**power, p=1))
    #print(GT_ABS_XY_OH)
    #print(thresh)
    #mask = (GT_ABS_XY > thresh).float()
    #GT_ABS_XY = GT_ABS_XY.clone() * mask
    #GT_ABS_XY[(GT_ABS_XY < thresh).detach()] = 0
    #norm = torch.norm(GT_ABS_XY, p=1)
    #GT_ABS_XY_OH = (GT_ABS_XY) / norm
    #print(GT_ABS_XY)
    #mask = (GT_ABS_XY[:, :] >= meanVal + stdVal * k).detach().float()
    #GT_ABS_XY_OH = (GT_ABS_XY * mask) / torch.norm((GT_ABS_XY * mask), p=1)

    GT_abs_EX = VectorDistribution(torch.sum(GT_ABS_XY_OH, dim=0))
    GT_abs_EY = VectorDistribution(torch.sum(GT_ABS_XY_OH, dim=1))

    means_gt = torch.stack((GT_abs_EX.mean, GT_abs_EY.mean))
    print(means_gt)

    #print(compABS(GT).max())
    # plt.cla()
    #plt.scatter(BF_GT.pixPos[:, 0].detach().cpu().numpy(), BF_GT.pixPos[:, 1].detach().cpu().numpy(),
                #c=GT_ABS_XY_OH.view(-1, 1).detach().cpu().numpy())
    #plt.show()

    plt.imshow(GT_ABS_XY_OH.detach().cpu().numpy())
    plt.savefig("pics\GT.png")
    #plt.show()

    val = 0
    thresh = .01
    optimizer = torch.optim.SGD([psEST], lr=.0005, momentum=0.0)
    criterion = torch.nn.PairwiseDistance(p=1.0)
    L1_weight = 1
    mean_weight = 1
    loss_val = 10000

    BF = Beamformer(sceneDimX=[-3, 3], sceneDimY=[-3, 3], sceneDimZ=[0, 0], nPix=[256, 256, 1], dim=2)
    count = 0

    while loss_val > thresh:
        simulateSASWaveformsPointSource(RP_EST, psEST)
        EST = BF.beamformTest(RP_EST)
        EST_ABS = compABS(EST)
        #EST_ABS_norm = EST_ABS/torch.norm(EST_ABS, p=1)
        EST_ABS.register_hook(lambda grad: grad.clamp_min_(0.0))
        EST_XY = EST_ABS.view(x_vals, y_vals)
        #h = EST_XY.register_hook(lambda x: print("Scene " + str(x)))
        #RP_EST.hooks.append(h)

        #maxVal = torch.max(EST_XY)
        #meanVal = torch.mean(EST_XY)
        #stdVal = torch.std(EST_XY)
        #k = torch.tensor([2.0])
        #thresh = meanVal + k * stdVal

        #mask = (EST_XY > thresh).float()
        #EST_XY[(EST_XY < thresh).detach()] = 0
        #EST_XY = EST_XY.clone() * mask
        power = 4
        norm = (torch.norm(EST_XY ** power, p=1))
        EST_XY_OH = (EST_XY ** power) / norm
       #EST_XY_OH.register_hook(lambda x: print("EST_XY_OH: " + str(x)))

        #mask = (EST_XY >= thresh).detach().float()
        #mask = (EST_XY[:, :] > meanVal + k * stdVal)
        #mask.requires_grad = True
        #mask.register_hook(lambda x: print("Mask: " + str(x)))
        #norm = torch.norm(EST_XY, p=1)
        #EST_XY_norm = (EST_XY) / norm
        #print(EST_XY_norm)
        #mask = torch.ones(EST_XY.size()[0], dtype=torch.float32)
        #print(mask)
        #sim_vec = torch.nonzero((EST_XY >= thresh)*mask)
        #print(sim_vec)
        #EST_XY_OH = (EST_XY*mask)/torch.norm((EST_XY * mask), p=1)
        #print(EST_XY_OH)
        #EST_XY_OH.register_hook(lambda x: print("EST_XY_OH: " + str(x)))

        EST_EX = VectorDistribution(torch.sum(EST_XY_OH, dim=0))
        EST_EY = VectorDistribution(torch.sum(EST_XY_OH, dim=1))
        #EST_EX.mean.register_hook(lambda x: print("EST_EX: " + str(x)))
        #EST_EY.mean.register_hook(lambda x: print("EST_EY: " + str(x)))

        means_est = torch.stack((EST_EX.mean, EST_EY.mean))
        #h = means_est.register_hook(lambda x: print("means est:" + str(x)))
        print(means_est)
        #print(means_est)
        #print(means_gt)
        #L1_loss = L1_weight * (criterion(EST[:, 0].unsqueeze(0), GT[:, 0].unsqueeze(0)) +
                               #criterion(EST[:, 1].unsqueeze(0), GT[:, 1].unsqueeze(0)))

        mean_loss = torch.sqrt(torch.sum((means_gt - means_est)**2))
        loss = mean_loss
        #+ mean_loss
        #mean_loss.register_hook(lambda x: print("mean_loss" + str(x)))

        loss.backward(retain_graph=True)
        print("Mean Loss: " + str(mean_loss.data) + "\t" + "L1_loss: " + str(
            mean_loss.data) + "\t" + "Point estimate: " + str(psEST), "Gradient: " + str(psEST.grad))
        optimizer.step()
        optimizer.zero_grad()
        RP_EST.freeHooks()

        plt.imshow(EST_XY_OH.detach().cpu().numpy())
        plt.savefig('pics\est' + str(count) + ".png")
        #plt.show()

        #fig = plt.scatter(BF.pixPos[:, 0].detach().cpu().numpy(), BF.pixPos[:, 1].detach().cpu().numpy(),
                         # c=compABS(EST).detach().cpu().numpy())
        #plt.savefig('pics\est' + str(count) + ".png")
        count = count + 1
    # plt.show()
