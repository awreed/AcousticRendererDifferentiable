import torch

class VectorDistribution:
    def __init__(self, T):
        self.T = T
        self.mean = self.calcMean(self.T)

    def calcMean(self, T):
        # index 1 contains the other dim if complex array
        N = list(T.shape)[0]
        #absT = torch.abs(T)
        mag = torch.norm(T, p=1)
        normVec = T / mag
        indexVec = torch.linspace(1, N, N)
        return torch.sum(indexVec * normVec)
