import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
  
  
# 支持多分类和二分类
class focal_loss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
      super(focal_loss, self).__init__()
      self.class_num = class_num
      self.alpha = alpha
      self.gamma = gamma
      if use_alpha:
          self.alpha = torch.tensor(alpha).cuda()

      self.softmax = nn.Softmax(dim=-1)
      self.use_alpha = use_alpha
      self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)
        #print(prob)
        target_ = torch.zeros(target.size(0),self.class_num).cuda()   #target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)
        #print(target_.double())
        #print(torch.pow(1-prob,self.gamma).double() * prob.log().double())
        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)
        print(prob)
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss