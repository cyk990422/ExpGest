import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        #assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        #self.rot2xyz = self.model.rot2xyz
        #self.translation = self.model.translation
        #self.njoints = self.model.njoints
        #self.nfeats = self.model.nfeats
        #self.data_rep = self.model.data_rep
        #self.cond_mode = self.model.cond_mode

    def forward(self, x_body,x_hand, timesteps, y=None):
        #cond_mode = self.model.cond_mode
        #assert cond_mode in ['text', 'action']
        detached_tensor_dict = {key: value for key, value in y.items()}
        #y_uncond = deepcopy(detached_tensor_dict)
        y_uncond = detached_tensor_dict
        y_uncond['uncond'] = True
        out_body,out_hand = self.model(x_body,x_hand,timesteps,**y)
        out_uncond_body,out_uncond_hand = self.model(x_body,x_hand,timesteps, **y_uncond)
        y['scale']=torch.ones(1, device='cuda') * 3.0
        #return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
        return out_uncond_body + (y['scale'].view(-1, 1, 1, 1) * (out_body - out_uncond_body)),out_uncond_hand + (y['scale'].view(-1, 1, 1, 1) * (out_hand - out_uncond_hand))

#model_output_body,model_output_hand = model(x_body,x_hand, self._scale_timesteps(t), **model_kwargs)