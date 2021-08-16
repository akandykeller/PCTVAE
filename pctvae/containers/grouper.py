import torch
import torch.nn as nn
import torch.nn.functional as F

class Grouper(nn.Module):
    def __init__(self, model, padder):
        super(Grouper, self).__init__()
        self.model = model
        self.padder = padder

    def forward(self, z, u):
        raise NotImplementedError

    def normalize_weights(self):
        with torch.no_grad():
            torch.clamp_(self.model.weight, min=0.0)
            w = self.model.weight
            assert (w.shape[-2] > 1 and w.shape[-3] > 1)
            norm_shape = (-1, w.shape[-1])
            norms = w.view(norm_shape).abs().sum([-1], keepdim=True)
            self.model.weight.view(norm_shape).div_(norms)
    
class Chi_Squared_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Chi_Squared_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            capdim = self.model.weight.shape[3]
            W = torch.ones([capdim, capdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

            # For half/speed partial roll
            # W = W.repeat_interleave(repeats=2, dim=0).to('cuda')
            # for r in range(0, capdim):
            #     if r-2 >= 0:
            #         W[r*2, r-2] += 0.5
            #     if r+1 < capdim:
            #         W[r*2, r+1] -= 0.5

            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def get_v(self, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t, self.cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)
        return v

    def forward(self, z, u):
        v = self.get_v(u).reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        return s


class Causal_Capsules_from_Gaussian_1d(Grouper):
    def __init__(self, model, padder, n_caps, u_cap_dim, z_cap_dim, n_transforms,
                 mu_init=1.0, n_off_diag=1, trainable=False, eps=1e-6):
        super(Causal_Capsules_from_Gaussian_1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.u_cap_dim = u_cap_dim
        self.cap_dim = self.z_cap_dim = z_cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        
        wdim = self.model.weight.shape[3]
        self.n_u = wdim // 2
        self.z_cap_dim_cropped = wdim - self.n_u
        self.n_t_out = self.z_cap_dim_cropped

        nn.init.zeros_(self.model.weight)
        with torch.no_grad():
            W = torch.ones([wdim, wdim])
            W = torch.triu(W, -n_off_diag)
            W = W*W.t()

            W[self.n_u:, :] = 0.0
            W = W.tril()
            
            # For half/speed partial roll
            # W = W.repeat_interleave(repeats=2, dim=0).to('cuda')
            # for r in range(0, wdim):
            #     if r-2 >= 0:
            #         W[r*2, r-2] += 0.5
            #     if r+1 < capdim:
            #         W[r*2, r+1] -= 0.5

            self.model.weight.data[0, 0, :, :, 0] = W

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):
        h,w = u.shape[2], u.shape[3]
        u_caps = u.view(-1, self.n_t, self.n_caps, self.u_cap_dim, h*w) # (bsz, t, n_caps, capdim, h*w)
        u_caps = u_caps.permute((0, 2, 1, 3, 4)) # (bsz, n_caps, t, capdim, h*w)
        u_caps = u_caps.reshape(-1, 1, self.n_t, self.u_cap_dim, h*w)
        u_caps = u_caps ** 2.0
        u_caps_padded = self.padder(u_caps)
        v = self.model(u_caps_padded).squeeze(1)
        v = v.view(-1, self.n_caps, self.n_t_out, self.z_cap_dim, h*w)
        v = v.permute((0, 2, 1, 3, 4)) # (bsz, t, n_caps, capdim, h*w)
        
        v = v.reshape(z.shape)
        std = 1.0 / torch.sqrt(v + self.eps)
        s = (z + self.correlated_mean_beta) * std
        return s



class NonTopographic_Capsules1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 mu_init=0, n_off_diag=0, trainable=False, eps=1e-6):
        super(NonTopographic_Capsules1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.eps = eps
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
       
    def forward(self, z, u):
        s = z
        return s
