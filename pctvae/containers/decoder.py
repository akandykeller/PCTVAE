import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from pctvae.utils.vis import plot_filters

class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError

    def plot_weights(self, name, wandb_on=True):
        name_w_idx = name + '_L{}'
        for idx, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                plot_filters(layer.weight, name_w_idx.format(idx), wandb_on=wandb_on)

    def normalize_weights(self):
        with torch.no_grad():   
            for l in self.model:
                if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
                    norm_shape = (l.weight.shape[0], -1)
                    norms = torch.sqrt(l.weight.view(norm_shape).pow(2).sum([-1], keepdim=True))
                    l.weight.view(norm_shape).div_(norms)
                elif isinstance(l, nn.Linear):
                    norm_shape = (l.weight.shape[0], l.weight.shape[1])
                    norms = torch.sqrt(l.weight.view(norm_shape).pow(2).sum([-1], keepdim=True))
                    l.weight.view(norm_shape).div_(norms)

class Bernoulli_Decoder(Decoder):
    def __init__(self, model):
        super(Bernoulli_Decoder, self).__init__(model)

    def forward(self, z, x):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x)
        neg_logpx_z = -1 * p.log_prob(x)

        return probs_x, neg_logpx_z


class Causal_Bernoulli_Decoder(Decoder):
    def __init__(self, model, n_t, n_t_out):
        super(Causal_Bernoulli_Decoder, self).__init__(model)
        self.n_t = n_t
        self.n_t_out = n_t_out

    def forward(self, z, x, decode_current=False):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x)
        n_x_keep = self.n_t_out if not decode_current else self.n_t_out + 1
        x_crop = x.view(-1, self.n_t, *probs_x.shape[1:])[:, -n_x_keep:].reshape(probs_x.shape)
        neg_logpx_z = -1 * p.log_prob(x_crop)

        return probs_x, neg_logpx_z

class Gaussian_Decoder(Decoder):
    def __init__(self, model, scale=0.0):
        super(Gaussian_Decoder, self).__init__(model)
        self.logscale = nn.Parameter(torch.tensor([scale]))

    def forward(self, z, x):
        mu_x = self.model(z)
        p = Normal(loc=mu_x, scale=torch.exp(self.logscale))
        neg_logpx_z = -1 * p.log_prob(x)
        return mu_x, neg_logpx_z