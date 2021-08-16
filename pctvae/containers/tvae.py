import torch
import wandb
from pctvae.utils.vis import plot_traversal_recon

class TVAE(torch.nn.Module):
    def __init__(self, z_encoder, u_encoder, decoder, grouper):
        super(TVAE, self).__init__()
        self.z_encoder = z_encoder
        self.u_encoder = u_encoder
        self.decoder = decoder
        self.grouper = grouper

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u, kl_u, _, _ = self.u_encoder(x)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z

    def normalize_weights(self, including_grouper=False): 
        with torch.no_grad():
            if including_grouper:
                self.grouper.normalize_weights()
            self.z_encoder.normalize_weights()
            self.u_encoder.normalize_weights()
            self.decoder.normalize_weights()

    def plot_decoder_weights(self, wandb_on=True):
        self.decoder.plot_weights(name='Decoder Weights', wandb_on=wandb_on)

    def plot_encoder_weights(self, wandb_on=True):
        self.z_encoder.plot_weights(name='Z-Encoder Weights', wandb_on=wandb_on)
        self.u_encoder.plot_weights(name='U-Encoder Weights', wandb_on=wandb_on)

    def plot_capsule_traversal(self, x, s_dir, e, wandb_on=True):
        assert hasattr(self.grouper, 'n_caps')
        assert hasattr(self.grouper, 'cap_dim')
        z, u, s, probs_x, _, _, _ = self.forward(x)
        x_traversal = [x.cpu().detach(), probs_x.cpu().detach()]

        s_caps = s.view(-1, self.grouper.n_caps, self.grouper.cap_dim, *s.shape[2:])
        for i in range(self.grouper.cap_dim):
            s_rolled = torch.roll(s_caps, shifts=i, dims=2)
            x_recon, _ = self.decoder(s_rolled.view(s.shape), x)
            x_traversal.append(x_recon.cpu().detach())
        
        plot_traversal_recon(x_traversal, s_dir, e, self.grouper.n_t, wandb_on=wandb_on)

    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            u, kl_u, log_q_u, log_p_u = self.u_encoder(x)
            s = self.grouper(z, u)
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_u.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_u.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate



class PCTVAE(TVAE):
    def __init__(self, z_encoder, u_encoder, decoder, grouper, bsz, n_t, n_caps, cap_dim, train_delta_t):
        super(PCTVAE, self).__init__(z_encoder, u_encoder, decoder, grouper)
        self.bsz = bsz
        self.n_t = n_t
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.s_shape = (bsz, n_t, n_caps, cap_dim, 1, 1)
        self.train_delta_t = train_delta_t

    def forward(self, x, delta_t=None):
        z, kl_z, _, _ = self.z_encoder(x)
        u, kl_u, _, _ = self.u_encoder(x)
        s = self.grouper(z, u)
        
        s_caps = s.view(self.s_shape) # s_0 ... s_17
        if delta_t is None:
            delta_t = self.train_delta_t
        s_rolled = torch.roll(s_caps, shifts=(delta_t, delta_t), dims=(1,3)).view(self.bsz*self.n_t, -1, 1, 1)
        # s_rolled = s_rolled[:, 1:].reshape(self.bsz*(self.n_t_out-1), -1, 1, 1)

        # s_17, s_0, ..., s_16 
        # x_0, x_1, ..., x_17

        probs_x, neg_logpx_z = self.decoder(s_rolled, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z

    def plot_capsule_traversal(self, x, s_dir, e, wandb_on=True):
        assert hasattr(self.grouper, 'n_caps')
        assert hasattr(self.grouper, 'cap_dim')
        z, u, s, probs_x, _, _, _ = self.forward(x, delta_t=0)
        x_traversal = [x.cpu().detach(), probs_x.cpu().detach()]

        s_caps = s.view(-1, self.grouper.n_caps, self.grouper.cap_dim, *s.shape[2:])
        for i in range(self.grouper.cap_dim):
            s_rolled = torch.roll(s_caps, shifts=i, dims=2)
            x_recon, _ = self.decoder(s_rolled.view(s.shape), x)
            x_traversal.append(x_recon.cpu().detach())
        
        plot_traversal_recon(x_traversal, s_dir, e, self.grouper.n_t, wandb_on=wandb_on)


    def get_IS_estimate(self, x, n_samples=100, delta_t=0):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            u, kl_u,log_q_u, log_p_u = self.u_encoder(x)
            s = self.grouper(z, u)
            
            s_caps = s.view(self.s_shape) # s_0 ... s_17 
            s_rolled = torch.roll(s_caps, shifts=(delta_t, delta_t), dims=(1,3)).view(self.bsz*self.n_t, -1, 1, 1)

            # s_8, ..., s_17 
            # x_9, ..., x_18

            probs_x, neg_logpx_z = self.decoder(s_rolled, x)

            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_u.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_u.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate




class VAE(TVAE):
    def get_IS_estimate(self, x, n_samples=100, delta_t=0):
        log_likelihoods = []

        if delta_t > 0:
            return torch.tensor([0.0])

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            s = self.grouper(z, torch.zeros_like(z))
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u = torch.zeros_like(z)
        kl_u = torch.zeros_like(kl_z)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z

    def normalize_weights(self, including_grouper=False): 
        with torch.no_grad():
            self.z_encoder.normalize_weights()
            self.decoder.normalize_weights()