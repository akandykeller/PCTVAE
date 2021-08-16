import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F

from pctvae.data.mnist import Preprocessor
from pctvae.containers.tvae import VAE
from pctvae.models.mlp import MLP_Encoder, MLP_Decoder
from pctvae.containers.encoder import Gaussian_Encoder
from pctvae.containers.decoder import Bernoulli_Decoder
from pctvae.containers.grouper import NonTopographic_Capsules1d
from pctvae.utils.logging import configure_logging, get_dirs
from pctvae.utils.train_loops import train_epoch, eval_epoch, final_results

def create_model(n_caps, cap_dim, n_transforms):
    s_dim = n_caps * cap_dim
    group_kernel = (1,1,1)
    z_encoder = Gaussian_Encoder(MLP_Encoder(s_dim=s_dim, n_cin=3, n_hw=28),
                                 loc=0.0, scale=1.0)

    u_encoder = None

    decoder = Bernoulli_Decoder(MLP_Decoder(s_dim=s_dim, n_cout=3, n_hw=28))

    grouper = NonTopographic_Capsules1d(
                      nn.ConvTranspose3d(in_channels=1, out_channels=1,
                                          kernel_size=group_kernel, 
                                          padding=(0,0,0),
                                          stride=(1,1,1), padding_mode='zeros', bias=False),
                      lambda x: F.pad(x, (0,0,0,0,0,0), mode='circular'),
                    n_caps=n_caps, cap_dim=cap_dim, n_transforms=n_transforms)
    
    return VAE(z_encoder, u_encoder, decoder, grouper)


def main():
    config = {
        'wandb_on': False,
        'lr': 1e-4,
        'momentum': 0.9,
        'batch_size': 8,
        'max_epochs': 100,
        'eval_epochs': 5,
        'dataset': 'MNIST',
        'train_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        'test_angle_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340', 
        'train_color_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        'test_color_set': '0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340',
        'train_scale_set': '0.60 0.64 0.68 0.72 0.76 0.79 0.83 0.87 0.91 0.95 0.99 1.03 1.07 1.11 1.14 1.18 1.22 1.26',
        'test_scale_set': '0.60 0.64 0.68 0.72 0.76 0.79 0.83 0.87 0.91 0.95 0.99 1.03 1.07 1.11 1.14 1.18 1.22 1.26',
        'pct_val': 0.2,
        'pct_train': 0.8,
        'random_crop': 28,
        'seed': 1,
        'n_caps': 18,
        'cap_dim': 18,
        'n_transforms': 18,
        'train_eq_loss': False,
        'n_is_samples': 10
        }

    name = 'NonT-VAE_MNIST_L=0'

    config['savedir'], config['data_dir'], config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)
    preprocessor = Preprocessor(config)
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders(batch_size=config['batch_size'])

    model = create_model(n_caps=config['n_caps'], cap_dim=config['cap_dim'], n_transforms=config['n_transforms'])
    model.to('cuda')

    log, checkpoint_path = configure_logging(config, name, model)
    # model.load_state_dict(torch.load(load_checkpoint_path))

    optimizer = optim.SGD(model.parameters(), 
                           lr=config['lr'],
                           momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    best_val = -1_000_000_000

    for e in range(config['max_epochs']):
        log('Epoch', e)

        total_loss, total_neg_logpx_z, total_kl, total_eq_loss, num_batches = train_epoch(model, optimizer, 
                                                                     train_loader, log,
                                                                     savepath, e, eval_batches=3000,
                                                                     plot_weights=False,
                                                                     plot_fullcaptrav=True,
                                                                     wandb_on=config['wandb_on'])

        log("Epoch Avg Loss", total_loss / num_batches)
        log("Epoch Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
        log("Epoch Avg KL", total_kl / num_batches)
        log("Epoch Avg EQ Loss", total_eq_loss / num_batches)
        scheduler.step()

        if e % config['eval_epochs'] == 0:
            total_loss, total_neg_logpx_z, total_kl, total_is_estimate, total_eq_loss, num_batches = eval_epoch(model, val_loader, log, savepath, e, 
                                                                                                                n_is_samples=config['n_is_samples'],
                                                                                                                plot_maxact=False, 
                                                                                                                plot_class_selectivity=False,
                                                                                                                plot_cov=False,
                                                                                                                wandb_on=config['wandb_on'])
            val_is_estimate = total_is_estimate / num_batches
            log("Val Avg Loss", total_loss / num_batches)
            log("Val Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
            log("Val Avg KL", total_kl / num_batches)
            log("Val IS Estiamte", val_is_estimate)
            log("Val EQ Loss", total_eq_loss / num_batches)

            if val_is_estimate >= best_val:
                torch.save(model.state_dict(), checkpoint_path)

                best_val = val_is_estimate
                total_loss, total_neg_logpx_z, total_kl, total_is_estimate, total_eq_loss, num_batches = eval_epoch(model, test_loader, log, savepath, e, 
                                                                                                                    n_is_samples=config['n_is_samples'],
                                                                                                                    plot_maxact=False, 
                                                                                                                    plot_class_selectivity=False,
                                                                                                                    plot_cov=False,
                                                                                                                    wandb_on=config['wandb_on'])
                val_is_estimate = total_is_estimate / num_batches
                log("Test Avg Loss", total_loss / num_batches)
                log("Test Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
                log("Test Avg KL", total_kl / num_batches)
                log("Test IS Estiamte", total_is_estimate / num_batches)
                log("Test EQ Loss", total_eq_loss / num_batches)

    model.load_state_dict(torch.load(checkpoint_path))
    is_by_dt = final_results(model, test_loader, log, n_is_samples=10, dt_set=range(0, 9), wandb_on=True)
    print(is_by_dt)
    for dt in is_by_dt:
        log('Final_IS_dt{}'.format(dt), is_by_dt[dt])
    log('Avg IS', sum(list(is_by_dt.values())) / len(is_by_dt))
    
if __name__ == '__main__':
    main()