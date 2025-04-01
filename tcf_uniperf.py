import sys
import os
current_dir = os.getcwd()
sys.path.insert(0, current_dir)
upper_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, upper_dir)

# Add the performer_pytorch_main directory to path
performer_path = os.path.join(upper_dir, 'baseline_models', 'performer_pytorch_main')
print("[INFO] Adding Performer PyTorch path: ", performer_path)
sys.path.insert(0, performer_path)

import matplotlib.pyplot as plt
from Dataset import *
from trainer import BaseTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
import argparse
import multiprocessing
from utils import dict2namespace, count_parameters, LpLoss
from torch.utils.data import DataLoader, Subset
from performer_pytorch.performer_pytorch import Performer
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import StepLR
from libs.basics import PreNorm, MLP, masked_instance_norm
from libs.conv_block import conv_layer_circular_padding, convtranspose_layer_circular_padding

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the configuration file
cur_folder = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(cur_folder, 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
config = dict2namespace(config)

# Calculate spatial resolution early
nx = config.data.resolution[0] // config.data.subsample_rate
ny = config.data.resolution[1] // config.data.subsample_rate

# in_dim = 3
# out_dim = 3

# Define the TCF3D Performer model
class TCF3DUnitaryPerformerModel(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 nx, ny,
                 dim=192, 
                 depth=6, 
                 heads=8, 
                 dim_head=64,
                 ff_mult=4, 
                 ff_dropout=0.0, 
                 attn_dropout=0.0,
                 latent_multiplier=1.0):
        super(TCF3DUnitaryPerformerModel, self).__init__()
        
        # Store grid dimensions
        self.nx = nx
        self.ny = ny
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.latent_multiplier = latent_multiplier

        # Input projection
        self.to_in = nn.Linear(in_dim, dim, bias=True)
        
        # Position embedding for full grid size
        self.pos_embedding = nn.Parameter(torch.randn(1, nx*ny, dim) * 0.02)
        
        # Performer main body
        self.performer = Performer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            causal=False,
            ff_mult=ff_mult,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            use_scalenorm=True
        )

        self.forward_dim = int(dim * self.latent_multiplier)
        self.latent_dim = dim * int(self.latent_multiplier)
        self.expand_latent = nn.Linear(dim, self.forward_dim, bias=False)
        self.forward_op = nn.Linear(self.forward_dim, self.forward_dim, bias=False)

        self.init_fwd_weights()  # Initialize unitary weights

        self.simple_to_out = nn.Sequential(
            ### New components
            nn.Linear(self.forward_dim, self.latent_dim, bias=True),
            ### 
            Rearrange('b nx ny c -> b c nx ny'),
            nn.GroupNorm(num_groups=int(8 * int(self.latent_multiplier)), num_channels=self.latent_dim),
            conv_layer_circular_padding(self.latent_dim, self.dim, kernel_size=9, stride=1, 
                                        groups=int(8 * int(self.latent_multiplier))),
            nn.GELU(),
            conv_layer_circular_padding(self.dim, self.dim // 2, kernel_size=7, stride=1,
                                        groups=int(8 * int(self.latent_multiplier))),
            nn.GELU(),
            conv_layer_circular_padding(self.dim // 2, self.dim // 4, kernel_size=5, stride=1,
                                        groups=int(8 * int(self.latent_multiplier))),
            nn.GELU(),
            nn.Conv2d(self.dim // 4, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def init_fwd_weights(self) -> None:
        print("[INFO] Initializing unitary forward operator")
        # SVD decomposition
        U, _, V = torch.svd(self.forward_op.weight)
        # Whiten the weights
        W_whiten = torch.matmul(U, V.t())
        self.forward_op.weight.data = W_whiten

    def forward_process(self, u):
        return self.forward_op(u)

    def consistant_loss(self):
        G_fwd = self.forward_op.weight
        G_star = G_fwd.t()

        dim = self.forward_dim
        # Huichison trace estimation
        num_samples = dim * int(np.sqrt(dim)) 
        device = next(self.parameters()).device
        v = torch.randn(num_samples, dim).to(device)
        
        quad_form1 = torch.einsum('bi,ij,bj->b', v, G_star @ G_fwd, v) / dim
        # Estimate expectations
        E_quad1 = quad_form1.mean()
        # Compute the unitary loss
        loss = torch.norm(E_quad1 - 1, p=2)
        return loss
    

    def forward(self, x, pos_lst=None, latent_steps=1):
        b, nx, ny, c = x.shape
        
        # Initial projection
        x = self.to_in(x)
        
        # Reshape to sequence format and add position embedding
        x_flat = rearrange(x, 'b nx ny c -> b (nx ny) c')
        x_flat = x_flat + self.pos_embedding[:, :nx*ny, :]
        
        # Apply Performer
        x_flat = self.performer(x_flat)
        
        # Reshape back for further processing
        x = rearrange(x_flat, 'b (nx ny) c -> b nx ny c', nx=nx, ny=ny)
        # print('x shape before forward expansion:', x.shape) torch.Size([4, 192, 192, 128])
        # Apply unitary forward operator
        x = self.expand_latent(x)
        
        # Apply forward process for specified number of latent steps
        u_lst = []
        u = x
        for _ in range(latent_steps):
            u = self.forward_process(u)
            u_lst.append(u)
        
        u = torch.cat(u_lst, dim=0)
        
        # Output projection with scale parameter
        out = self.simple_to_out(u)
        # out = out * self.scale_param
        out = rearrange(out, '(b t) c nx ny -> b t nx ny c', nx=nx, ny=ny, t=latent_steps)
        
        return out


    def compute_loss(self, x, pos_lst=None, latent_steps=1):
        return self.forward(x, pos_lst, latent_steps)


class TCF3D_Trainer_UnitaryPerformer(BaseTrainer):
    def __init__(self, config, grid_lst, train_loss, test_loss,  constraint_weight=1):
        super().__init__(config)
        self.grid_lst = grid_lst
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.constraint_weight = constraint_weight
    
    def train_function(self, model, train_loader, optimizer):
        train_loss = {"Total": [], 
                    "Fwd": [],        
                    "Cons": []}
        device = next(model.parameters()).device
        num_batches = len(train_loader)
        model.train()
        
        for idx, batch in enumerate(train_loader):
            if idx % 100 == 0:
                print("[INFO] {}/{} ... ".format(idx, num_batches))
                
            optimizer.zero_grad()
            
            x, x_next = batch
            x, x_next = x.to(device), x_next.to(device)

            x_next_hat = model.compute_loss(x, self.grid_lst)
            cons = model.consistant_loss()
            loss_fwd = self.train_loss(x_next_hat[:, 0], x_next)

            # loss = s_fwd*loss_fwd + s_bwd*loss_bwd + s_cons*cons
            loss = loss_fwd + self.constraint_weight*cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss["Total"].append(loss.item())
            train_loss["Fwd"].append(loss_fwd.item())
            train_loss["Cons"].append(cons.item())
        
        train_loss = {key: np.mean(value) for key, value in train_loss.items()}
        return train_loss

    def test_function(self, model, test_loader):
        test_loss = []
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x, self.grid_lst)
                loss = self.test_loss(y_hat[:, 0], y)
                test_loss.append(loss.item())

        # Create 2x3 visualization grid to match FactFormer
        fig, ax = plt.subplots(2, 3, figsize=(16, 11))
        
        # Plot ground truth for all 3 channels
        for i in range(3):
            cbar_1 = ax[0][i].imshow(y[0,...,i].cpu().numpy())
            fig.colorbar(cbar_1, ax=ax[0][i], orientation='horizontal')
        ax[0][0].set_ylabel('Ground Truth')
        
        # Plot predictions for all 3 channels
        with torch.no_grad():
            sample = model(x, self.grid_lst).cpu().detach().numpy()
        for i in range(3):
            cbar_2 = ax[1][i].imshow(sample[0,0,...,i])
            fig.colorbar(cbar_2, ax=ax[1][i], orientation='horizontal')
        ax[1][0].set_ylabel("Prediction")
        
        plt.savefig(self.save_plot_path + "/test_{}.png".format(self.cur_epoch))
        plt.close(fig)

        return np.mean(test_loss)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TCF3D with Performer')
    parser.add_argument('--dim', type=int, default=config.model.dim, help='Model dimension')
    parser.add_argument('--depth', type=int, default=config.model.depth, help='Model depth')
    parser.add_argument('--heads', type=int, default=config.model.heads, help='Number of attention heads')
    parser.add_argument('--dim_head', type=int, default=config.model.dim_head, help='Dimension per head')
    args = parser.parse_args()

    print("[INFO] Number of CPU cores: ", multiprocessing.cpu_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device: {}".format(device))
    if device.type == "cuda":
        print("[INFO] CUDA Device: {}".format(torch.cuda.get_device_properties(device)))
    
    # Create position encoding list similar to FactFormer
    pos_lst = [torch.linspace(0, 2*np.pi, nx).to(device).unsqueeze(-1), 
               torch.linspace(0, 2*np.pi, ny).to(device).unsqueeze(-1)]
    
    # Initialize model
    model = TCF3DUnitaryPerformerModel(
        in_dim=config.model.in_dim, 
        out_dim=config.model.out_dim, 
        nx=nx,  
        ny=ny,  # Explicitly passing both dimensions
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        ff_dropout=config.model.ff_dropout,
        attn_dropout=config.model.attn_dropout,
        latent_multiplier=config.model.latent_multiplier
    )
    model.to(device)
    print(f"[INFO] Number of parameters: {count_parameters(model)}")

    # Set and Load your datase here
    train_loader = DataLoader('your settings')
    test_loader = DataLoader('your settings')
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.training.lr, 
                                weight_decay=config.training.weight_decay)
    scheduler = StepLR(optimizer, 
                      step_size=config.training.step_size, 
                      gamma=config.training.gamma)
    
    # Setup losses
    train_loss = LpLoss(size_average=False)
    test_loss = LpLoss(size_average=False)

    # Initialize trainer
    trainer = TCF3D_Trainer_UnitaryPerformer(config, pos_lst,         
                                             train_loss, 
                                            test_loss,
                                            constraint_weight=config.training.unitary_loss_weight
                                        )
    
    # Start training
    trainer.main(model, train_loader, test_loader, optimizer, scheduler)