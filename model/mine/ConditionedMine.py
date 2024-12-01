# [Imports]
import torch
import torch.nn as nn
from typing import Union
from model.common.ConvBlocks import Downsample1d, Upsample1d, Conv1dBlock, ConditionalResidualBlock1D
from model.common.SinusoidalPosEmbedding import SinusoidalPosEmb

class MINEConditionalNet(nn.Module):
    def __init__(self,
                 input_dim,
                 global_cond_dim,
                 diffusion_step_embed_dim=64 * 2,
                 down_dims=[128, 256, 512],
                 kernel_size=5,
                 n_groups=8):
        """
        input_dim: Dim de las acciones.
        global_cond_dim: Dim del condicionamiento global aplicado con FiLM
        diffusion_step_embed_dim: Tamaño del encoding posicional
        down_dims: Tamaño de canal para cada nivel de UNet.
        kernel_size: Tamaño de kernel para convolución
        n_groups: Número de grupos para GroupNorm
        """

        super().__init__()
        all_dims = [input_dim * 2] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim

        self.ma_et = None

        # Encoder de pasos de difusión
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        # Modificación para adaptar el cálculo de T(X_t, X_t+1)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size, n_groups=n_groups),
        ])

        # Downsampling
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if ind < len(in_out) - 1 else nn.Identity()
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        
        # Input is last hidden * 4

        # Capa final adaptada a MINE
        self.final_linear = nn.Sequential(
            nn.Linear(down_dims[-1] * 4, down_dims[-2]),
            nn.ReLU(),
            nn.Linear(down_dims[-2], down_dims[-3]),
            nn.ReLU(),
            nn.Linear(down_dims[-3], 1)
        )

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, X_t: torch.Tensor, epsilon: torch.Tensor, timestep: Union[torch.Tensor, float, int], global_cond=None):
        """
        X_t, X_t+1: Tensores de entrada de forma (B, T, input_dim)
        timestep: (B,) o int, paso de difusión
        global_cond: (B, global_cond_dim)
        salida: Valor escalar T(X)
        """
        # Combina X_t y X_t+1
        sample = torch.cat([X_t, epsilon], dim=-1)
        sample = sample.moveaxis(-1, -2)  # Cambia a (B, C, T)

        # Codificación del paso de tiempo
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        
        # Capa final para obtener T(X)
        x = x.view(x.size(0), -1)
        T_x = self.final_linear(x)

        return T_x.squeeze(-1)  # Retorna T(X) como un escalar
