
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

import time
import torch
from torch import nn, optim
import numpy as np
import numpy
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from torch.autograd.functional import jvp

from scipy import stats

import pandas as pd
import os
import h5py
import math
from aux_function import *
from lie_lee import *
from transforms_3d import *

class ResBlock(EquivariantModule):

    def __init__(self, in_type: FieldType, channels: int, out_type: FieldType = None, stride: int = 1, features: str = '2_96'):

        super(ResBlock, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if features == 'ico':
            L = 2
            grid = {'type': 'ico'}
        elif features == '2_96':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 4}
        elif features == '2_72':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 3}
        elif features == '3_144':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 6}
        elif features == '3_192':
            L = 3 #3 Originally this was 3. Now changed to 5 for experimentation.
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        #print(f'\nFourier bandlimit (L): {L}')
        #print(f"Grid sampling method: {grid['type']}")
        #print(f"Grid sample resolution (N): {grid['N']}")
        so3: SO3 = self.in_type.fibergroup
        #####

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))
        #print(f"\nNo. of samples for discrete Fourier Transform: {S}\n")

        #  We try to keep the width of the model approximately constant
        _channels = channels / S
        _channels = int(round(_channels))
        #print(f"\nModel width (_channels): {_channels}\n")

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=5, padding=2, padding_mode='circular', bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, out_type, kernel_size=5, padding=2, stride=stride, padding_mode='circular', bias=False, initialize=False),
        )

        # self.deconv = R3ConvTransposed(self.out_type, self.out_type, kernel_size=3, padding=1, stride=stride, output_padding=stride-1, bias=False, initialize=False)
        # self.reconv = R3Conv(self.out_type, self.out_type, kernel_size=3, padding=1, stride=1, bias=False, initialize=False)

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, padding_mode='circular', bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type

        # res = self.res_block(input)
        # res = self.deconv(res)
        # res = self.reconv(res)

        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape


class SE3CNN(nn.Module):

    def __init__(self, pool: str = "snub_cube", res_features: str = '2_96', init: str = 'delta'):

        super(SE3CNN, self).__init__()

        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = init

        layer_types = [
            (FieldType(self.gs, [self.build_representation(2)] * 3), 100),
            (FieldType(self.gs, [self.build_representation(3)] * 2), 240),
            (FieldType(self.gs, [self.build_representation(3)] * 6), 240),
            (FieldType(self.gs, [self.build_representation(3)] * 12), 480),
            (FieldType(self.gs, [self.build_representation(3)] * 8), None),
        ]
        #print(f'Layer type: {layer_types[0][0]}, {layer_types[0][1]}, {layer_types[1][0]}')
        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, stride=1, padding_mode='circular', bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], stride=1, features=res_features)
            )

        # For pooling, we map the features to a spherical representation (bandlimited to freq 2)
        # Then, we apply pointwise ELU over a number of samples on the sphere and, finally, compute the average
        # # (i.e. recover only the frequency 0 component of the output features)
        if pool == "icosidodecahedron":
            # samples the 30 points of the icosidodecahedron
            # this is only perfectly equivarint to the 12 tethrahedron symmetries
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
            # samples the 24 points of the snub cube
            # this is perfectly equivariant to all 24 rotational symmetries of the cube
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")

        ftgpool = QuotientFourierELU(self.gs, (False, -1), 128, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)

        final_features = ftgpool.in_type
        blocks += [
            R3Conv(layer_types[-1][0], final_features, kernel_size=5, padding=2, padding_mode='circular',  bias=False, initialize=False),
            ftgpool,
        ]
        C = ftgpool.out_type.size
        self.blocks = SequentialModule(*blocks)


        H = 256
        self.classifier = nn.Sequential(
            nn.Linear(128*3*3*3, H, bias=False),
            #nn.Linear(C, H, bias=False),

            #nn.BatchNorm1d(H, affine=True),
            nn.LayerNorm(H),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H, H // 2, bias=False),

            #nn.BatchNorm1d(H // 2, affine=True),
            nn.LayerNorm(H // 2),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H//2, 10, bias=True),
        )

    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):

        input = GeometricTensor(input, self.in_type)

        features = self.blocks(input)
        #print(f"Shape before reshape: {features.shape}")

        shape = features.shape
        k = shape[2]
        # features = features.tensor.view(shape[0], -1)
        features = torch.mean(features.tensor, dim=1, keepdim=True)
        features = features.view(-1, 1, k, k, k)
        #features = features.tensor.reshape(shape[0], shape[1])
        out = features
        #out = self.classifier(features)
        #print(f'Output shape is {out.shape}')
        # out = out.view(shape)

        return out

def e_lee(model, imgs):
    """ Computes the Expected Equivariance Error (E[|Lf|^2]/d_out) w.r.t. rotation. """

    lie_derivative = translation_lie_deriv_3d(model, imgs)

    # Compute squared norm of Lie derivative
    equivariance_error = lie_derivative.pow(2).mean()

    print(f'\nLie Derivative: {lie_derivative.mean().item():.6f}')
    print(f"Equivariance Error: {equivariance_error.item():.6f}\n")

    lie_derivative_np = lie_derivative.detach().cpu().numpy()
    with h5py.File("lie_derivative.h5", "w") as f:
        f.create_dataset("/data", data=lie_derivative_np)

    
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Name: {torch.cuda.get_device_name()}\n')
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # Paths to the training and testing directories
    train_directory = './../../escnn_trial_data/67files/train_rotated_data/'
    test_directory = './../../escnn_trial_data/67files/test_2/plain/'
    data_dir = './../../escnn_trial_data/trainData/plain/plainExperiment/'

    with h5py.File(test_directory+'p_32_Cr_25_Fe_12.5_Co_37.5_Ni_25_temp_5000_run_3_snap_6_sad.h5','r') as file:
        data = file['/data'][:]
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(1,1,*data.shape)
        data = data.to(device)

    # Build and initialize the SE(3) equivariant model
    model = SE3CNN(pool='snub_cube', res_features='2_96', init='he').to(device)
    model.init()
    
    imgs = data

    #equivariance_error = e_lee(model, imgs)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1
    output_save_str = 'plainLieDerivative'
    ckpt_path = './models_plainLieDerivative/'

    load_model = False
    
    if load_model == True:
        model, optimizer, start_epoch = load_checkpoint_model(model, optimizer, ckpt_path)
        additional_epochs = 1
        num_epochs = additional_epochs
        e_lee(model, imgs)
    else:
        model = model
        optimizer = optimizer
        num_epochs = num_epochs
        start_epoch = 0
        e_lee(model, imgs)
    
    #new_train(model=model, device=device, data_dir=data_dir, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, ckpt_path=ckpt_path, loss_save_str=output_save_str, start_epoch=start_epoch)
    #test_eq_net(test_directory=test_directory, device=device, model=model, output_save_str=output_save_str, criterion=criterion)              
