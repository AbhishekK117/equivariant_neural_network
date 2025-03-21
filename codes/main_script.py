import argparse
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *

import torch
from torch import nn, optim

from scipy import stats
import math
from utility_functions import *

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
            L = 3
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup

        S = len(so3.grid(**grid))

        _channels = channels / S
        _channels = int(round(_channels))

        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        print(f'ResBlock: {in_type.size} -> {res_type.size} -> {self.out_type.size} | {S*_channels}')

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=5, padding=2, padding_mode='circular', bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, out_type, kernel_size=5, padding=2, stride=stride, padding_mode='circular', bias=False, initialize=False),
        )

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
            (FieldType(self.gs, [self.build_representation(2)] * 3), 200),
            (FieldType(self.gs, [self.build_representation(3)] * 2), 480),
            (FieldType(self.gs, [self.build_representation(3)] * 6), 480),
            (FieldType(self.gs, [self.build_representation(3)] * 12), 960),
            (FieldType(self.gs, [self.build_representation(3)] * 8), None),
        ]
        print(f'Layer type: {layer_types[0][0]}, {layer_types[0][1]}, {layer_types[1][0]}')
        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, stride=1, padding_mode='circular', bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], stride=1, features=res_features)
            )

        if pool == "icosidodecahedron":
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
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

        shape = features.shape
        k = shape[2]
        features = torch.mean(features.tensor, dim=1, keepdim=True)
        features = features.view(-1, 1, k, k, k)
        out = features

        return out

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Device Name: {torch.cuda.get_device_name()}')

    train_data_dir = "./../data/" + args.train_case + "/"
    test_data_dir = "./../data/test/" + args.test_case + "/"

    model = SE3CNN(pool='snub_cube', res_features='2_96', init='he').to(device)
    model.init()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epochs
    start_epoch = 0
    output_save_str = args.run_name
    ckpt_path = "./../models/" + output_save_str + "/"

    if args.load_model:
        model, optimizer, start_epoch = load_checkpoint_model(model, optimizer, ckpt_path)
        num_epochs = args.additional_epochs

    train_model(model=model, device=device, data_dir=train_data_dir, num_epochs=num_epochs, optimizer=optimizer, criterion=criterion, ckpt_path=ckpt_path, loss_save_str=args.run_name, start_epoch=start_epoch)

    test_model(test_data_dir, device, model, output_save_str, criterion)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SE(3)-CNN with equivariant architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_case", type=str, default="", help="Which data to use for training?")
    parser.add_argument("--test_case", type=str, default="plain", help="Which data to use for testing?")
    parser.add_argument("--run_name", type=str, default="anonymous", help="What should be the name of this run?")
    parser.add_argument("--load_model", action="store_true", help="Do you want to load any previous model?")
    parser.add_argument("--additional_epochs", type=int, default=0, help="These are the extra epochs to run the model if it the model is loaded using any previous checkpoint")

    args = parser.parse_args()
    main(args)
