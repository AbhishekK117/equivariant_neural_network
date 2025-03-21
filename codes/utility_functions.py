import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from scipy.ndimage import zoom

import pandas as pd
import os
import h5py
from prettytable import PrettyTable
from lie_lee import *
from transforms_3d import *

def pearson_correlation_coefficient(output, rotated):
    output_flat = output.flatten()
    rotated_flat = rotated.flatten()

    mean_output = torch.mean(output_flat)
    mean_rotated = torch.mean(rotated_flat)

    demeaned_output = output_flat-mean_output
    demeaned_rotated = rotated_flat-mean_rotated

    numerator = torch.sum(demeaned_output * demeaned_rotated)
    denominator = torch.sqrt(torch.sum(demeaned_output ** 2) * torch.sum(demeaned_rotated ** 2))

    r_corr_coef = numerator / denominator
    return r_corr_coef

def e_lee(model, imgs):
    """ Computes the Expected Equivariance Error (E[|Lf|^2]/d_out) w.r.t. rotation. """

    lie_derivative = rotation_lie_deriv_3d(model, imgs)

    # Compute squared norm of Lie derivative
    equivariance_error = lie_derivative.pow(2).mean()

    print(f'\nLie Derivative: {lie_derivative.mean().item():.6f}')
    print(f"Equivariance Error: {equivariance_error.item():.6f}\n")

    lie_derivative_np = lie_derivative.detach().cpu().numpy()
    with h5py.File("lie_derivative.h5", "w") as f:
        f.create_dataset("/data", data=lie_derivative_np)

    return equivariance_error

def resize_3d(data, target_shape=(67, 67, 67)):
    """ Resize a 3D volume to the target shape using cubic interpolation. """
    zoom_factors = [t / s for s, t in zip(data.shape, target_shape)]
    return zoom(data, zoom_factors, order=3)

def create_dataloaders(directory, batch_size, shuffle_status):
    present_dir = directory
    input_files = [f for f in os.listdir(present_dir) if f.endswith('_sad_file.h5')]
    pairedDataset = []

    for input_file in input_files:
        output_file= input_file.replace('_sad_file.h5','_rho_sad.h5')

        input_path = os.path.join(present_dir, input_file)
        output_path = os.path.join(present_dir, output_file)

        with h5py.File(input_path, 'r') as f:
            sad_data = f['data'][:]
            if sad_data.shape != (67,67,67):
                sad_data = resize_3d(sad_data, target_shape=(67,67,67))
            sad_data = np.expand_dims(sad_data, axis=0)
            sad_data = np.expand_dims(sad_data, axis=0)

        with h5py.File(output_path, 'r') as f:
            rho_sad_data = f['data'][:]
            if rho_sad_data.shape != (67,67,67):
                rho_sad_data = resize_3d(rho_sad_data, target_shape=(67,67,67))
            rho_sad_data = np.expand_dims(rho_sad_data, axis=0)
            rho_sad_data = np.expand_dims(rho_sad_data, axis=0)

        sad_tensor = torch.tensor(sad_data, dtype=torch.float32)
        rho_sad_tensor =  torch.tensor(rho_sad_data, dtype=torch.float32)

        pairedDataset.append(TensorDataset(sad_tensor, rho_sad_tensor))

    combined_dataset = ConcatDataset(pairedDataset)
    pairedDataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle_status)

    return pairedDataloader

def load_checkpoint_model(model, optimizer, ckpt_path):
    ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith('model_epoch_') and f.endswith('.pth')]

    if not ckpt_files:
        print('No checkpoints found!')
        return model, optimizer, 0
    
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[2].split('.')[0]))

    latest_ckpt = os.path.join(ckpt_path, ckpt_files[-1])

    ckpt = torch.load(latest_ckpt)
    #print('Checkpoint Keys:')
    #print(ckpt['model_state_dict'].keys())

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    #print('\nModel Keys:')
    #print(model.state_dict().keys())

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']
    print(f'Loaded checkpoint {latest_ckpt} from epoch {start_epoch}')
    return model, optimizer, start_epoch

def train_model(model, device, data_dir, num_epochs, optimizer, criterion, ckpt_path, loss_save_str,  start_epoch=0):
    ckpt_path = ckpt_path
    output_dir = './../outputs/'+loss_save_str
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    print(f'\nOutput directory created: {output_dir}')

    train_data_dir = data_dir + 'train/'
    train_dataloader = create_dataloaders(directory=train_data_dir, batch_size=1, shuffle_status=True)

    val_data_dir = data_dir + 'val/'
    val_dataloader = create_dataloaders(directory=val_data_dir, batch_size=1, shuffle_status=False)

    epoch_list = []
    loss_list = []
    val_loss_list = []
    epoch_time_list = []
    print('Training Initiated ...')
    train_tic  = time.time()

    for epoch in range(start_epoch, start_epoch+num_epochs):
        model.train()
        epoch_tic = time.time()
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss/len(train_dataloader)
        
        loss_list.append(epoch_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                running_val_loss += val_loss.item() * val_inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_dataloader)
        val_loss_list.append(epoch_val_loss)
        epoch_list.append(epoch+1)
        epoch_time = time.time() - epoch_tic
        epoch_time_list.append(epoch_time)

        table = PrettyTable()
        table.field_names = ['Epoch','Training Loss','Validation Loss','Epoch Time (s)']
        table.add_row([epoch+1, f'{epoch_loss:.6f}',f'{epoch_val_loss:.6f}',f'{epoch_time:.4f}'])
        print(table)

        if (epoch + 1)%2 == 0:
            ckpt = os.path.join(ckpt_path, f'model_epoch_{epoch+1}_train_loss_{epoch_loss}_val_loss_{epoch_val_loss}.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': epoch_val_loss,
            }, ckpt)
            print(f'Model for epoch {epoch+1} saved at {ckpt_path}')

    df = pd.DataFrame({'Epoch':epoch_list,
                       'Training Loss':loss_list,
                       'Validation Loss':val_loss_list,
                       })
    loss_filename = 'loss_log_'+loss_save_str+'.csv'
    csv_file_path = os.path.join(output_dir, loss_filename)
    df.to_csv(csv_file_path, index=False)

    print('Training Complete !!!')
    print(f'Training Time: {time.time() - train_tic} seconds')

def test_model(data_directory, device, model, output_save_str, criterion):
    save_outputs_at = './../outputs/' + output_save_str
    os.makedirs(save_outputs_at, exist_ok=True)

    test_dataloader = create_dataloaders(data_directory, batch_size=1, shuffle_status=False)
    
    test_loss = 0.0
    table = PrettyTable()
    table.field_names = ['File no.', 'Test Loss', 'Max. Equivariance Error', 'Prediction Error', 'Prediction NRMSE', 'Correlation Coefficient']
    
    test_time_tic = time.time()
    model.eval()
    print('Equivariance Test Initiated...')
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            rotated_input = torch.rot90(inputs, k=1, dims=(2,3))
            output_of_rotated_inputs = model(rotated_input)
            rotated_original_output = torch.rot90(outputs, k=1, dims=(2,3))

            corr_coeff = pearson_correlation_coefficient(output_of_rotated_inputs, rotated_original_output)

            equivariance_error = torch.abs(output_of_rotated_inputs - rotated_original_output)
            prediction_error = torch.abs(outputs - targets)
            
            norm_constant = abs(targets.max().item() - targets.min().item())
            nrmse = torch.sqrt(torch.mean(prediction_error ** 2))/norm_constant

            loss = criterion(outputs, targets)
            test_loss += loss.item()*inputs.size(0)


            table.add_row([idx+1, f'{test_loss:.10f}', f'{equivariance_error.max().item():.10f}', f'{prediction_error.max().item():.10f}', f'{nrmse:.10f}', f'{corr_coeff:.10f}'])

            output_save_path = os.path.join(save_outputs_at, f'original_output_{idx+1}.h5')
            with h5py.File(output_save_path, 'w') as f:
                f.create_dataset('data',data=outputs.detach().cpu().numpy())

            output_of_rotated_input_save_path = os.path.join(save_outputs_at, f'output_of_rotated_input_{idx+1}.h5')
            with h5py.File(output_of_rotated_input_save_path, 'w') as f:
                f.create_dataset('data',data=output_of_rotated_inputs.detach().cpu().numpy())

            rotated_orignal_outputs_save_path = os.path.join(save_outputs_at, f'rotated_original_outputs_{idx+1}.h5')
            with h5py.File(rotated_orignal_outputs_save_path, 'w') as f:
                f.create_dataset('data',data=rotated_original_output.detach().cpu().numpy())

    print(table)
    print(f'Equivariance Test Complete! Time required: {time.time()-test_time_tic} seconds')