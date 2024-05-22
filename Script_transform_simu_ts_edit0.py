

%reset -f

import torch
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as sio
import os
import random
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_neuron = 6
N_trial = 70
N_t = 170
response_windows = [(70, 90), (80, 100), (90, 110), (100, 120), (110, 130), (120, 140)]

# Create the array with random data
spk_count = np.random.randint(0, 256, (N_neuron, N_trial, N_t), dtype=np.uint8)

# Add a unique response curve for each neuron using sin() and cos() functions with different phases
for i_n in range(N_neuron):
    response_start, response_end = response_windows[i_n]
    x = np.linspace(0, np.pi, response_end - response_start)
    if i_n % 2 == 0:
        response_curve = 127 + 127 * np.sin(x + i_n * np.pi / N_neuron)
    else:
        response_curve = 127 + 127 * np.cos(x + i_n * np.pi / N_neuron)
    response_curve = response_curve.astype(np.uint8)
    
    for i_tr in range(N_trial):
        # Adding stronger random noise
        noise = np.random.randint(0, 100, response_end - response_start)
        noisy_response_curve = response_curve + noise
        noisy_response_curve = np.clip(noisy_response_curve, 0, 255)  # Ensure values are within uint8 range   
        spk_count[i_n, i_tr, response_start:response_end] = noisy_response_curve[:]


#%%
plot_data = 0
if plot_data == 1:
    # Plotting the response curves for all neurons in a single trial
    fig, axs = plt.subplots(N_neuron, 1)
    for i_n in range(N_neuron):
        axs[i_n].plot(spk_count[i_n, 0, :], color='black',linewidth=0.5)
        axs[i_n].set_ylabel(f'neuron {i_n + 1} \n activity')
        if i_n == (N_neuron-1):
            axs[i_n].set_xlabel('time unit (arb.)')
    plt.tight_layout()
    plt.show()
    
    # Plotting all trials of one neuron as a 2D image
    neuron_to_plot = 0
    plt.figure()
    plt.imshow(spk_count[neuron_to_plot], aspect='auto', cmap='jet')
    plt.colorbar(label='neural activity')
    plt.title(f'all trials of neuron {neuron_to_plot + 1}')
    plt.xlabel('time unit (arb.)')
    plt.ylabel('trial')
    plt.show()

#%%

N_test = 10
N_train = N_trial-N_test
random.seed(0)
ind_test = random.sample(range(N_trial), N_test)
spk_test = np.zeros((N_neuron,N_test,N_t))
spk_train = np.zeros((N_neuron,N_train,N_t))
count_test = 0
count_train = 0
for i_tr in range(N_trial):
    # i_tr = ind_test[0]
    if i_tr in ind_test:
        spk_test[:,count_test,:] = spk_count[:,i_tr,:]
        count_test += 1
    else:
        spk_train[:,count_train,:] = spk_count[:,i_tr,:]
        count_train += 1
    
do_scaler = 1
if do_scaler == 1: 
    # try this after; not sure if I should scale if all data is same magnitude
    # if I split then scale the prediction may go down
    
    # obs is a 3D array with shape (num_features, num_trials, num_time_steps)
    def scale_3d_array_per_feature(obs):
        num_features = obs.shape[0]
        num_trials = obs.shape[1]
        num_time_steps = obs.shape[2]
    
        obs_scaled = np.empty_like(obs, dtype=np.float32)
        
        for i in range(num_features):
            # Reshape the feature array to 2D (num_trials * num_time_steps)
            feature_data = obs[i].reshape(-1, 1)
            
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            scaled_feature_data = scaler.fit_transform(feature_data)
            
            # scaled_feature_data = feature_data
            # scaled_feature_data[feature_data > 0] = 1
            
            # Reshape back to the original shape and store in obs_scaled
            obs_scaled[i] = scaled_feature_data.reshape(num_trials, num_time_steps)
        
        return obs_scaled
    
    spk_test = scale_3d_array_per_feature(spk_test)
    spk_train = scale_3d_array_per_feature(spk_train)

# Sequence Data Preparation
SEQUENCE_SIZE = 10

def to_sequences_neur_trial_time(seq_size, obs):
    num_features = obs.shape[0]
    num_trials = obs.shape[1]
    num_time_steps = obs.shape[2]

    # Initialize empty arrays for x and y
    x = np.empty((0, seq_size, num_features), dtype=np.float32)
    y = np.empty((0, num_features), dtype=np.float32)

    for trial in range(num_trials):
        trial_x = np.empty((0, seq_size, num_features), dtype=np.float32)
        trial_y = np.empty((0, num_features), dtype=np.float32)
        for i in range(num_time_steps - seq_size):
            window = np.expand_dims(obs[:, trial, i:(i + seq_size)].T, axis=0)  # Shape (1, seq_size, num_features)
            after_window = np.expand_dims(obs[:, trial, i + seq_size], axis=0)  # Shape (1, num_features)
            trial_x = np.concatenate((trial_x, window), axis=0)
            trial_y = np.concatenate((trial_y, after_window), axis=0)
        x = np.concatenate((x, trial_x), axis=0)
        y = np.concatenate((y, trial_y), axis=0)

    # Convert numpy arrays to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return x_tensor, y_tensor


# obs = spk_train
# seq_size = SEQUENCE_SIZE
# i = 0
# trial = 0
x_spk_train, y_spk_train = to_sequences_neur_trial_time(SEQUENCE_SIZE, spk_train)
x_spk_test,  y_spk_test  = to_sequences_neur_trial_time(SEQUENCE_SIZE, spk_test)
# aaa = x_spk_train[:,:,1].numpy()


# Setup data loaders for batch
train_spk_dataset = TensorDataset(x_spk_train, y_spk_train)
train_spk_loader = DataLoader(train_spk_dataset, batch_size=32, shuffle=True)

test_spk_dataset = TensorDataset(x_spk_test, y_spk_test)
test_spk_loader = DataLoader(test_spk_dataset, batch_size=32, shuffle=False)



# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
# Model definition using Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, input_dim) #1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :])
        return x

# model = TransformerModel().to(device)
model = TransformerModel(input_dim = N_neuron).to(device)


# Train the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 1000
early_stop_count = 0
min_val_loss = float('inf')
early_stop = 0 #1
for epoch in range(epochs):
    # epoch = 0
        
    for batch in train_spk_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_spk_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())
    # aaa = outputs.cpu().numpy()
    # bbb = y_batch.cpu().numpy()
    
    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)
    if early_stop == 1:
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
    
        if early_stop_count >= 5:
            print("Early stopping!")
            break
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
    
# Evaluation
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_spk_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        predictions.extend(outputs.squeeze().tolist())

rmse = np.sqrt(np.mean((np.array(predictions).reshape(-1, 1) - y_spk_test.numpy().reshape(-1, 1))**2))
aaa = np.array(predictions) #.reshape(-1, 1) 
bbb = y_spk_test.numpy() #.reshape(-1, 1)
print(f"Score (RMSE): {rmse:.4f}")



#%%

fig, axs = plt.subplots(6, sharex=True, sharey=True)

# Define colors and labels
colors = ['black', 'red']
labels = ['neural activity', 'predicted']

# Plot data for each subplot
for i in range(6):
    axs[i].plot(bbb[:, i], color=colors[0], linewidth=0.5)
    axs[i].plot(aaa[:, i], color=colors[1], linewidth=0.5)
    axs[i].autoscale(tight=True)
    axs[i].set_ylabel(f'neuron {i + 1} \n activity')
    if i == 5:
        axs[i].plot([], [], color=colors[0], label=labels[0])  # Empty plot for legend
        axs[i].plot([], [], color=colors[1], label=labels[1])  # Empty plot for legend
        axs[i].legend()
        axs[i].set_xlabel('time unit (arb.)')

#%%

N_t_pred = (N_t - SEQUENCE_SIZE)
aaa_rs = np.empty((N_neuron, N_test, N_t_pred), dtype=np.float32)
bbb_rs = np.empty((N_neuron, N_test, N_t_pred), dtype=np.float32)

for i_tr in range(N_test):
    aaa_rs[:,i_tr,:] = aaa[(i_tr*N_t_pred):((i_tr+1)*N_t_pred),:].T
    bbb_rs[:,i_tr,:] = bbb[(i_tr*N_t_pred):((i_tr+1)*N_t_pred),:].T

ccc_rs = abs(aaa_rs-bbb_rs)
ccc_mn = np.mean(ccc_rs,1)
aaa_mn = np.mean(aaa_rs,1)
bbb_mn = np.mean(bbb_rs,1)

t_ms = np.arange(-8400,8600-100*SEQUENCE_SIZE,100)
fig, axs = plt.subplots(6, sharex=True, sharey=True)
for i_n in range(N_neuron):
    axs[i_n].plot(t_ms,bbb_mn[i_n,:], color='black',linewidth=0.5)
    axs[i_n].plot(t_ms,aaa_mn[i_n,:], color='red',linewidth=0.5)
    axs[i_n].autoscale(tight=True)
    axs[i_n].set_ylabel(f'neuron {i_n + 1} \n activity')
axs[i_n].set_xlabel('time from reach (ms)')


fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
axs[0].imshow(aaa_rs[0,:,:], aspect='auto')
axs[1].imshow(bbb_rs[0,:,:], aspect='auto')
# plt.imshow(ccc_rs[0,:,:], aspect='auto')

