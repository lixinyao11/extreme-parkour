import h5py
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import wandb  # Add this import
import argparse  # Add this import

class DepthLatentDataset(Dataset):
    def __init__(self, hdf5_files, past_steps=5, future_steps=25, future_stride=0, device="cpu"):
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.future_stride = future_stride
        self.device = device
        
        self.depth_latent = []
        self.rewards = []
        for hdf5_file in hdf5_files:
            with h5py.File(hdf5_file, 'r') as f:
                self.depth_latent.append(f['depth_latent'][:])
                self.rewards.append(f['rew'][:])
        self.depth_latent = np.concatenate(self.depth_latent, axis=0)  # (1500 * 2, 1024=num_envs, 32)
        self.rewards = np.concatenate(self.rewards, axis=0)  # (1500 * 2, 1024=num_envs)
        
        # Normalize rewards
        self.rewards_mean = np.mean(self.rewards)
        self.rewards_std = np.std(self.rewards)
        self.rewards = (self.rewards - self.rewards_mean) / self.rewards_std
        np.save('predictor/statistics.npy', {'mean': self.rewards_mean, 'std': self.rewards_std})

        self.draw_data()

    def draw_data(self):
        if os.path.exists(f"predictor/latent_vs_rewards_{self.past_steps}_{self.future_stride}_{self.future_steps}.png"):
            return
        past_latents = []
        future_rewards = []
        for idx in range(len(self.depth_latent) - self.past_steps - self.future_steps - self.future_stride + 1):
            past_latents.append(self.depth_latent[idx:idx+self.past_steps].transpose(1, 0, 2)) # (num_env, past_steps, 32)
            rew = self.rewards[idx+self.past_steps+self.future_stride:idx+self.past_steps+self.future_stride+self.future_steps].transpose(1, 0)  # (num_env, future_steps)
            # rew = np.mean(rew, axis=1)  # (num_env, 1)
            future_rewards.append(rew)

        past_latents = np.array(past_latents)  # Shape: (samples, num_envs, past_steps, latent_dim)
        future_rewards = np.array(future_rewards)  # Shape: (samples, num_envs)

        # Reshape past_latents to 2D array for PCA: (samples * num_envs, past_steps * latent_dim)
        samples, num_envs, past_steps, latent_dim = past_latents.shape
        past_latents_reshaped = past_latents.reshape(samples * num_envs, past_steps * latent_dim)
        # Reshape future_rewards to 2D array: (samples * num_envs, future_steps)
        future_rewards_reshaped = future_rewards.reshape(samples * num_envs, self.future_steps)

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(past_latents_reshaped)

        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=future_rewards_reshaped.mean(axis=1), cmap='viridis')
        plt.colorbar()
        plt.title("Latent Space vs Future Rewards")
        plt.savefig(f"predictor/latent_vs_rewards_{self.past_steps}_{self.future_stride}_{self.future_steps}.png")
    
    def __len__(self):
        # 0 ~ len - past_steps - future_steps - future_stride
        return len(self.depth_latent) - self.past_steps - self.future_steps - self.future_stride + 1
    
    def __getitem__(self, idx):
        past_latents = self.depth_latent[idx:idx+self.past_steps].transpose(1, 0, 2)  # (num_env, past_steps, 32)
        future_rewards = self.rewards[idx+self.past_steps+self.future_stride:idx+self.past_steps+self.future_stride+self.future_steps].transpose(1, 0)  # (num_env, future_steps)
        future_rewards = np.mean(future_rewards, axis=1, keepdims=True)  # (num_env, 1)
        return torch.tensor(past_latents, dtype=torch.float32, device=self.device), torch.tensor(future_rewards, dtype=torch.float32, device=self.device)

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim, past_steps, future_steps):
        super(RewardPredictor, self).__init__()
        self.lstm = nn.LSTM(latent_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, future_steps)
    
    def forward(self, x):
        batch_size, num_envs, past_steps, latent_dim = x.size()
        x = x.view(batch_size * num_envs, past_steps, latent_dim)  # Merge batch and num_envs
        _, (hn, _) = self.lstm(x)
        hn = hn[-1]
        out = self.fc(hn)
        return out.view(batch_size, num_envs, -1)  # Reshape back to (batch_size, num_envs, future_steps)

class MLPRewardPredictor(nn.Module):
    def __init__(self, latent_dim, past_steps):
        super(MLPRewardPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim * past_steps, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        B, num_envs, past_steps, latent_dim = x.size()
        x = x.view(B * num_envs, -1)  # Merge batch and num_envs and flatten the input
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        return out.view(B, num_envs, -1)  # Reshape back to (batch_size, num_envs, 1)

def train_model(hdf5_files, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Automatically detect GPU
    dataset = DepthLatentDataset(hdf5_files, past_steps=args.past_len, future_steps=args.future_len, future_stride=args.future_stride, device=device)
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.model_type == 'mlp':
        model = MLPRewardPredictor(latent_dim=dataset.depth_latent.shape[-1], past_steps=args.past_len)
    else:
        model = RewardPredictor(latent_dim=dataset.depth_latent.shape[-1], past_steps=args.past_len, future_steps=args.future_len)
    
    model = model.to(device)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    wandb.init(
        project="reward_predictor",
        name=f"{args.exptid}_{args.past_len}_{args.future_stride}_{args.future_len}",
        entity="xinyaoli-sjtu-icec",
        dir="preditor/logs"
    )
    
    for epoch in range(args.num_epoch):
        model.train()
        epoch_loss = 0
        for past_latents, future_rewards in train_loader:
            optimizer.zero_grad()
            predictions = model(past_latents)
            loss = criterion(predictions, future_rewards)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"Train Loss": avg_epoch_loss})  # Log training loss to wandb
        print(f'Epoch {epoch+1}/{args.num_epoch}, Loss: {avg_epoch_loss}')
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for past_latents, future_rewards in test_loader:
                    predictions = model(past_latents)
                    loss = criterion(predictions, future_rewards)
                    test_loss += loss.item()
            avg_test_loss = test_loss / len(test_loader)
            wandb.log({"Test Loss": avg_test_loss})  # Log test loss to wandb
            print(f'Test Loss after Epoch {epoch+1}: {avg_test_loss}')
    
    torch.save(model.state_dict(), f"predictor/ckpts/predictor_{args.exptid}_{args.past_len}_{args.future_stride}_{args.future_len}.pth")
    wandb.finish()  # Finish the wandb run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Reward Predictor')
    parser.add_argument('--exptid', type=str, required=True, help='Experiment ID for wandb logging')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'lstm'], help='Type of model to use')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of data to use for testing')
    parser.add_argument('--past_len', type=int, default=5, help='Number of past steps to use')
    parser.add_argument('--future_len', type=int, default=25, help='Number of future steps to predict')
    parser.add_argument('--future_stride', type=int, default=0, help='Stride for future steps')
    args = parser.parse_args()

    hdf5_files = [
        '/home/xyli/Code/extreme-parkour/legged_gym/logs/parkour_new/dis_eval_dataset_1.5_2.5.hdf5', 
        '/home/xyli/Code/extreme-parkour/legged_gym/logs/parkour_new/dis_eval_dataset_0.5_1.5.hdf5'
    ]
    train_model(hdf5_files, args)
