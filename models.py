import torch
import torch.nn as nn
from torch.distributions import Normal


# xavier_uniform (glorot uniform)
def init_xavier_uniform(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


##################
### Components ###
##################


class GaussianMLP(nn.Module):
    def __init__(self, input_size, output_size, negative_slope=0.2):
        super().__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, output_size)
        self.std_layer = nn.Linear(256, output_size)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x = self.leaky_relu1(self.l1(x))
        x = self.leaky_relu2(self.l2(x))
        mean = self.mean_layer(x)
        std = self.softplus(self.std_layer(x)) + 1e-5
        return Normal(mean, std)


class Encoder(nn.Module):
    def __init__(self, feature_size=256, negative_slope=0.2):
        super().__init__()

        self.net = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.LeakyReLU(negative_slope),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(negative_slope),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(negative_slope),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(negative_slope),
            # (256, 4, 4) -> (latent_size, 1, 1)
            nn.Conv2d(256, feature_size, 4),
            nn.LeakyReLU(negative_slope)
        ).apply(init_xavier_uniform)

    def forward(self, x):
        batch_size, sequence_size, C, H, W = x.size()
        x = x.view(batch_size * sequence_size, C, H, W)
        x = self.net(x)
        x = x.view(batch_size, sequence_size, -1)

        return x


class Decoder(nn.Module):
    def __init__(self, input_size=288, std=1.0, negative_slope=0.2):
        super(Decoder, self).__init__()
        self.std_scale = std_scale

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_size, 256, 4),
            nn.LeakyReLU(negative_slope),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU(negative_slope)
        ).apply(init_xavier_uniform)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = torch.cat(x,  dim=-1)

        batch_size, sequence_size, latent_size = x.size()
        x = x.view(batch_size * sequence_size, latent_size, 1, 1)
        mean = self.net(x)
        _, C, W, H = mean.size()
        mean = mean.view(batch_size, sequence_size, C, W, H)
        std = torch.ones_like(x) * self.std
        return Normal(mean, std)


##############
### Models ###
##############

class LatentStateModel(nn.Module):
    def __init__(self, action_size, feature_size=256, latent1_size=32, latent2_size=256):
        super().__init__()
        self.feature_size = feature_size
        self.action_size = action_size
        self.latent1_size = latent1_size
        self.latent2_size = latent2_size

        # encode image to feature vector
        self.encoder = Encoder(feature_size)
        # decode state to image
        self.decoder = Decoder(latent1_size+latent2_size)

        ### priors ###

        # p(z_1^1)
        self.latent1_first_prior = Normal(torch.zeros(latent1_size), torch.ones(latent1_size))
        # p(z_1^2 | z_1^1)
        self.latent2_first_prior = GaussianMLP(latent1_size, latent2_size)
        # p(z_{t+1}^1 | z_t^2, a_t)
        self.latent1_prior = GaussianMLP(latent2_size+action_size, latent1_size)
        # p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_prior = GaussianMLP(latent1_size+latent2_size+action_size, latent2_size)

        ### posteriors ###

        # q(z_1^1 | x_1)
        self.latent1_first_posterior = GaussianMLP(feature_size, latent1_size)
        # q(z_1^2 | z_1^1) = p(z_1^2 | z_1^1)
        self.latent2_first_posterior = self.latent2_first_prior
        # q(z_{t+1}^1 | x_{t+1}, z_t^2, a_t)
        self.latent1_posterior = GaussianMLP(feature_size+latent2_size+action_size, latent1_size)
        # q(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t) = p(z_{t+1}^2 | z_{t+1}^1, z_t^2, a_t)
        self.latent2_posterior = GaussianMLP(latent1_size+latent2_size+action_size, latent2_size)
    
    @property
    def state_size(self):
        return self.latent1_size + self.latent2_size
    
    def sample_prior_or_posterior(self, actions, images=None):
        actions = actions.transpose(1, 0, 2)
        sequence_size, batch_size, _ = actions.size()
        
        if images in not None:
            features = self.encoder(images)

        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []
        for t in range(sequence_size + 1):
            is_conditional = images is not None and (t < images.size()[0])

            if t == 0:
                if is_conditional:
                    latent1_dist = self.latent1_init_posterior(features[t])
                    latent1_sample = latent1_dist.rsample()
                    latent2_dist = self.latent2_first_posterior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()
                else:
                    latent1_dist = self.latent1_first_prior()
                    latent1_sample = latent1_dist.rsample()
                    latent2_dist = self.latent2_first_prior(latent1_sample)
                    latent2_sample = latent2_dist.rsample()
            else:
                if is_conditional:
                    latent1_first_dist = self.latent1_first_posterior(features[t])
                    latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
                    latent1_sample = latent1_dist.sample()
                else:
                    latent1_first_dist = self.latent1_first_prior()
                    latent1_dist = self.latent1_prior(latent2_samples[t-1], actions[t-1])
                    latent1_sample = latent1_dist.rsample()
            
            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)
        
        latent1_samples = torch.stack(latent1_samples, dim=1)
        latent2_samples = torch.stack(latent2_samples, dim=1)

        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)

    def sample_posterior(self, actions, features=None):
        actions = actions.transpose(1, 0, 2)
        sequence_size, batch_size, _ = actions.size()

        if features is None:
            features = self.encoder(images)
            featuress = features.transpose(1, 0, 2)
        
        latent1_dists = []
        latent1_samples = []
        latent2_dists = []
        latent2_samples = []
        for t in range(sequence_length + 1):
            if t == 0:
                latent1_dist = self.latent1_first_posterior(features[t])
                latent1_sample = latent1_dist.rsample()
                latent2_dist = self.latent2_first_posterior(latent1_sample)
                latent2_sample = latent2_dist.rsample()
            else:
                latent1_first_dist = self.latent1_first_posterior(features[t])
                latent1_dist = self.latent1_posterior(features[t], latent2_samples[t-1], actions[t-1])
                latent1_sample = latent1_dist.rsample()

                latent2_first_dist = self.latent2_first_posterior(latent1_sample)
                latent2_dist = self.latent2_posterior(latent1_sample, latent2_samples[t-1], actions[t-1])
                latent2_sample = latent2_dist.rsample()

            latent1_dists.append(latent1_dist)
            latent1_samples.append(latent1_sample)
            latent2_dists.append(latent2_dist)
            latent2_samples.append(latent2_sample)

        latent1_samples = torch.stack(latent1_samples, axis=1)
        latent2_samples = torch.stack(latent2_samples, axis=1)

        return (latent1_samples, latent2_samples), (latent1_dists, latent2_dists)


class Actor(nn.Module):
    def __init__(self, action_size, latent_state_size=288, h_size=256):
        super().__init__()
        self.l1 = nn.Linear(latent_state_size, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.mean_layer = nn.Linear(h_size, action_size)
        self.std_layer = nn.Linear(h_size, action_size)
        self.softplus = nn.Softplus()
        self.apply(init_xavier_uniform)
    
    def forward(self, latent_state):
        x = torch.relu(self.l1(latent_state))
        x = torch.relu(self.l2(x))
        mean = self.mean_layer(x)
        std = self.std_layer(x).clamp(-20, 2)
        std = torch.exp(std)
        return mean, std
    
    def get_action(self, latent_state):
        with torch.no_grad():
            mean, std = self.forward(latent_state)
            action = Normal(mean, std).sample().flatten().numpy()
        return action


class Critic(nn.Module):
    def __init__(self, action_size, latent_state_size=288, h_size=256):
        super().__init__()
        self.l1 = nn.Linear(latent_state_size + action_size, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, 1)
        self.apply(init_xavier_uniform)
    
    def forward(latent_state, action):
        x = torch.cat([latent_state, action], dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)

