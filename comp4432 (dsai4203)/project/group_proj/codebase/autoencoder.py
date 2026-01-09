import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.2):
        super(Autoencoder, self).__init__()

        encoder_layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(previous_dim, hidden_dim))
            encoder_layers.append(nn.SiLU(inplace=True))
            encoder_layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        encoder_layers.append(nn.Linear(previous_dim, latent_dim))
        encoder_layers.append(nn.SiLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        # linear projection
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)



class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Enc
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Dec
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.encoder_fc1(x))
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = F.relu(self.decoder_fc1(z))
        recon_x = torch.sigmoid(self.decoder_fc2(h))  # Assuming input is normalized [0,1]
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum') # Reconstruction loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL-dvergence
        return bce + kld
