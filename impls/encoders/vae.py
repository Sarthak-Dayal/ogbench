from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import distrax

from utils.decoders import decoder_modules
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP

class VAETrainer(flax.struct.PyTreeNode):
    """VAE Train state."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()


    @jax.jit
    def loss(self, batch, grad_params, rng=None):
        """Compute the VAE Loss."""
        assert len(batch['observations'].shape) in [2, 3, 4]

        obs = batch['observations']

        z_mean = self.network.select('encoder')(obs, params=grad_params)
        z_logstd = self.network.select('encoder_std')(obs, params=grad_params)
        dist = distrax.MultivariateNormalDiag(loc=z_mean, scale_diag=jnp.exp(z_logstd))

        rng = rng if rng is not None else self.rng
        rng, sample_key = jax.random.split(rng)
        z = dist.sample(seed=sample_key)

        # Information Bottleneck loss
        if self.config['ib_loss_type'] == "l1":
            ib_loss = jnp.abs(z).sum(axis=-1).mean() + jnp.abs(z).sum(axis=-1).mean()
        elif self.config['ib_loss_type'] == "l2":
            ib_loss = (z ** 2).sum(axis=-1).mean() + (z ** 2).sum(axis=-1).mean()
        elif self.config['ib_loss_type'] == "None":
            ib_loss = 0.0
        else:
            raise NotImplementedError(f"ib_loss_type not implemented: {self.config['ib_loss_type']}.")

        recon = self.network.select('decoder')(z, params=grad_params)

        recon_reshaped = jnp.reshape(recon, (recon.shape[0], -1))
        obs_reshaped = jnp.reshape(obs, (obs.shape[0], -1))
        recon_loss = ((recon_reshaped - obs_reshaped) ** 2).sum(axis=-1).mean(axis=-1)

        pz = distrax.MultivariateNormalDiag(
            loc=jnp.zeros_like(z_mean),
            scale_diag=jnp.ones_like(z_logstd)
        )
        kl_penalty = dist.kl_divergence(pz).mean(axis=0)

        loss = recon_loss + self.config['ib_loss_coeff'] * ib_loss + self.config['kl_penalty_coeff'] * kl_penalty
        return loss, {
            'recon_loss': recon_loss,
            'ib_loss': ib_loss,
            'kl_penalty': kl_penalty,
        }
    
    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info # yay!
    
    @jax.jit
    def encode(self, observations):
        """Encode observations into representations."""
        return self.network.select['encoder'](observations, params=self.network.params) # thank you copilot :)


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create VAE trainer instance. Train VAE using action prediction loss and encoder to file."""

        if config is None:
            config = get_config()


        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            vae_enc_mean_seq = [encoder_module()]
            vae_enc_std_seq = [encoder_module()]
        else:
            vae_enc_mean_seq = []
            vae_enc_std_seq = []

        vae_enc_mean_seq.append(
            MLP(
                hidden_dims=(*config['encoder_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )

        vae_enc_std_seq.append(
            MLP(
                hidden_dims=(*config['encoder_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        
        vae_enc_mean_def = nn.Sequential(vae_enc_mean_seq)
        vae_enc_std_def = nn.Sequential(vae_enc_std_seq)

        mlp_out_dim = 512 if config['encoder'] else ex_observations.shape[-1]
        vae_dec_seq = [MLP(
            hidden_dims=(*reversed(config['encoder_hidden_dims']), mlp_out_dim),
            activate_final=False,
            layer_norm=config['layer_norm'],
        )]

        if config['encoder'] is not None:
            decoder_module = decoder_modules[config['encoder']]
            vae_dec_seq.append(decoder_module())

        vae_dec_def = nn.Sequential(vae_dec_seq)

        ex_latent = jnp.zeros((ex_observations.shape[0], config['rep_dim']))
        network_info = dict(
            encoder=(vae_enc_mean_def, (ex_observations,)), # All Encoders must contain some encoder :4)
            encoder_std=(vae_enc_std_def, (ex_observations,)),
            decoder=(vae_dec_def, (ex_latent,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))



def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            encoder_name='vae',  # Encoder name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            encoder_hidden_dims=(256, 256),  # Hidden dimensions for the encoder MLP.
            layer_norm=True,  # Whether to use layer normalization.
            rep_dim=10,  # VAE representation dimension.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            ib_loss_type='None',  # Type of information bottleneck loss ('l1' or 'l2' or none).
            ib_loss_coeff=1e-3,  # Coefficient for the information bottleneck loss.
            kl_penalty_coeff=1.0,  # Coefficient for the KL divergence penalty.
            # Dataset hyperparameters.
            dataset_class='VAEDataset',  # Dataset class name.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            # Miscellaneous.
            discrete=False,
        )
    )
    return config