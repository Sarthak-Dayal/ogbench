from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, GCValue, Identity, LengthNormalize

class ACROTrainer(flax.struct.PyTreeNode):
    """ACRO Train state."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()


    @jax.jit
    def loss(self, batch, grad_params, rng=None):
        """Compute the total loss, combining regularization on the encoder and action prediction loss."""
        assert len(batch['observations_t'].shape) in [2, 3, 4]
        assert len(batch['observations_t_k'].shape) in [2, 3, 4]
        assert len(batch['actions_t'].shape) == 2
        info = {}
        
        obs_t = batch['observations_t']
        obs_t_k = batch['observations_t_k']
        actions_t = batch['actions_t']

        z_t = self.network.select('encoder')(obs_t, params=grad_params)
        z_t_k = self.network.select('encoder')(obs_t_k, params=grad_params)

        # Information Bottleneck loss
        if self.config['ib_loss_type'] == "l1":
            ib_loss = jnp.abs(z_t).sum(axis=-1).mean() + jnp.abs(z_t_k).sum(axis=-1).mean()
        elif self.config['ib_loss_type'] == "l2":
            ib_loss = (z_t ** 2).sum(axis=-1).mean() + (z_t_k ** 2).sum(axis=-1).mean()
        elif self.config['ib_loss_type'] == "None":
            ib_loss = 0.0
        else:
            raise NotImplementedError(f"ib_loss_type not implemented: {self.config['ib_loss_type']}.")

        action_preds = self.network.select('acro_pred')(jnp.concatenate([z_t, z_t_k], axis=-1), params=grad_params)
        
        if self.config['discrete']:
            action_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
                action_preds, actions_t.squeeze(-1)
            ).mean()
        else:
            action_loss = ((action_preds - actions_t) ** 2).mean()

        loss = action_loss + self.config['ib_loss_coeff'] * ib_loss
        return loss, {
            'action_loss': action_loss,
            'ib_loss': ib_loss,
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
        """Create ACRO trainer instance. Train ACRO using action prediction loss and encoder to file."""

        if config is None:
            config = get_config()

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            acro_enc_seq = [encoder_module()]
        else:
            acro_enc_seq = []

        acro_enc_seq.append(
            MLP(
                hidden_dims=(*config['encoder_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        
        acro_enc_seq.append(LengthNormalize())
        acro_enc_def = nn.Sequential(acro_enc_seq)

        acro_pred_def = MLP(
            hidden_dims=(*config['pred_hidden_dims'], action_dim),
            activate_final=False,
            layer_norm=config['layer_norm'],
        )

        ex_acro_pred_input = jnp.zeros((1, 2 * config['rep_dim']))
        network_info = dict(
            encoder=(acro_enc_def, (ex_observations,)), # All Encoders must contain some encoder :4)
            acro_pred=(acro_pred_def, (ex_acro_pred_input,))
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
            encoder_name='acro',  # Encoder name.
            lr=3e-4,  # Learning rate.
            batch_size=10240,  # Batch size.
            encoder_hidden_dims=(256, 256),  # Hidden dimensions for the encoder MLP.
            pred_hidden_dims=(256, 256),  # Hidden dimensions for the action predictor MLP.
            layer_norm=True,  # Whether to use layer normalization.
            acro_k_step=25,  # acro k step.
            rep_dim=10,  # ACRO representation dimension.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            ib_loss_type='None',  # Type of information bottleneck loss ('l1' or 'l2' or none).
            ib_loss_coeff=1e-3,  # Coefficient for the information bottleneck loss.
            # Dataset hyperparameters.
            dataset_class='ACRODataset',  # Dataset class name.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config