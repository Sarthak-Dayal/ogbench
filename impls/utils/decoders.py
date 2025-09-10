import functools

import flax.linen as nn
import jax.image as jimage

from utils.networks import MLP

class ResBlock(nn.Module):
    features: int
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding='SAME', kernel_init=self.kernel_init)(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (3, 3), padding='SAME', kernel_init=self.kernel_init)(x)
        return x + residual

class ResnetUpStack(nn.Module):
    """IMPALA decoder stack: (optional) 2x upsample then ResBlocks."""
    num_features: int
    num_blocks: int
    upsample: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    up_method: str = "linear"   # "nearest" or "linear" (bilinear for 2D)

    @nn.compact
    def __call__(self, x):
        if self.upsample:
            b, h, w, c = x.shape
            x = jimage.resize(x, (b, h * 2, w * 2, c), method=self.up_method)
        # Entry conv mirrors encoder
        x = nn.Conv(self.num_features, (3, 3), padding='SAME', kernel_init=self.kernel_init)(x)
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_features, kernel_init=self.kernel_init)(x)
        return x


class ImpalaDecoder(nn.Module):
    """IMPALA-style decoder that mirrors ImpalaEncoder."""
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    layer_norm: bool = False

    # Output image spec
    out_channels: int = 3
    output_size: tuple = (84, 84)  # (H, W) of final image

    # Latent spec
    up_method: str = "linear"                # "nearest" or "linear"
    final_activation: str = None   # e.g., "sigmoid", "tanh", or None

    def setup(self):
        # Reverse the stacks for decoding: deepest -> shallowest
        rev = list(reversed(self.stack_sizes))
        self.up_stacks = [
            ResnetUpStack(
                num_features=s * self.width,
                num_blocks=self.num_blocks,
                upsample=True,                 # each stack upsamples by 2
                up_method=self.up_method,
            )
            for s in rev
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, z, *, train: bool = True):
        """z: [B, latent_dim] -> image [B, H, W, out_channels]"""
        H, W = self.output_size
        num_levels = len(self.stack_sizes)
        scale = 2 ** num_levels

        # Compute the starting spatial grid (must be integer)
        assert H % scale == 0 and W % scale == 0, \
            f"output_size {(H, W)} must be divisible by 2**len(stack_sizes)={scale}"
        h0, w0 = H // scale, W // scale

        # Start features taken to mirror the encoder's deepest stage
        start_features = self.stack_sizes[-1] * self.width

        # Project latent to (h0, w0, start_features)
        x = nn.Dense(h0 * w0 * start_features)(z)
        x = nn.relu(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = x.reshape((z.shape[0], h0, w0, start_features))

        # Go up through mirrored stacks
        for up in self.up_stacks:
            x = up(x)
            if self.dropout_rate is not None:
                x = self.dropout(x, deterministic=not train)

        x = nn.relu(x)

        # Final conv to desired channels
        x = nn.Conv(self.out_channels, (1, 1), padding='SAME')(x)

        # Optional output activation
        if self.final_activation == "sigmoid":
            x = nn.sigmoid(x)
        elif self.final_activation == "tanh":
            x = nn.tanh(x)
        return x

decoder_modules = {
    'impala': ImpalaDecoder,
    'impala_debug': functools.partial(ImpalaDecoder, num_blocks=1, stack_sizes=(4, 4), latent_dim=64),
    'impala_small': functools.partial(ImpalaDecoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaDecoder, stack_sizes=(64, 128, 128), latent_dim=1024),
}