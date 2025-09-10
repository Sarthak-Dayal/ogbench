from .acro import ACROTrainer
from .vae import VAETrainer

encoders = dict(
    acro=ACROTrainer,
    vae=VAETrainer
)