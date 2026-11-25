"""A small causal Transformer whose weights are initialized from scratch."""
from dataclasses import asdict, dataclass
import mlx.core as mx
import mlx.nn as nn

from .resources import LabError, MAX_PARAMETERS


@dataclass
class ModelConfig:
    vocabulary: int
    dimensions: int = 128
    layers: int = 4
    heads: int = 4
    context: int = 128

    def validate(self):
        if not (1 <= self.layers <= 4 and self.dimensions in {64, 128, 192}
                and self.heads in {2, 4} and self.dimensions % self.heads == 0
                and 32 <= self.context <= 256 and 256 <= self.vocabulary <= 8192):
            raise LabError("Model configuration exceeds the bounded laptop pilot.")
        if self.parameter_estimate() > MAX_PARAMETERS:
            raise LabError("This pilot permits at most 5 million model parameters.")

    def parameter_estimate(self):
        d = self.dimensions
        return self.vocabulary * d + self.context * d + self.layers * (12 * d * d + 11 * d) + d


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dimensions
        self.heads = config.heads
        self.attention_norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.projection = nn.Linear(d, d, bias=False)
        self.feedforward_norm = nn.RMSNorm(d)
        self.expand = nn.Linear(d, 4 * d)
        self.contract = nn.Linear(4 * d, d)

    def __call__(self, x):
        batch, length, dimensions = x.shape
        q, k, v = mx.split(self.qkv(self.attention_norm(x)), 3, axis=-1)
        q, k, v = [part.reshape(batch, length, self.heads, -1).transpose(0, 2, 1, 3) for part in (q, k, v)]
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=(dimensions // self.heads) ** -0.5, mask="causal")
        attention = attention.transpose(0, 2, 1, 3).reshape(batch, length, dimensions)
        x = x + self.projection(attention)
        return x + self.contract(nn.gelu(self.expand(self.feedforward_norm(x))))


class IgboLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocabulary, config.dimensions)
        self.positions = nn.Embedding(config.context, config.dimensions)
        self.blocks = [CausalBlock(config) for _ in range(config.layers)]
        self.final_norm = nn.RMSNorm(config.dimensions)

    def __call__(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context:
            raise LabError("Input exceeds the model context.")
        x = self.embedding(tokens) + self.positions(mx.arange(tokens.shape[1]))
        for block in self.blocks:
            x = block(x)
        # Share the input embedding with the output head to save parameters.
        return self.final_norm(x) @ self.embedding.weight.T

    def configuration(self):
        return asdict(self.config)
