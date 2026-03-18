import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def cos_sin_transformation(
    position: torch.Tensor, max_power: int = 4, include_raw_coordinates=False
) -> torch.Tensor:
    if position.ndim == 1:
        position = position.unsqueeze(0)  # [1, 4]

    powers = 2.0 ** torch.arange(
        max_power + 1, device=position.device, dtype=position.dtype
    )  # [5]
    scaled_pos = position.unsqueeze(-1) * powers  # [B, 4, 5]
    scaled_pos = scaled_pos.view(position.size(0), -1)  # [B, 20]

    cos_vec = torch.cos(scaled_pos)
    sin_vec = torch.sin(scaled_pos)

    concat = torch.cat([cos_vec, sin_vec], dim=-1)

    if include_raw_coordinates:
        concat = torch.cat([concat, position])

    return concat  # [B, 40]


class TransformerHead(nn.Module):
    """Transformer-based classification head with coordinate-to-token projection.

    This module converts physical position and radial distance into a set of
    learned positional tokens. These tokens are prepended to the input sequence
    (alongside the CLS token) and processed through a Transformer Encoder block.

    Args:
        d_output (int): Dimension of the final output (e.g., number of classes).
        input_token_dim (int): Embedding dimension (D). Defaults to 512.
        p_drop (float): Dropout probability. Defaults to 0.1.
        max_position_element (int): Highest power used in sine/cosine
            transform. Defaults to 4.
        num_pos_token (int): Number of positional tokens to inject. Defaults to 10.
        nhead (int): Number of attention heads. Defaults to 8.

    Attributes:
        num_pos_token (int): The number of latent tokens generated from
            the positional encoding.
        input_token_dim (int): The dimensionality of the latent tokens (D).
        pos_encoder (callable): A function applying sine/cosine transformations
            to raw coordinates.
        projection (nn.Linear): Linear layer mapping encoded positions to
            the token space.
        transformer_block (nn.TransformerEncoderLayer): A single transformer
            layer for cross-token communication.
        classifier (nn.Linear): Final mapping to output dimension.
    """

    def __init__(
        self,
        d_output,
        input_token_dim=512,
        p_drop=0.1,
        max_position_element=4,
        num_pos_token=10,
        nhead=8,
    ):
        super().__init__()
        self.num_pos_token = num_pos_token
        self.input_token_dim = input_token_dim

        self.pos_encoder = lambda pos: cos_sin_transformation(
            pos, max_power=max_position_element
        )

        self.projection = nn.Linear(
            2 * (max_position_element + 1) * 4 + 1, num_pos_token * input_token_dim
        )
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=input_token_dim,
            nhead=nhead,
        )
        self.dropout = nn.Dropout(p_drop)
        self.classifier = nn.Linear(input_token_dim, d_output)

    def forward(self, x, position, r_distance):  # x: [B, 256, 512]

        pos_enc = self.pos_encoder(position)
        pos_enc = torch.cat([pos_enc, r_distance.view(-1, 1)], dim=1)
        pos_token = self.projection(pos_enc)
        pos_token = pos_token.view(
            x.shape[0], self.num_pos_token, self.input_token_dim
        )  # [B, self.num_pos_token, 512]

        combined_emb = torch.cat(
            [
                x[:, :1, :],  # CLS token
                pos_token,  # positional tokens
                x[:, 1:, :],  # Spatial patches
            ],
            dim=1,
        )  # [batch, # token, 512]

        embed = self.transformer_block(combined_emb)

        cls_token = embed[:, 0, :]  # [B, 512]
        return self.classifier(self.dropout(cls_token))


# class Transformer


class SimpleLinear(nn.Module):
    """Multi-Layer Perceptron head for flattened feature processing.

    This head flattens the input feature map and concatenates it with harmonic
    positional encodings (sine/cosine) and radial distance before passing
    the combined vector through a non-linear MLP.

    Args:
        d_output (int): Dimension of the final output.
        input_feature_dim (int): Dimension of the input features before flattening.
        max_position_element (int): Highest power used in sine/cosine
            transform. Defaults to 4.
        position_size (int): Number of raw coordinate variables. Defaults to 4.
        hidden_dim (int): Width of the first hidden layer. Defaults to 16.
        p_drop (float): Dropout probability. Defaults to 0.1.

    Attributes:
        d_output (int): Output dimensionality.
        input_feature_dim (int): Dimension of the input features.
        hidden_dim (int): Hidden layer width.
        p_drop (float): Dropout rate used in the network.
        max_position_element (int): Complexity of the harmonic encoding.
        position_size (int): Number of coordinate inputs.
        network (nn.Sequential): The MLP architecture (Linear -> ReLU -> Dropout).
    """

    def __init__(
        self,
        d_output,
        input_feature_dim,
        max_position_element=4,
        position_size=4,
        hidden_dim=16,
        p_drop=0.1,
    ):
        super().__init__()
        self.d_output = d_output
        self.input_feature_dim = input_feature_dim
        self.network = None
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.max_position_element = max_position_element
        self.position_size = position_size

        # Calculate total input dimension
        # 2 (cos and sine), 4 (position: psp location and footpoints), 5 exponents (from 0 to max_position_element)
        r_distance_dim = 1  # assuming scalar
        pos_encoding_dim = 2 * position_size * (max_position_element + 1)
        total_input_dim = input_feature_dim + pos_encoding_dim + r_distance_dim

        self.network = nn.Sequential(
            nn.Linear(total_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.hidden_dim // 2, self.d_output),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, position, r_distance):
        batch_size = x.size(0)

        # Flatten input while preserving batch dimension
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Get position encoding
        pos_encoded = cos_sin_transformation(
            position, max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined = torch.cat([x, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1)

        return self.network(combined)


class SkipLinearHead(nn.Module):
    """Deep MLP head with skip connections for coordinate-aware regression.

    This module implements a deep architecture where the initial concatenated
    input (features + encodings) is re-injected at specified layers. This
    residual-style connection helps maintain high-frequency coordinate
    information throughout the depth of the network.

    Args:
        d_output (int): Dimension of the final output.
        input_feature_dim (int): Dimension of a single input frame.
        max_position_element (int): Highest power for harmonic encoding. Defaults to 4.
        position_size (int): Number of raw coordinate variables. Defaults to 4.
        hidden_dim (int): Latent width of the hidden layers. Defaults to 16.
        skips (list[int]): Layer indices where the initial input is concatenated
            back into the hidden state. Defaults to [4].
        include_raw_coordinates (bool): If True, appends non-encoded coordinates
            to the input vector. Defaults to False.
        num_hidden_layers (int): Total number of linear layers in the backbone.
            Defaults to 8.
        number_of_frames (int): Number of temporal frames to flatten. Defaults to 1.

    Attributes:
        d_output (int): Output dimensionality.
        hidden_dim (int): Latent width of the network.
        max_position_element (int): Harmonic encoding complexity.
        position_size (int): Number of coordinate inputs.
        skips (list[int]): Indices of layers performing skip-connections.
        num_hidden_layers (int): Total depth of the MLP.
        pts_linears (nn.ModuleList): Collection of linear layers including
            skip-connection logic.
        output_linear (nn.Linear): Final layer mapping to output dimension.
    """

    def __init__(
        self,
        d_output,
        input_feature_dim,
        max_position_element=4,
        position_size=4,
        hidden_dim=16,
        skips=[4],
        include_raw_coordinates=False,
        num_hidden_layers=8,
        number_of_frames=1,
    ):
        super().__init__()
        self.d_output = d_output
        self.network = None
        self.hidden_dim = hidden_dim
        self.max_position_element = max_position_element
        self.position_size = position_size
        self.skips = skips
        self.num_hidden_layers = num_hidden_layers

        # Calculate total input dimension
        # 2 (cos and sine), 4 (position: psp location and footpoints), 5 exponents (from 0 to max_position_element)
        r_distance_dim = 1  # assuming scalar
        pos_encoding_dim = 2 * position_size * (max_position_element + 1)
        total_input_dim = (
            input_feature_dim * number_of_frames + pos_encoding_dim + r_distance_dim
        )

        if include_raw_coordinates:
            total_input_dim = total_input_dim + position_size

        self.pts_linears = nn.ModuleList(
            [nn.Linear(total_input_dim, hidden_dim)]
            + [
                (
                    nn.Linear(hidden_dim, hidden_dim)
                    if i not in self.skips
                    else nn.Linear(hidden_dim + total_input_dim, hidden_dim)
                )
                for i in range(self.num_hidden_layers - 1)
            ]
        )

        self.output_linear = nn.Linear(hidden_dim, d_output)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, position, r_distance):
        batch_size = x.size(0)

        # Flatten input while preserving batch dimension
        if x.dim() > 2:
            x = x.view(batch_size, -1)

        # Get position encoding
        pos_encoded = cos_sin_transformation(
            position, max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined_net = torch.cat(
            [x, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1
        )
        init_net = combined_net

        for i, l in enumerate(self.pts_linears):
            combined_net = self.pts_linears[i](combined_net)
            combined_net = F.relu(combined_net)
            if i in self.skips:
                combined_net = torch.cat([combined_net, init_net], -1)

        outputs = self.output_linear(combined_net)

        return outputs


class ClsLinear(nn.Module):
    """MLP head designed for Transformer [CLS] token representations.

    This module specifically extracts the class (CLS) token from the first
    index of the input sequence and combines it with physical metadata
    (positional encodings and radial distance) for final prediction.

    Args:
        d_output (int): Dimension of the final output.
        embedding_dim (int): Dimensionality of the tokens in the input sequence.
        max_position_element (int): Highest power for harmonic encoding. Defaults to 4.
        position_size (int): Number of raw coordinate variables. Defaults to 4.
        hidden_dim (int): Hidden width of the MLP. Defaults to 16.
        p_drop (float): Dropout probability. Defaults to 0.1.

    Attributes:
        d_output (int): Output dimensionality.
        embedding_dim (int): Dimension of the input tokens (D).
        hidden_dim (int): Hidden layer width.
        p_drop (float): Dropout rate.
        max_position_element (int): Complexity of the harmonic encoding.
        position_size (int): Number of coordinate inputs.
        network (nn.Sequential): MLP layers processing the combined CLS and
            metadata vector.
    """

    def __init__(
        self,
        d_output,
        embedding_dim,
        max_position_element=4,
        position_size=4,
        hidden_dim=16,
        p_drop=0.1,
    ):
        super().__init__()
        self.d_output = d_output
        self.embedding_dim = embedding_dim
        self.network = None
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.max_position_element = max_position_element
        self.position_size = position_size

        # Calculate total input dimension
        # 2 (cos and sine), 4 (position: psp location and footpoints), 5 exponents (from 0 to max_position_element)
        r_distance_dim = 1  # assuming scalar
        pos_encoding_dim = 2 * position_size * (max_position_element + 1)
        total_input_dim = embedding_dim + pos_encoding_dim + r_distance_dim

        self.network = nn.Sequential(
            nn.Linear(total_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.hidden_dim // 2, self.d_output),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, position, r_distance):  # x: [B, 256, 512]
        batch_size = x.size(0)
        cls_tokens = x[:, 0, :]  # [1, B]

        pos_encoded = cos_sin_transformation(
            position, max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined = torch.cat(
            [cls_tokens, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1
        )

        return self.network(combined)
