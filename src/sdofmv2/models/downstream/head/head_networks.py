import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def cos_sin_transformation(position: torch.Tensor, max_power: int = 4, include_raw_coordinates=False) -> torch.Tensor:
    if position.ndim == 1:
        position = position.unsqueeze(0)  # [1, 4]

    powers = 2.0 ** torch.arange(max_power + 1, device=position.device, dtype=position.dtype)  # [5]
    scaled_pos = position.unsqueeze(-1) * powers  # [B, 4, 5]
    scaled_pos = scaled_pos.view(position.size(0), -1)  # [B, 20]

    cos_vec = torch.cos(scaled_pos)
    sin_vec = torch.sin(scaled_pos)

    concat = torch.cat([cos_vec, sin_vec], dim=-1) 

    if include_raw_coordinates:
        concat = torch.cat([concat, position])
    
    return   concat# [B, 40]

class TransformerHead(nn.Module):
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
        # 2: sin/cos 4:lat/lon, lat/lon
        self.projection = nn.Linear(
            2 * (max_position_element + 1) * 4 + 1,
            num_pos_token * input_token_dim
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
        pos_token = pos_token.view(x.shape[0], self.num_pos_token, self.input_token_dim) # [B, self.num_pos_token, 512]

        combined_emb = torch.cat([
            x[:, :1, :],      # CLS token
            pos_token,        # positional tokens
            x[:, 1:, :]       # Spatial patches
        ], dim=1) # [batch, # token, 512]

        embed = self.transformer_block(combined_emb)

        cls_token = embed[:, 0, :]  # [B, 512]
        return self.classifier(self.dropout(cls_token))
        

# class Transformer 

class SimpleLinear(nn.Module):
    def __init__(
            self,
            d_output,
            input_feature_dim,
            max_position_element=4,
            position_size=4,
            hidden_dim=16,
            p_drop=0.1
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
            nn.Linear(self.hidden_dim // 2, self.d_output)
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
        # print(f"linear input shape: {x.shape}")
        # Get position encoding
        pos_encoded = cos_sin_transformation(
            position,
            max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # print(f"encoded position shape: {pos_encoded.shape}")
        # print(f"r_tensor shape: {r_tensor.shape}")

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined = torch.cat([x, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1)

        return self.network(combined)


class SkipLinearHead(nn.Module):
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
            number_of_frames = 1
            
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
        total_input_dim = input_feature_dim*number_of_frames + pos_encoding_dim + r_distance_dim 

        if include_raw_coordinates:
            total_input_dim = total_input_dim + position_size
            
        self.pts_linears = nn.ModuleList(
            [nn.Linear(total_input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) 
                        if i not in self.skips else nn.Linear(hidden_dim + total_input_dim, hidden_dim) for i in range(self.num_hidden_layers-1)])

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
        # print(f"linear input shape: {x.shape}")
        # Get position encoding
        pos_encoded = cos_sin_transformation(
            position,
            max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # print(f"encoded position shape: {pos_encoded.shape}")
        # print(f"r_tensor shape: {r_tensor.shape}")

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined_net = torch.cat([x, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1)
        init_net = combined_net

        for i, l in enumerate(self.pts_linears):
            combined_net = self.pts_linears[i](combined_net)
            combined_net = F.relu(combined_net)
            if i in self.skips:
                combined_net = torch.cat([combined_net, init_net], -1)

        outputs = self.output_linear(combined_net)

        return outputs


class ClsLinear(nn.Module):
    def __init__(
            self,
            d_output,
            embedding_dim,
            max_position_element=4,
            position_size=4,
            hidden_dim=16,
            p_drop=0.1
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
            nn.Linear(self.hidden_dim // 2, self.d_output)
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
            position,
            max_power=self.max_position_element
        )
        pos_encoded = pos_encoded.detach().clone().to(dtype=x.dtype, device=x.device)
        r_tensor = r_distance.to(dtype=x.dtype, device=x.device)

        # Ensure position encoding matches batch size
        if pos_encoded.size(0) != batch_size:
            pos_encoded = pos_encoded.expand(batch_size, -1)

        # Concatenate and process
        r_tensor = r_tensor.view(batch_size, 1) if r_tensor.ndim == 1 else r_tensor
        combined = torch.cat([cls_tokens, pos_encoded, r_tensor.reshape(batch_size, -1)], dim=-1)

        return self.network(combined)