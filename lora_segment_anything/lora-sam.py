from segment_anything.modeling import Sam

from torch import nn
import torch
from torch.nn.parameter import Parameter

import math


# TODO: Credit SAMed
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim :] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    def __init__(self, sam: Sam) -> None:
        """
        LoRA_SAM is the version of SAM with LoRA.

        Arguments:
          sam (Sam): Original Sam model
        """
        super().__init__()

        # Matrice's rank
        R = 1

        # Weights for the low rank matrices
        self.w_As = []
        self.w_Bs = []

        # Freeze the image encoder
        for param in sam.image_encoder.parameters():
            param.requires_grad = False

        # Build up our AB matrices
        for _, transf_layer in enumerate(sam.image_encoder.blocks):
            w_qkv_linear = transf_layer.attn.qkv

            # Size transformer layers
            self.dim = w_qkv_linear.in_features

            # Create a corresponding A and B layer sized r x t_dim
            w_a_linear_q = nn.Linear(self.dim, R, bias=False)
            w_b_linear_q = nn.Linear(R, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, R, bias=False)
            w_b_linear_v = nn.Linear(R, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            # Replace the layer with a version using A and B's qv
            transf_layer.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        # Initialize A and B
        self.reset_parameters()

        # The "new" SAM with LoRA
        self.sam = sam

    def reset_parameters(self) -> None:
        # A is initialized with the Kaiming Uniform
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))

        # B starts zeored
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def load_lora_params(self, filename: str) -> None:
        state_dict = torch.load(filename)

        # Loads AB matrices
        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        # Loads prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if "prompt_encoder" in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {
            k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)
        }
        sam_dict.update(prompt_encoder_new_state_dict)

        # Loads mask decoder
        mask_decoder_keys = [k for k in sam_keys if "mask_decoder" in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {
            k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)
        }
        sam_dict.update(mask_decoder_new_state_dict)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        self.sam.load_state_dict(sam_dict)

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)
