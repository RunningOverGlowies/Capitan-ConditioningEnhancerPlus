import torch
import comfy.model_management as mm
import math

class ConditioningEnhancerPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Input prompt embeddings."
                }),
                "enhance_strength": ("FLOAT", {
                    "default": 0.0, "min": -5.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Positive: Polishes/Blends. Negative: Sharpen/Literal control."
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stabilizes embeddings. Recommended for Qwen."
                }),
                "add_self_attention": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allows prompt parts to influence each other (coherence)."
                }),
                "attn_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Intensity of self-attention mixing."
                }),
                "mlp_hidden_mult": ("INT", {
                    "default": 2, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Network width. Higher = more detail extraction but risky."
                }),
                "activation": (["GELU", "ReLU", "SiLU", "Tanh"], {
                    "tooltip": "Math function style. GELU=smooth, ReLU=sharp."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Fixed seed for consistent enhancement patterns."
                }),
            },
            "optional": {
                "noise_injection": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Adds faint noise to break stuck patterns."
                }),
                "clamp_output": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Limits extreme values to stop artifacts. If artifacts appear at high strength, try 20.0 - 50.0. 0=Off."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "enhance"
    CATEGORY = "conditioning/enhance"

    def enhance(self, conditioning, enhance_strength, normalize, add_self_attention, attn_strength, mlp_hidden_mult, activation, seed, noise_injection=0.0, clamp_output=0.0):
        if not conditioning:
            return (conditioning, )

        device = mm.get_torch_device()
        enhanced = []

        # Map activation strings to functions
        act_map = {
            "GELU": torch.nn.GELU(),
            "ReLU": torch.nn.ReLU(),
            "SiLU": torch.nn.SiLU(),
            "Tanh": torch.nn.Tanh(),
        }
        act_fn = act_map.get(activation, torch.nn.GELU())

        for emb_tensor, meta in conditioning:
            # Create a generator based on the seed for reproducible random weights
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            
            emb = emb_tensor.to(device)  # [B, seq, 2560]
            orig_dtype = emb.dtype

            # 1. Noise Injection (Dithering) - Breaks quantization banding
            if noise_injection > 0:
                noise = torch.randn(emb.shape, generator=gen, device=device, dtype=emb.dtype) * noise_injection
                emb = emb + noise

            # 2. Normalization
            if normalize:
                # Add epsilon to prevent division by zero
                emb = (emb - emb.mean(dim=-1, keepdim=True)) / (emb.std(dim=-1, keepdim=True) + 1e-6)

            # 3. Dynamic MLP Construction
            dim = emb.shape[-1]
            hidden_dim = dim * mlp_hidden_mult

            # Define weights manually using the generator to ensure seed compliance
            # Layer 1
            w1 = torch.empty((hidden_dim, dim), device=device, dtype=emb.dtype)
            b1 = torch.zeros(hidden_dim, device=device, dtype=emb.dtype)
            torch.nn.init.kaiming_uniform_(w1, a=math.sqrt(5), generator=gen)
            
            # Layer 2
            w2 = torch.empty((dim, hidden_dim), device=device, dtype=emb.dtype)
            b2 = torch.zeros(dim, device=device, dtype=emb.dtype)
            # Init Layer 2 close to Identity/Zero to make 'strength' behavior linear
            torch.nn.init.kaiming_uniform_(w2, a=math.sqrt(5), generator=gen)
            w2 = w2 * 0.1 # Scale down initial random influence

            # Functional MLP pass (Manual Linear layers)
            # x @ W.t() + b
            refined = torch.nn.functional.linear(emb, w1, b1)
            refined = act_fn(refined) # Apply selected activation
            refined = torch.nn.functional.linear(refined, w2, b2)

            # 4. Blend
            blended = emb + enhance_strength * (refined - emb)

            # 5. Self-Attention (Deterministic)
            if add_self_attention:
                # We manually create Q, K, V projections to keep it seeded
                q_proj = torch.nn.Linear(dim, dim, bias=False, device=device, dtype=emb.dtype)
                k_proj = torch.nn.Linear(dim, dim, bias=False, device=device, dtype=emb.dtype)
                v_proj = torch.nn.Linear(dim, dim, bias=False, device=device, dtype=emb.dtype)
                
                # Init weights with seed
                with torch.no_grad():
                    torch.nn.init.xavier_uniform_(q_proj.weight, generator=gen)
                    torch.nn.init.xavier_uniform_(k_proj.weight, generator=gen)
                    torch.nn.init.xavier_uniform_(v_proj.weight, generator=gen)

                # Calculate attention manually or use optimized functional (scaled dot product)
                q = q_proj(blended)
                k = k_proj(blended)
                v = v_proj(blended)
                
                # Scaled Dot-Product Attention
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )
                
                blended = blended + (attn_strength * attn_out)

            # 6. Safety Clamp
            if clamp_output > 0:
                blended = torch.clamp(blended, -clamp_output, clamp_output)

            enhanced.append((blended.to("cpu", dtype=orig_dtype), meta))

        return (enhanced, )

NODE_CLASS_MAPPINGS = {
    "ConditioningEnhancerPlus": ConditioningEnhancerPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningEnhancerPlus": "Conditioning Enhancer Plus(Qwen/Z-Image)",
}
