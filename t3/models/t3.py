"""
Transferable Tactile Transformer (T3)

Author: Jialiang (Alan) Zhao
Email: alanzhao@csail.mit.edu
MIT License
"""

import hydra

import os
from torch import nn
from t3.utils import logging

from omegaconf import OmegaConf

class T3(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.encoders = {}
        self.decoders = {}
        self.loss_funcs = {}
        cfg = OmegaConf.to_container(cfg)
        cfg = OmegaConf.create(cfg)

        self.trunk = hydra.utils.instantiate(cfg.shared_trunk)
        self._is_trunk_transformer = "Transformer" in cfg.shared_trunk._target_

        for name, encoder_cfg in cfg.encoders.items():
            self.encoders[name] = hydra.utils.instantiate(encoder_cfg)
        
        for name, decoder_cfg in cfg.decoders.items():
            self.decoders[name] = hydra.utils.instantiate(decoder_cfg)
            if hasattr(decoder_cfg, "loss_func"):
                self.loss_funcs[name] = hydra.utils.instantiate(decoder_cfg.loss_func)
            else:
                self.loss_funcs[name] = None
        
        self.encoders = nn.ModuleDict(self.encoders)
        self.decoders = nn.ModuleDict(self.decoders)
        self.loss_funcs = nn.ModuleDict(self.loss_funcs)
        self._encoder_domain = None
        self._decoder_domain = None
    
    def model_summary(self):
        print("==========================================")
        encoder_parameters = sum(p.numel() for p in self.encoders.parameters() if p.requires_grad)
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        decoder_parameters = sum(p.numel() for p in self.decoders.parameters() if p.requires_grad)
        n_parameters = encoder_parameters + trunk_parameters + decoder_parameters
        logging(
            f"number of total trainable params (M): {n_parameters / 1.0e6:.3f} \n\
                encoder: {encoder_parameters / 1.0e6:.3f} \n\
                    trunk: {trunk_parameters / 1.0e6:.3f} \n\
                        decoder: {decoder_parameters / 1.0e6:.3f}", True, "green")
    
    def set_domains(self, encoder_domain, decoder_domain, forward_mode):
        assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
        assert decoder_domain in self.decoders, f"decoder domain {decoder_domain} not found in decoders"
        self._encoder_domain = encoder_domain
        self._decoder_domain = decoder_domain
        self._forward_mode = forward_mode
    
    def freeze_encoder(self, encoder_domain=None):
        if encoder_domain is None:
            for encoder in self.encoders.values():
                encoder.freeze()
        else:
            assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
            self.encoders[encoder_domain].freeze()
    
    def unfreeze_encoder(self, encoder_domain=None):
        if encoder_domain is None:
            for encoder in self.encoders.values():
                encoder.unfreeze()
        else:
            assert encoder_domain in self.encoders, f"encoder domain {encoder_domain} not found in encoders"
            self.encoders[encoder_domain].unfreeze()
    
    def freeze_trunk(self):
        self.trunk.freeze()
    
    def unfreeze_trunk(self):
        self.trunk.unfreeze()
    
    def encoder_output(self, x):
        if self._forward_mode == "single_tower":
            return self.encoders[self._encoder_domain](x)[0]
        elif self._forward_mode == "multi_tower":
            return [self.encoders[self._encoder_domain](x)[0] for x in xs]
        else:
            raise ValueError(f"forward mode {self._forward_mode} not recognized")
    
    def forward(self, *args, **kwargs):
        if self._forward_mode == "single_tower":
            return self.single_tower_forward(*args, **kwargs)
        elif self._forward_mode == "multi_tower":
            return self.multi_tower_forward(*args, **kwargs)
        else:
            raise ValueError(f"forward mode {self._forward_mode} not recognized")
    
    def single_tower_forward(self, x):
        x = self.encoders[self._encoder_domain](x)
        x = self.trunk(x)
        x = self.decoders[self._decoder_domain](x)
        return x
    
    def multi_tower_forward(self, *xs):
        xs = [self.encoders[self._encoder_domain](x) for x in xs]
        xs = [self.trunk(x) for x in xs]
        x = self.decoders[self._decoder_domain](*xs)
        return x
    
    def compute_loss(self, y_pred, y_true):
        return self.loss_funcs[self._decoder_domain](y_pred, y_true)
    
    def save_components(self, dir):
        os.makedirs(f"{dir}/encoders", exist_ok=True)
        os.makedirs(f"{dir}/decoders", exist_ok=True)
        for encoder_name, encoder in self.encoders.items():
            encoder.save(f"{dir}/encoders/{encoder_name}.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.save(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.save(f"{dir}/trunk.pth")

    def load_components(self, dir):
        for encoder_name, encoder in self.encoders.items():
            encoder.load(f"{dir}/encoders/{encoder_name}.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.load(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.load(f"{dir}/trunk.pth")

def make_T3_tiny(cfg):
    return T3(cfg)

"""
Unified Transferable Tactile Transformer (T3_Uni)
A version of T3 that uses a single encoder for unified modality learning

Based on the original T3 by Jialiang (Alan) Zhao
"""

class T3_Uni(T3):
    def __init__(self, cfg, **kwargs):
        # Initialize nn.Module but not T3's full initialization
        super(T3, self).__init__()
        
        self.cfg = cfg
        self.decoders = {}
        self.loss_funcs = {}
        
        # Convert config
        cfg = OmegaConf.to_container(cfg, resolve=False)
        cfg = OmegaConf.create(cfg)
        
        # Initialize trunk
        self.trunk = hydra.utils.instantiate(cfg.shared_trunk)
        self._is_trunk_transformer = "Transformer" in cfg.shared_trunk._target_
        
        # Initialize single unified encoder instead of multiple encoders
        self.encoder = hydra.utils.instantiate(cfg.encoder)
        
        # Initialize decoders (same as parent)
        for name, decoder_cfg in cfg.decoders.items():
            self.decoders[name] = hydra.utils.instantiate(decoder_cfg)
            if hasattr(decoder_cfg, "loss_func"):
                self.loss_funcs[name] = hydra.utils.instantiate(decoder_cfg.loss_func)
            else:
                self.loss_funcs[name] = None
        
        # Convert to ModuleDicts
        self.decoders = nn.ModuleDict(self.decoders)
        self.loss_funcs = nn.ModuleDict(self.loss_funcs)
        
        # Set defaults
        self._decoder_domain = None
        self._forward_mode = "single_tower"
        self.path = cfg.encoder_path
        
    def model_summary(self):
        print("==========================================")
        encoder_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        decoder_parameters = sum(p.numel() for p in self.decoders.parameters() if p.requires_grad)
        n_parameters = encoder_parameters + trunk_parameters + decoder_parameters
        logging(
            f"number of total trainable params (M): {n_parameters / 1.0e6:.3f} \n\
                encoder: {encoder_parameters / 1.0e6:.3f} \n\
                    trunk: {trunk_parameters / 1.0e6:.3f} \n\
                        decoder: {decoder_parameters / 1.0e6:.3f}", True, "green")
    
    def set_decoder(self, decoder_domain, forward_mode="single_tower"):
        """Set only the decoder domain since there's just one encoder"""
        assert decoder_domain in self.decoders, f"decoder domain {decoder_domain} not found in decoders"
        self._decoder_domain = decoder_domain
        self._forward_mode = forward_mode
    
    def encoder_output(self, x):
        if self._forward_mode == "single_tower":
            return self.encoder(x)[0]
        elif self._forward_mode == "multi_tower":
            return [self.encoder(x)[0] for x in xs]
        else:
            raise ValueError(f"forward mode {self._forward_mode} not recognized")
    
    # For backward compatibility with T3 API
    def set_domains(self, encoder_domain, decoder_domain, forward_mode):
        """Override to ignore encoder_domain since we only have one encoder"""
        assert decoder_domain in self.decoders, f"decoder domain {decoder_domain} not found in decoders"
        self._decoder_domain = decoder_domain
        self._forward_mode = forward_mode
    
    def freeze_encoder(self, encoder_domain=None):
        """Freeze the unified encoder, ignoring encoder_domain"""
        self.encoder.freeze()
    
    def unfreeze_encoder(self, encoder_domain=None):
        """Unfreeze the unified encoder, ignoring encoder_domain"""
        self.encoder.unfreeze()
    
    def single_tower_forward(self, x):
        """Forward pass with single tower architecture"""
        x = self.encoder(x)
        x = self.trunk(x)
        x = self.decoders[self._decoder_domain](x)
        return x
    
    def multi_tower_forward(self, *xs):
        """Forward pass with multi-tower architecture"""
        xs = [self.encoder(x) for x in xs]
        xs = [self.trunk(x) for x in xs]
        x = self.decoders[self._decoder_domain](*xs)
        return x
    
    def save_components(self, dir):
        """Save model components"""
        os.makedirs(f"{dir}/decoders", exist_ok=True)
        self.encoder.save(f"{dir}/encoder.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.save(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.save(f"{dir}/trunk.pth")

    def load_components(self, dir):
        """Load model components"""
        self.encoder.load(f"{dir}/encoder.pth")
        for decoder_name, decoder in self.decoders.items():
            decoder.load(f"{dir}/decoders/{decoder_name}.pth")
        self.trunk.load(f"{dir}/trunk.pth")
    
    # Add unified-model-specific methods
    def get_embeddings(self, x):
        """Get encoded embeddings"""
        return self.encoder(x)

    def load_encoder(self):
        """Load encoder weights from a pretrained checkpoint."""
        self.encoder.load(self.path)

def make_T3_Uni(cfg):
    """Factory function to create a T3_Uni model"""
    return T3_Uni(cfg)