"""Shims for library-version drift (nunchaku 1.3.0dev vs diffusers 0.41)."""
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel


def patch_nunchaku_zimage():
    from nunchaku.models.transformers.transformer_zimage import (
        NunchakuZImageRopeHook, NunchakuZImageTransformer2DModel)

    # nunchaku passes patch_size/f_patch_size positionally; diffusers 0.41 inserted new params before them.
    def forward(self, x, t, cap_feats, return_dict=True, **kwargs):
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            return ZImageTransformer2DModel.forward(self, x, t, cap_feats, return_dict=return_dict, **kwargs)
        finally:
            self.unregister_rope_hook()

    NunchakuZImageTransformer2DModel.forward = forward
