import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class DINOv2_ViT:
    # def __init__(self, model_name="dinov2_vits14", resize_size=224, device=None):
    def __init__(self, model_name="dinov2_vitb14", resize_size=224, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        self.transform = self.make_transform(resize_size)

    def make_transform(self, resize_size: int = 224):
        # DINOv2 uses ImageNet normalization
        return T.Compose([
            T.Lambda(lambda img: self._ensure_rgb(img)),
            T.Resize(int(resize_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(resize_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

    @staticmethod
    def _ensure_rgb(img):
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype("uint8")
            img = Image.fromarray(img)

        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        return img

    def _to_tensor_batch(self, images):
        if isinstance(images, list):
            x = torch.stack([self.transform(im) for im in images], dim=0)
        elif torch.is_tensor(images):
            x = images
            # If user passes raw uint8-like [0,255], they must normalize themselves.
            # We assume it's already (N,3,H,W) float and normalized if coming as tensor.
        else:
            raise TypeError("images must be a list of PIL/ndarray or a torch.Tensor (N,3,H,W).")
        if x.ndim == 3:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def extract_cls(self, images, batch_size=128):
        x = self._to_tensor_batch(images)
        outs = []
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i+batch_size].to(self.device)
            # get_intermediate_layers with return_class_token=True returns (patch_tokens, cls_token)
            patch_tokens, cls_token = self.model.get_intermediate_layers(
                xb, n=1, return_class_token=True
            )[0]
            outs.append(cls_token.detach().cpu())  # (B, D)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def extract_patches(self, images, batch_size=128, layer = 11):
        x = self._to_tensor_batch(images)
        outs = []
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i+batch_size].to(self.device)
            patch_tokens = self.model.get_intermediate_layers(
                xb, n=[layer], return_class_token=False
            )[0]  # (B, N_patches, D)
            outs.append(patch_tokens.detach().cpu())
        return torch.cat(outs, dim=0)
    
    @torch.no_grad()
    def extract_features(self, images, batch_size=128,layer=11):
        x = self._to_tensor_batch(images)
        outs = []
        for i in range(0, x.size(0), batch_size):
            xb = x[i:i+batch_size].to(self.device)
            # get_intermediate_layers with return_class_token=True returns (patch_tokens, cls_token)
            patch_tokens, cls_token = self.model.get_intermediate_layers(
                xb, n=[layer], return_class_token=True
            )[0]
            outs.append(cls_token.detach().cpu())  # (B, D)
        return torch.cat(outs, dim=0)
    

    
    @torch.no_grad()
    def extract_cls_and_patches(self, images, batch_size=128,layer = 11):
        """
        Returns:
          cls_tokens:    (N, D)
          patch_tokens:  (N, N_patches, D)
        """
        x = self._to_tensor_batch(images)
        cls_outs = []
        patch_outs = []

        for i in range(0, x.size(0), batch_size):
            xb = x[i:i+batch_size].to(self.device)

            # One call -> last layer tokens
            # returns list length=1; element is (patch_tokens, cls_token)
            patch_tokens, cls_token = self.model.get_intermediate_layers(
                xb, n= [layer], return_class_token=True
            )[0]  # patch_tokens: (B, N_patches, D), cls_token: (B, D)

            cls_outs.append(cls_token.detach().cpu())
            patch_outs.append(patch_tokens.detach().cpu())

        cls_tokens = torch.cat(cls_outs, dim=0)
        patch_tokens = torch.cat(patch_outs, dim=0)
        return cls_tokens, patch_tokens


if __name__ == "__main__":
    model = DINOv2_ViT("dinov2_vitb14", resize_size=224)
    # smoke test with normalized-ish random inputs (already tensor, no transform applied)
    dummy = torch.randn(10, 3, 224, 224)
    cls = model.extract_cls(dummy, batch_size=4)
    patches = model.extract_patches(dummy, batch_size=4)
    print("CLS:", cls.shape)        # (10, 384) for vits14
    print("Patches:", patches.shape) # (10, N_patches, 384)
