import torch
import torchvision.transforms as T

REPO_DIR = ""
model_path = ""
model_name = ""


class DINO_vit:
    def __init__(self, model_name="", model_path=""):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_name = model_name
        self.model_path = model_path
        self.model = self.load_model()
        self.image_batch_size = 128
        resize_size = 224
        self.transform = self.make_transform(resize_size)

    def load_model(self):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.eval().to(self.device)
        return model
    
    def operate_batched_image(self, image_batch):
        if len(image_batch.shape) == 3:
            image_batch = image_batch.unsqueeze(0)

        # make sure there are atleast 6 images in the batch
        # if not append last image to make it 6
        actual_image_count = image_batch.size(0)
        while image_batch.size(0) < 6:
            image_batch = torch.cat((image_batch, image_batch[-1].unsqueeze(0)), dim=0)

        with torch.no_grad():
            feats = self.model.forward_features(image_batch.to(self.device))
            cls_token = feats["x_norm_clstoken"]
            features = cls_token[:actual_image_count]
        return features
    
    def batchify_image(self, All_image , batch_size = 128):
        image_batch_list = []
        for i in range(0, All_image.size(0), batch_size):
            image_batch_list.append(All_image[i:i+batch_size])


        return image_batch_list
    

    def extract_features(self, All_image, batch_size = 128):
        if isinstance(All_image, list):
            tensor_images = torch.stack([self.transform(img) for img in All_image])
        else:
            # already a tensor of shape (N, 3, H, W)
            tensor_images = All_image

        image_batch_list = self.batchify_image(tensor_images, batch_size)
        features_list = []
        for image_batch in image_batch_list:
            features = self.operate_batched_image(image_batch)
            features_list.append(features.cpu())
        All_features = torch.cat(features_list, dim=0)
        return All_features.cpu().numpy()
    


    def make_transform(self, resize_size: int = 224):
        def ensure_rgb(img):
            import numpy as np
            from PIL import Image

            # If input is numpy array
            if isinstance(img, np.ndarray):
                # If grayscale (H,W), expand to (H,W,3)
                if img.ndim == 2:
                    img = np.repeat(img[..., None], 3, axis=-1)
                img = (img * 255).astype("uint8") if img.dtype != np.uint8 else img
                img = Image.fromarray(img)

            # If already PIL, just ensure it's RGB
            if isinstance(img, Image.Image):
                img = img.convert("RGB")
            return img

        to_rgb = T.Lambda(ensure_rgb)
        resize = T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)
        crop = T.CenterCrop(224)
        to_tensor = T.ToTensor()
        normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return T.Compose([to_rgb, resize,crop, to_tensor, normalize])
    

if __name__ == "__main__":
    model = DINO_vit(model_name, model_path)
    dummy_image = torch.randn(500, 3, 518, 518)  # Batch of 10 random images
    features = model.extract_features(dummy_image, batch_size=128)
    print(features.shape)  # Should print torch.Size([10, feature_dim])
