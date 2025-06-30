import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import argparse
from romatch.models.transformer import vit_base

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- 可视化函数 --------------------
def vis_feat_map(features, patch_h, patch_w, resize_hw=(560, 560)):
    features = features.reshape(patch_h * patch_w, -1)
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(features)
    pca_feats = (pca_feats - pca_feats.mean(0)) / (pca_feats.std(0) + 1e-5)
    pca_feats = np.clip(pca_feats * 0.5 + 0.5, 0, 1)
    img = pca_feats.reshape(patch_h, patch_w, 3)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img.resize(resize_hw, Image.BICUBIC)



def save_all_visualizations(
    feat_dino, feat_fit3d, feat_L2M,
    patch_h, patch_w, base_name, save_dir, original_image=None
):
    os.makedirs(save_dir, exist_ok=True)

    # 单图保存
    img_dino = vis_feat_map(feat_dino, patch_h, patch_w)
    img_fit3d = vis_feat_map(feat_fit3d, patch_h, patch_w)
    img_L2M = vis_feat_map(feat_L2M, patch_h, patch_w)

    img_dino.save(os.path.join(save_dir, f"{base_name}_dino.png"))
    img_fit3d.save(os.path.join(save_dir, f"{base_name}_fit3d_fit3d.png"))
    img_L2M.save(os.path.join(save_dir, f"{base_name}_L2M.png"))

    # 拼图（含原图）
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, im, title in zip(
        ax,
        [original_image, img_dino, img_fit3d, img_L2M],
        ["Original", "DINOv2", "Fit3D", "L2M (Ours)"]
    ):
        a.imshow(im)
        a.set_title(title, fontsize=12)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_compare.png"))
    plt.close()


# -------------------- 特征提取函数 --------------------
def extract_features(model, image_tensor):
    with torch.no_grad():
        return model.forward_features(image_tensor)["x_norm_patchtokens"].squeeze(0).cpu().numpy()


# -------------------- 主脚本 --------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    patch_h, patch_w = 37, 37
    img_size = patch_h * 14  # = 560
    feat_dim = 768

    transform = T.Compose([
        T.GaussianBlur(9, sigma=(0.1, 2.0)),
        T.Resize((img_size, img_size)),
        T.CenterCrop((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    # 初始化模型
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0
    )

    # DINOv2
    dino = vit_base(**vit_kwargs).eval().to(device)
    dino_ckpt_raw = torch.load(args.ckpt_dino, map_location="cpu")
    dino_ckpt = {k.replace("model.", ""): v for k, v in dino_ckpt_raw.items()}
    dino.load_state_dict(dino_ckpt, strict=False)

    # Fit3D
    fit3d = vit_base(**vit_kwargs).eval().to(device)
    fit3d_ckpt_raw = torch.load(args.ckpt_fit3d, map_location="cpu")["model"]
    fit3d_ckpt = {k.replace("model.", ""): v for k, v in fit3d_ckpt_raw.items()}
    fit3d.load_state_dict(fit3d_ckpt, strict=False)

    # L2M (Ours)
    L2M = vit_base(**vit_kwargs).eval().to(device)
    L2M_ckpt = torch.load(args.ckpt_L2M, map_location="cpu")        
    L2M.load_state_dict(L2M_ckpt, strict=False)

    for i, img_path in enumerate(args.img_paths):
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # 提取特征
        feat_dino = extract_features(dino, x)
        feat_fit3d = extract_features(fit3d, x)
        feat_L2M = extract_features(L2M, x)

        # 保存图
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_all_visualizations(
            feat_dino, feat_fit3d, feat_L2M,
            patch_h, patch_w, base_name, args.save_dir,
            original_image=img
        )


        print(f"[{i+1}/{len(args.img_paths)}] Saved visualizations for {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_paths",
        nargs="+",
        default=[
            "assets/sacre_coeur_A.jpg",
            "assets/sacre_coeur_B.jpg"
        ],
        help="List of image paths"
    )
    parser.add_argument(
        "--ckpt_fit3d",
        default="ckpts/fit3d.pth",
        help="Original Fit3D checkpoint"
    )
    parser.add_argument(
        "--ckpt_L2M",
        default="ckpts/l2m_vit_base.pth",
        help="L2M Fit3D checkpoint"
    )
    parser.add_argument(
        "--ckpt_dino",
        default="ckpts/dinov2.pth",
        help="dino checkpoint"
    )
    parser.add_argument(
        "--save_dir",
        default="outputs_vis_feat",
        help="Directory to save visualizations"
    )
    args = parser.parse_args()
    main(args)
