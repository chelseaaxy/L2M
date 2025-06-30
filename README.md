
# Lift to Match (L2M): Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space

*Accepted to ICCV 2025 Conference*

---

## 🧠 Overview

**Lift to Match (L2M)** is a novel two-stage framework for **dense feature matching** that lifts 2D images into 3D space to enhance feature generalization and robustness. Unlike traditional methods that depend on multi-view image pairs, L2M is trained on large-scale, diverse single-view image collections.

- **Stage 1:** Learn a **3D-aware ViT-based encoder** using multi-view image synthesis and 3D Gaussian feature representation.
- **Stage 2:** Learn a **feature decoder** through novel-view rendering and synthetic data, enabling robust matching across diverse scenarios.

> 🚧 Code under construction.

---

## 🧪 Feature Visualization

We compare the 3D-aware ViT encoder from L2M (Stage 1) with other recent methods:

- **DINOv2**
- **FIT3D**
- **Ours: L2M Encoder**

Below are feature comparison results on the Sacré-Cœur dataset:

<div align="center">
  <img src="./assets/sacre_coeur_A_compare.png" width="90%">
  <br/>
</div>

<div align="center">
  <img src="./assets/sacre_coeur_B_compare.png" width="90%">
  <br/>
</div>

---

## 🏗️ Data Generation (WIP)

We synthesize multi-view images and 3D-aware Gaussian features from single-view inputs.  
Scripts for data generation will be released soon.

---

## 🏋️‍♀️ Model Training (Stage 1)

We provide pretrained weights for the 3D-aware ViT encoder.

> 🔗 **[Download pretrained encoder weights](#)** (Coming soon)

You can visualize features using:

```bash
python vis_feats.py --input ./assets/sacre_coeur_A.jpg --model vit_encoder.pth
```

---

## 🚀 Inference & Stage 2 (Coming Soon)

The second stage—feature decoding with novel-view rendering—is **under development**. Stay tuned!

---

## 📌 Citation

```bibtex
@article{liang2025lift2match,
  title={Learning Dense Feature Matching via Lifting Single 2D Image to 3D Space},
  author={Liang, Yingping and Hu, Yutao and Shao, Wenqi and Fu, Ying},
  journal={ICCV},
  year={2025}
}
```

---

## 📋 License

This project is licensed under **CC BY 4.0**.

---

## 🙋‍♂️ Acknowledgements

We build upon recent advances in ROMA and FIT3D.
