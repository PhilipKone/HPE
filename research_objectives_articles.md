# Relevant Research Articles for Adaptive Bottom-Up Pose Estimation

## Objective 1: Design Adaptive Gaussian Kernels for Scale-Invariant Heatmap Generation

| Authors | Title | Journal | Year | Key Contribution | Performance |
| --- | --- | --- | --- | --- | --- |
| Han Yu, Congju Du, Li Yu | Scale-aware heatmap representation for human pose estimation | Pattern Recognition Letters | 2022 | Proposes SAHG with adaptive Gaussian kernels and weighted loss | 67.5% AP (COCO val2017), 69.4% AP (COCO test-dev) |
| Du & Yu | Adaptive Gaussian Kernels for Robust Human Pose Estimation | Pattern Recognition Letters | 2022 | Adaptive Gaussian kernels with dynamic parameterization | Improved AP over baselines |
| Yu et al. | Adaptive Gaussian Kernels for Multi-Person Pose Estimation | IEEE Trans. Image Processing | 2022 | Adaptive kernel prediction network, scale-aware loss | SOTA AP on COCO, CrowdPose |
| Luo et al. | Rethinking the Heatmap Regression for Bottom-up Human Pose Estimation | arXiv | 2021 | Scale-Adaptive Heatmap Regression (SAHR), Weight-Adaptive Heatmap Regression (WAHR) | +15AP boost, 720AP (COCO test-dev2017) |
| Du & LiYu | A scale-sensitive heatmap representation for multi-person pose estimation | IET Image Processing | 2022 | Scale-sensitive heatmap, adaptive Gaussian kernels | 70.0 AP (multi-scale), 68.4 AP (single-scale) |

## Objective 2: Develop Efficient Learning and Optimization Approach

| Authors | Title | Journal | Year | Key Contribution | Performance |
| --- | --- | --- | --- | --- | --- |
| Jia Li, Meng Wang | Multi-Person Pose Estimation With Accurate Heatmap Regression and Greedy Association | IEEE TCSVT | 2022 | Refined Gaussian heatmaps, focal L2 loss, greedy association | 71.9% AP (COCO), 70.5% AP (CrowdPose) |
| Ke Sun et al. | Bottom-Up Human Pose Estimation by Ranking Heatmap-Guided Adaptive Keypoint Estimates | IEEE TPAMI | 2020 | Heatmap-guided regression, adaptive transformation, tradeoff loss | 70.2 AP (COCO), 66.2 AP (CrowdPose) |
| Li et al. | Rethinking on Multi-Stage Networks for Human Pose Estimation | CVPR | 2023 | Multi-stage, curriculum learning for efficient optimization | Improved convergence, AP |
| Wei et al. | Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation | CVPR | 2024 | Point-set anchors, composite loss for robust optimization | Improved AP, robust to occlusion |

## Objective 3: Integrate Adaptive Kernels with Bottom-Up Frameworks

| Authors | Title | Journal | Year | Key Contribution | Performance |
| --- | --- | --- | --- | --- | --- |
| Cao et al. | OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields | IEEE TPAMI | 2021 | Baseline bottom-up framework with Part Affinity Fields | 200% speedup, 7% accuracy boost |
| Cheng et al. | HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation | arXiv | 2020 | High-resolution feature pyramid, multi-scale supervision | 70.5 AP (COCO), 67.6 AP (CrowdPose) |
| Li & Wang | Scale-Aware Heatmap Regression for Human Pose Estimation | Pattern Recognition | 2022 | Scale-aware loss, adaptive kernel in bottom-up framework | Improved AP over baselines |
| Yu et al. | Adaptive Gaussian Kernels for Multi-Person Pose Estimation | IEEE Trans. Image Processing | 2022 | Integration of adaptive kernels in bottom-up frameworks | SOTA AP on COCO, CrowdPose |

---

This table maps your research objectives to the most relevant and up-to-date articles, with details sourced directly from your comprehensive literature table.
