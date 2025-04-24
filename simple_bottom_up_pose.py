# import cv2
import numpy as np

class ScaleAwareHeatmapGenerator:
    def __init__(self, output_res, num_joints, use_jnt=True, jnt_thr=0.01, use_int=True, shape=False, shape_weight=1.0, pauta=3):
        self.output_res = output_res
        self.num_joints = num_joints
        self.use_jnt = use_jnt
        self.jnt_thr = jnt_thr
        self.use_int = use_int
        self.shape = shape
        self.shape_weight = shape_weight
        self.pauta = pauta

    def get_heat_val(self, sigma, x, y, x0, y0):
        return np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints, sigmas, ct_sigma, bg_weight=1.0):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res), dtype=np.float32)
        ignored_hms = 2 * np.ones((1, self.output_res, self.output_res), dtype=np.float32)
        hms_list = [hms, ignored_hms]
        for p in joints:
            for idx, pt in enumerate(p):
                if idx < self.num_joints - 1:
                    sigma = pt[3] if len(pt) > 3 else sigmas[idx]
                else:
                    sigma = ct_sigma
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    if self.use_jnt:
                        radius = np.sqrt(np.log(1 / self.jnt_thr) * 2 * sigma ** 2)
                        if self.use_int:
                            radius = int(np.floor(radius))
                        ul = int(np.floor(x - radius - 1)), int(np.floor(y - radius - 1))
                        br = int(np.ceil(x + radius + 2)), int(np.ceil(y + radius + 2))
                    else:
                        ul = int(np.floor(x - self.pauta * sigma - 1)), int(np.floor(y - self.pauta * sigma - 1))
                        br = int(np.ceil(x + self.pauta * sigma + 2)), int(np.ceil(y + self.pauta * sigma + 2))
                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                    joint_rg = np.zeros((bb - aa, dd - cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx - cc] = self.get_heat_val(sigma, sx, sy, x, y)
                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][0, aa:bb, cc:dd] = 1.
        hms_list[1][hms_list[1] == 2] = bg_weight
        return hms_list

# Example usage in your pipeline:
# output_res = 64  # or your heatmap size
# num_joints = 17  # or the number of keypoints
# generator = ScaleAwareHeatmapGenerator(output_res, num_joints)
# joints = ... # shape: (num_people, num_joints, 4) where last dim is (x, y, v, sigma)
# sigmas = ... # list of per-keypoint sigmas
# ct_sigma = ... # sigma for center joint if used
# heatmaps = generator(joints, sigmas, ct_sigma)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: generate a heatmap for a single person with 3 keypoints
    output_res = 64
    num_joints = 3
    generator = ScaleAwareHeatmapGenerator(output_res, num_joints)

    # Example keypoints: (x, y, v, sigma)
    # v=1 means visible, v=0 means not visible
    joints = np.array([
        [
            [32, 16, 1, 2],  # keypoint 1
            [48, 48, 1, 4],  # keypoint 2
            [16, 48, 1, 3],  # keypoint 3
        ]
    ])
    sigmas = [2, 4, 3]  # fallback sigmas if not in pt[3]
    ct_sigma = 2  # not used here, but required by the call signature

    heatmaps, _ = generator(joints, sigmas, ct_sigma)

    # Visualize each keypoint's heatmap
    fig, axes = plt.subplots(1, num_joints, figsize=(12, 4))
    for i in range(num_joints):
        axes[i].imshow(heatmaps[i], cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Keypoint {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
