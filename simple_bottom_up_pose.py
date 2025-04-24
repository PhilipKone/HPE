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

    output_res = 64
    num_joints = 17
    generator = ScaleAwareHeatmapGenerator(output_res, num_joints)

    # Example: synthetic keypoints for a single person (COCO order)
    # (x, y, v, sigma) for each keypoint
    joints = np.array([
        [
            [32, 8, 1, 2],   # nose
            [36, 6, 1, 2],   # left eye
            [28, 6, 1, 2],   # right eye
            [40, 10, 1, 2],  # left ear
            [24, 10, 1, 2],  # right ear
            [44, 20, 1, 3],  # left shoulder
            [20, 20, 1, 3],  # right shoulder
            [48, 36, 1, 3],  # left elbow
            [16, 36, 1, 3],  # right elbow
            [52, 52, 1, 3],  # left wrist
            [12, 52, 1, 3],  # right wrist
            [40, 56, 1, 4],  # left hip
            [24, 56, 1, 4],  # right hip
            [44, 64, 1, 4],  # left knee
            [20, 64, 1, 4],  # right knee
            [48, 60, 1, 4],  # left ankle
            [16, 60, 1, 4],  # right ankle
        ]
    ])
    sigmas = [2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4]  # Example per-keypoint sigmas
    ct_sigma = 2  # not used here

    heatmaps, _ = generator(joints, sigmas, ct_sigma)

    # Visualize all 17 keypoint heatmaps in a 4x5 grid
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for i in range(num_joints):
        ax = axes[i//5, i%5]
        ax.imshow(heatmaps[i], cmap='hot', interpolation='nearest')
        ax.set_title(f'Keypoint {i+1}')
        ax.axis('off')
    # Hide any unused subplots
    for j in range(num_joints, 20):
        axes[j//5, j%5].axis('off')
    plt.tight_layout()
    plt.show()
