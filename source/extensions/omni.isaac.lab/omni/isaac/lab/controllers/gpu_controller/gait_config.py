import numpy as np


class GaitConfig:
    stepping_frequency = 2
    initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    desired_base_height = 0.28
    foot_height = 0.1
    foot_landing_clearance = 0.
    max_velocity = np.array([0.6, 0.5, 0.5], dtype=np.float32)



