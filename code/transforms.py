import torch
import numpy as np
import random
from typing import List, Optional
from torch.distributions import Normal, Uniform


def random_affine_transforms(coordinates,
                             scale_vals=(0.0, 0.0235),
                             rotation_range=(-30, 30),
                             shear_range=(-0.15, 0.15),
                             reflection_probability=0.3):
    """
    Spatial transformations simulating realistic camera movements.
    :param coordinates: Feature vector
    :param scale_vals: Mean and std of Gaussian distribution used for scaling
    :param rotation_range: Low and high of Uniform distribution sampling range
    :param shear_range:  Low and high of uniform distribution sampling range
    :param reflection_probability: Probability of flipping left and right hand
    """

    # detects hand coordinates by shape since hands are (21, 3) - 3 channels
    is_hand = coordinates.shape[-1] == 3

    if is_hand:
        xy = coordinates[..., :2]
        z = coordinates[..., 2:]
    else:
        xy = coordinates
        z = None

    if scale_vals is not None:
        scale_factor = Normal(loc=scale_vals[0], scale=scale_vals[1]).sample()
        coordinates *= scale_factor

    center = xy.mean(dim=0, keepdim=True)  # centering transforms are applied around the center
    xy = xy - center

    if rotation_range is not None:
        angle = Uniform(low=rotation_range[0], high=rotation_range[1]).sample()
        theta = angle * np.pi / 180
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [cosine, -sine],
            [sine, cosine]
        ])

        xy = xy @ rotation_matrix.T

    if shear_range is not None:
        shear_x = Uniform(low=shear_range[0], high=shear_range[1]).sample()
        shear_y = Uniform(low=shear_range[0], high=shear_range[1]).sample()
        shear_matrix = torch.tensor([
            [torch.tensor(1.0), shear_x],
            [shear_y, torch.tensor(1.0)]
        ])

        xy = xy @ shear_matrix.T

    if reflection_probability is not None:
        do_reflect = Uniform(0.0, 1.0).sample() < reflection_probability
        if do_reflect:
            xy[..., 0] = -xy[..., 0]


    xy += center

    if is_hand:
        coordinates_transformed = torch.cat([xy, z], dim=-1)
    else:
        coordinates_transformed = xy

    return coordinates_transformed



def temporal_jitter_and_shuffle(
    frame_indices: List[int],
    num_frames: int,
    max_jitter: int = 1,
    jitter_prob: float = 0.2,
    shuffle_prob: float = 0.15,
) -> List[int]:
    """
    Apply temporal jitter and light frame shuffling to a list of frame indices.

    :param frame_indices: list of frame indices (feature vector)
    :param num_frames: number of available frames
    :param max_jitter: maximum absolute offset for jitter within frame
    :param jitter_prob: probability of jitter
    :param shuffle_prob: probability of swapping frame with neighboring frame
    """
    if num_frames <= 0 or len(frame_indices) == 0:
        return frame_indices

    # Randomly nudge frames
    jittered = []
    for idx in frame_indices:
        if max_jitter > 0 and random.random() < jitter_prob:
            delta = random.randint(-max_jitter, max_jitter)
            new_idx = idx + delta
            new_idx = max(0, min(num_frames - 1, new_idx))   # clip to valid range
            jittered.append(new_idx)
        else:
            jittered.append(idx)

    # Randomly swap frames with its neighbor
    shuffled = jittered[:]
    i = 0
    while i < len(shuffled) - 1:
        if random.random() < shuffle_prob:
            shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
            i += 2             # skip next to avoid double-swapping
        else:
            i += 1

    return shuffled
