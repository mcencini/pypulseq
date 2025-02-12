# Rotate gradient waveforms according to provided rotation matrix

import numpy as np
import copy

def rotate_ndarray(grad, rot_matrix):
    grad_channels = ["gx", "gy", "gz"]
    grad = copy.deepcopy(grad)

    # get length of gradient waveforms
    wave_length = []
    for ch in grad_channels:
        if ch in grad:
            wave_length.append(len(grad[ch]))

    assert (np.unique(wave_length) != 0).sum() == 1, "All the waveform along different channels must have the same length"

    wave_length = np.unique(wave_length)
    wave_length = wave_length[wave_length != 0].item()

    # create zero-filled waveforms for empty gradient channels
    for ch in grad_channels:
        if ch in grad:
            grad[ch] = grad[ch].squeeze()
        else:
            grad[ch] = np.zeros(wave_length)

    # stack matrix
    grad_mat = np.stack((grad["gx"], grad["gy"], grad["gz"]), axis=0) # (3, wave_length)

    # apply rotation
    grad_mat = rot_matrix @ grad_mat

    # put back in dictionary
    for j in range(3):
        ch = grad_channels[j]
        grad[ch] = grad_mat[j]

    # remove all zero waveforms
    for ch in grad_channels:
        if np.allclose(grad[ch], 0.0):
            grad.pop(ch)

    return grad