from types import SimpleNamespace

import numpy as np


def make_rotation(rot_matrix: np.ndarray) -> SimpleNamespace:
    """
     Create a rotation event to instruct the interpreter to rotate
     the gx, gy and gz waveforms according to the given rotation matrix.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

     Parameters
     ----------
     rot_matrix : np.ndarray
         Rotation matrix of shape (3, 3).

     Returns
     -------
     rotation : SimpleNamespace
         Rotation event.

    """
    rotation = SimpleNamespace()
    rotation.type = 'rotation'
    rotation.rot_matrix = rot_matrix

    return rotation
