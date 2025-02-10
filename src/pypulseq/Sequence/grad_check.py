import numpy as np

from pypulseq import eps


def grad_check(self, block_index, check_g, duration, rot_event):
    """
    Check if connection to the previous block is correct.

    Parameters
    ----------
    block_index : int
        Current block index.
    check_g : SimpleNamespace
        Structure containing current gradient start and end (t, g) values for each
        axis.
    duration : float
        Current block duration.
    rot_event : SimpleNamespace
        Current block rotation event.

    Raises
    ------
    RuntimeError
        If either 1) initial block start with non-zero amplitude;
        2) gradients starting with non-zero amplitude have a delay;
        3) gradients starting with non-zero amplitude have different initial
           amplitude value than the previous block at connecting point;
        4) gradients ending with non-zero amplitude are not aligned with block raster;
        4) gradients ending with non-zero amplitude are not aligned have different initial
           amplitude value than the next block at connecting point.

    """
    if not self.rotation_library.data:
        grad_check_norot(self, block_index, check_g, duration)
    else:
        grad_check_rot(self, block_index, check_g, duration, rot_event)


def grad_check_norot(self, block_index, check_g, duration):
    """
    Check continuity of adjacent gradient events in absence of rotations.

    Parameters
    ----------
    block_index : int
        Current block index.
    check_g : SimpleNamespace
        Structure containing current gradient start and end (t, g) values for each
        axis.
    duration : float
        Current block duration.

    Raises
    ------
    RuntimeError
        If either 1) initial block start with non-zero amplitude;
        2) gradients starting with non-zero amplitude have a delay;
        3) gradients starting with non-zero amplitude have different initial
           amplitude value than the previous block at connecting point;
        4) gradients ending with non-zero amplitude are not aligned with block raster;
        4) gradients ending with non-zero amplitude are not aligned have different initial
           amplitude value than the next block at connecting point.

    """
    for grad_to_check in check_g.values():
        if (
            abs(grad_to_check.start[1])
            > self.system.max_slew * self.system.grad_raster_time
        ):  # noqa: SIM102
            if grad_to_check.start[0] > eps:
                raise RuntimeError(
                    "No delay allowed for gradients which start with a non-zero amplitude"
                )

        # Check whether any blocks exist in the sequence
        if self.next_free_block_ID > 1:
            # Look up the previous block (and the next block in case of a set_block call)
            if block_index == self.next_free_block_ID:
                # New block inserted
                prev_block_index = next(reversed(self.block_events))
                next_block_index = None
            else:
                blocks = list(self.block_events)
                try:
                    # Existing block overwritten
                    idx = blocks.index(block_index)
                    prev_block_index = blocks[idx - 1] if idx > 0 else None
                    next_block_index = (
                        blocks[idx + 1] if idx < len(blocks) - 1 else None
                    )
                except ValueError:
                    # Inserting a new block with non-contiguous numbering
                    prev_block_index = next(reversed(self.block_events))
                    next_block_index = None

            # Look up the last gradient value in the previous block
            last = 0
            if prev_block_index is not None:
                prev_id = self.block_events[prev_block_index][grad_to_check.idx]
                if prev_id != 0:
                    prev_lib = self.grad_library.get(prev_id)
                    prev_type = prev_lib["type"]

                    if prev_type == "t":
                        last = 0
                    elif prev_type == "g":
                        last = prev_lib["data"][5]

            # Check whether the difference between the last gradient value and
            # the first value of the new gradient is achievable with the
            # specified slew rate.
            if (
                abs(last - grad_to_check.start[1])
                > self.system.max_slew * self.system.grad_raster_time
            ):
                raise RuntimeError(
                    "Two consecutive gradients need to have the same amplitude at the connection point"
                )

            # Look up the first gradient value in the next block
            # (this only happens when using set_block to patch a block)
            if next_block_index is not None:
                next_id = self.block_events[next_block_index][grad_to_check.idx]
                if next_id != 0:
                    next_lib = self.grad_library.get(next_id)
                    next_type = next_lib["type"]

                    if next_type == "t":
                        first = 0
                    elif next_type == "g":
                        first = next_lib["data"][4]
                else:
                    first = 0

                # Check whether the difference between the first gradient value
                # in the next block and the last value of the new gradient is
                # achievable with the specified slew rate.
                if (
                    abs(first - grad_to_check.stop[1])
                    > self.system.max_slew * self.system.grad_raster_time
                ):
                    raise RuntimeError(
                        "Two consecutive gradients need to have the same amplitude at the connection point"
                    )
        elif (
            abs(grad_to_check.start[1])
            > self.system.max_slew * self.system.grad_raster_time
        ):
            raise RuntimeError(
                "First gradient in the the first block has to start at 0."
            )

        # Check if gradients, which do not end at 0, are as long as the block itself.
        if (
            abs(grad_to_check.stop[1])
            > self.system.max_slew * self.system.grad_raster_time
            and abs(grad_to_check.stop[0] - duration) > 1e-7
        ):
            raise RuntimeError(
                "A gradient that doesn't end at zero needs to be aligned to the block boundary."
            )


def grad_check_rot(self, block_index, check_g, duration, rot_event):
    """
    Check continuity of adjacent gradient events in presence of rotations.

    Parameters
    ----------
    block_index : int
        Current block index.
    check_g : SimpleNamespace
        Structure containing current gradient start and end (t, g) values for each
        axis.
    duration : float
        Current block duration.
    rot_event : SimpleNamespace
        Current block rotation event.

    Raises
    ------
    RuntimeError
        If either 1) initial block start with non-zero amplitude;
        2) gradients starting with non-zero amplitude have a delay;
        3) gradients starting with non-zero amplitude have different initial
           amplitude value than the previous block at connecting point;
        4) gradients ending with non-zero amplitude are not aligned with block raster;
        4) gradients ending with non-zero amplitude are not aligned have different initial
           amplitude value than the next block at connecting point.

    """
    for grad_to_check in check_g.values():
        # Check beginning of gradient event
        if (
            abs(grad_to_check.start[1])
            > self.system.max_slew * self.system.grad_raster_time
        ):  # noqa: SIM102
            if grad_to_check.start[0] > eps:
                raise RuntimeError(
                    "No delay allowed for gradients which start with a non-zero amplitude"
                )

    # Check whether any blocks exist in the sequence
    if self.next_free_block_ID > 1:
        current_has_rot = rot_event is not None
        previous_has_rot = False
        next_has_rot = False

        # Rotation extension ID
        rot_type_id = self.get_extension_type_ID("ROTATIONS")

        # Get indexes of previous and next blocks
        if block_index == self.next_free_block_ID:
            # New block inserted
            prev_block_index = next(reversed(self.block_events))
            next_block_index = None
        else:
            blocks = list(self.block_events)
            try:
                # Existing block overwritten
                idx = blocks.index(block_index)
                prev_block_index = blocks[idx - 1] if idx > 0 else None
                next_block_index = blocks[idx + 1] if idx < len(blocks) - 1 else None
            except ValueError:
                # Inserting a new block with non-contiguous numbering
                prev_block_index = next(reversed(self.block_events))
                next_block_index = None

        # 1) Comparison with previous block
        prev_grad_last = np.zeros(3, dtype=np.float32)
        if prev_block_index is not None:
            # Look up the last gradient value in the previous block
            for grad_to_check in check_g.values():
                prev_id = self.block_events[prev_block_index][grad_to_check.idx]
                if prev_id != 0:
                    prev_lib = self.grad_library.get(prev_id)
                    prev_type = prev_lib["type"]

                    # Trapezoids end with zeros,
                    # so we cannot have the same amplitude
                    # as the initial value of current block
                    if prev_type == "g":
                        prev_grad_last[grad_to_check.idx - 2] = prev_lib["data"][5]

            # Get previous block rotation matrix
            ext_id = self.block_events[prev_block_index][-1]
            while ext_id and not previous_has_rot:
                try:
                    ext = self.extensions_library.data.get(ext_id)
                    if ext[0] == rot_type_id:
                        previous_has_rot = True
                        previous_rotmat = np.asarray(
                            self.rotation_library.data.get(ext[1])
                        ).reshape((3, 3))
                    else:
                        ext_id = ext[-1]
                except KeyError:
                    ext_id = 0

            # Rotate previous gradient
            if previous_has_rot:
                prev_grad_last = previous_rotmat @ prev_grad_last

        # Look up the first gradient value in current block
        curr_grad_first = np.zeros(3, dtype=np.float32)
        for grad_to_check in check_g.values():
            curr_grad_first[grad_to_check.idx - 2] = grad_to_check.start[1]

        # Rotate current gradient
        if current_has_rot:
            curr_grad_first = rot_event.rot_matrix @ curr_grad_first

        # Compare current block with previous
        if any(
            abs(curr_grad_first - prev_grad_last)
            > self.system.max_slew * self.system.grad_raster_time
        ):
            raise RuntimeError(
                f"Error in block {block_index}: Two consecutive gradients need to have the same amplitude at the connection point."
            )

        # 2) Comparison with next block
        if next_block_index is not None:

            # Look up the last gradient value in the previous block
            next_grad_first = np.zeros(3, dtype=np.float32)
            for grad_to_check in check_g.values():
                next_id = self.block_events[next_block_index][grad_to_check.idx]
                if next_id != 0:
                    next_lib = self.grad_library.get(next_id)
                    next_type = next_lib["type"]

                    # Trapezoids start with zeros,
                    # so we cannot have the same amplitude
                    # as the final value of current block
                    if next_type == "g":
                        next_grad_first[grad_to_check.idx - 2] = next_lib["data"][4]

            # Get next block rotation matrix
            ext_id = self.block_events[next_block_index][-1]
            while ext_id and not next_has_rot:
                try:
                    ext = self.extensions_library.data.get(ext_id)
                    if ext[0] == rot_type_id:
                        next_has_rot = True
                        next_rotmat = np.asarray(
                            self.rotation_library.data.get(ext[1])
                        ).reshape((3, 3))
                    else:
                        ext_id = ext[-1]
                except KeyError:
                    ext_id = 0

            # Rotate next gradient
            if next_has_rot:
                next_grad_first = next_rotmat @ next_grad_first

            # Look up the last gradient value in current block
            curr_grad_last = np.zeros(3, dtype=np.float32)
            for grad_to_check in check_g.values():
                curr_grad_last[grad_to_check.idx - 2] = grad_to_check.stop[1]

            # Rotate current gradient
            if current_has_rot:
                curr_grad_last = rot_event.rot_matrix @ curr_grad_last

            # Compare current block with next
            if any(
                abs(curr_grad_last - next_grad_first)
                > self.system.max_slew * self.system.grad_raster_time
            ):
                raise RuntimeError(
                    f"Error in block {block_index}: Two consecutive gradients need to have the same amplitude at the connection point."
                )
    else:
        for grad_to_check in check_g.values():
            # Check beginning of gradient event
            if (
                abs(grad_to_check.start[1])
                > self.system.max_slew * self.system.grad_raster_time
            ):
                raise RuntimeError(
                    "First gradient in the the first block has to start at 0."
                )

    # Check if gradients, which do not end at 0, are as long as the block itself.
    for grad_to_check in check_g.values():
        if (
            abs(grad_to_check.stop[1])
            > self.system.max_slew * self.system.grad_raster_time
        ):
            if abs(grad_to_check.stop[0] - duration) > 1e-7:
                raise RuntimeError(
                    "A gradient that doesn't end at zero needs to be aligned to the block boundary."
                )
