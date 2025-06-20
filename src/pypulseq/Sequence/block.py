import math
from types import SimpleNamespace
from typing import List, Tuple, Union

import numpy as np

from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.Sequence.grad_check import grad_check
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.tracing import trace_enabled


def set_block(self, block_index: int, *args: SimpleNamespace) -> None:
    """
    Replace block at index with new block provided as block structure, add sequence block, or create a new block
    from events and store at position specified by index. The block or events are provided in uncompressed form and
    will be stored in the compressed, non-redundant internal libraries.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.get_block()`
    - `pypulseq.Sequence.sequence.Sequence.add_block()`

    Parameters
    ----------
    block_index : int
        Index at which block is replaced.
    args : SimpleNamespace
        Block or events to be replaced/added or created at `block_index`.

    Raises
    ------
    ValueError
        If trigger event that is passed is of unsupported control event type.
        If delay is set for a gradient even that starts with a non-zero amplitude.
    RuntimeError
        If two consecutive gradients to not have the same amplitude at the connection point.
        If the first gradient in the block does not start with 0.
        If a gradient that doesn't end at zero is not aligned to the block boundary.
    """
    events = block_to_events(*args)
    new_block = np.zeros(7, dtype=np.int32)
    duration = 0

    check_g = {
        0: SimpleNamespace(start=(0, 0), stop=(0, 0)),
        1: SimpleNamespace(start=(0, 0), stop=(0, 0)),
        2: SimpleNamespace(start=(0, 0), stop=(0, 0)),
    }  # Key-value mapping of index and  pairs of gradients/times
    extensions = []

    for event in events:
        if not isinstance(event, float):  # If event is not a block duration
            if event.type == 'rf':
                if new_block[1] != 0:
                    raise ValueError('Multiple RF events were specified in set_block')

                if hasattr(event, 'id'):
                    rf_id = event.id
                else:
                    rf_id, _ = register_rf_event(self, event)

                new_block[1] = rf_id
                duration = max(duration, event.shape_dur + event.delay + event.ringdown_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    self.block_trace[block_index].rf = event.trace
            elif event.type == 'grad':
                channel_num = ['x', 'y', 'z'].index(event.channel)
                idx = 2 + channel_num

                if new_block[idx] != 0:
                    raise ValueError(f'Multiple {event.channel.upper()} gradient events were specified in set_block')

                grad_start = (
                    event.delay + math.floor(event.tt[0] / self.grad_raster_time + 1e-10) * self.grad_raster_time
                )
                grad_duration = (
                    event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time
                )

                check_g[channel_num] = SimpleNamespace()
                check_g[channel_num].idx = idx
                check_g[channel_num].start = (grad_start, event.first)
                check_g[channel_num].stop = (grad_duration, event.last)

                if hasattr(event, 'id'):
                    grad_id = event.id
                else:
                    grad_id, _ = register_grad_event(self, event)

                new_block[idx] = grad_id
                duration = max(duration, grad_duration)

                if trace_enabled() and hasattr(event, 'trace'):
                    setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
            elif event.type == 'trap':
                channel_num = ['x', 'y', 'z'].index(event.channel)
                idx = 2 + channel_num

                if new_block[idx] != 0:
                    raise ValueError(f'Multiple {event.channel.upper()} gradient events were specified in set_block')

                if hasattr(event, 'id'):
                    trap_id = event.id
                else:
                    trap_id = register_grad_event(self, event)

                new_block[idx] = trap_id
                duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
            elif event.type == 'adc':
                if new_block[5] != 0:
                    raise ValueError('Multiple ADC events were specified in set_block')

                if hasattr(event, 'id'):
                    adc_id = event.id
                else:
                    adc_id = register_adc_event(self, event)

                new_block[5] = adc_id
                duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)

                if trace_enabled() and hasattr(event, 'trace'):
                    self.block_trace[block_index].adc = event.trace
            elif event.type == 'delay':
                duration = max(duration, event.delay)
            elif event.type in ['output', 'trigger']:
                if hasattr(event, 'id'):
                    event_id = event.id
                else:
                    event_id = register_control_event(self, event)

                ext = {'type': self.get_extension_type_ID('TRIGGERS'), 'ref': event_id}
                extensions.append(ext)
                duration = max(duration, event.delay + event.duration)
            elif event.type in ['labelset', 'labelinc']:
                if hasattr(event, 'id'):
                    label_id = event.id
                else:
                    label_id = register_label_event(self, event)

                ext = {
                    'type': self.get_extension_type_ID(event.type.upper()),
                    'ref': label_id,
                }
                extensions.append(ext)
        else:
            # Floating point number given as delay
            duration = max(duration, event)

    # =========
    # ADD EXTENSIONS
    # =========
    if len(extensions) > 0:
        """
        Add extensions now... but it's tricky actually we need to check whether the exactly the same list of extensions
        already exists, otherwise we have to create a new one... ooops, we have a potential problem with the key
        mapping then... The trick is that we rely on the sorting of the extension IDs and then we can always find the
        last one in the list by setting the reference to the next to 0 and then proceed with the other elements.
        """
        sort_idx = np.argsort([e['ref'] for e in extensions])
        extensions = np.take(extensions, sort_idx)
        all_found = True
        extension_id = 0
        for i in range(len(extensions)):
            data = (extensions[i]['type'], extensions[i]['ref'], extension_id)
            extension_id, found = self.extensions_library.find(data)
            all_found = all_found and found
            if not found:
                break

        if not all_found:
            # Add the list
            extension_id = 0
            for i in range(len(extensions)):
                data = (extensions[i]['type'], extensions[i]['ref'], extension_id)
                extension_id, found = self.extensions_library.find(data)
                if not found:
                    self.extensions_library.insert(extension_id, data)

        # Now we add the ID
        new_block[6] = extension_id

    # =========
    # PERFORM GRADIENT CHECKS
    # =========
    grad_check(self, block_index, check_g, duration)

    self.block_events[block_index] = new_block
    self.block_durations[block_index] = float(duration)


def get_block(self, block_index: int) -> SimpleNamespace:
    """
    Returns PyPulseq block at `block_index` position in `self.block_events`.

    The block is created from the sequence data with all events and shapes decompressed.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.block_events`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block at 'block_index' position in `self.block_events`.

    Raises
    ------
    ValueError
        If a trigger event of an unsupported control type is encountered.
        If a label object of an unknown extension ID is encountered.
    """
    # Check if block exists in the block cache. If so, return that
    if self.use_block_cache and block_index in self.block_cache:
        return self.block_cache[block_index]

    block = SimpleNamespace()
    attrs = ['block_duration', 'rf', 'gx', 'gy', 'gz', 'adc', 'label']
    values = [None] * len(attrs)
    for att, val in zip(attrs, values):
        setattr(block, att, val)
    event_ind = self.block_events[block_index]

    if event_ind[0] > 0:  # Delay
        delay = SimpleNamespace()
        delay.type = 'delay'
        delay.delay = self.delay_library.data[event_ind[0]][0]
        block.delay = delay

    if event_ind[1] > 0:  # RF
        if event_ind[1] in self.rf_library.type:
            block.rf = self.rf_from_lib_data(self.rf_library.data[event_ind[1]], self.rf_library.type[event_ind[1]])
        else:
            block.rf = self.rf_from_lib_data(self.rf_library.data[event_ind[1]], 'u')  # Undefined type/use

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(len(grad_channels)):
        if event_ind[2 + i] > 0:
            grad, compressed = SimpleNamespace(), SimpleNamespace()
            grad_type = self.grad_library.type[event_ind[2 + i]]
            lib_data = self.grad_library.data[event_ind[2 + i]]
            grad.type = 'trap' if grad_type == 't' else 'grad'
            grad.channel = grad_channels[i][1]
            if grad.type == 'grad':
                amplitude = lib_data[0]
                shape_id = lib_data[1]
                time_id = lib_data[2]
                delay = lib_data[3]
                shape_data = self.shape_library.data[shape_id]
                compressed.num_samples = shape_data[0]
                compressed.data = shape_data[1:]
                g = decompress_shape(compressed)
                grad.waveform = amplitude * g

                if time_id == 0:
                    grad.tt = (np.arange(1, len(g) + 1) - 0.5) * self.grad_raster_time
                    t_end = len(g) * self.grad_raster_time
                else:
                    t_shape_data = self.shape_library.data[time_id]
                    compressed.num_samples = t_shape_data[0]
                    compressed.data = t_shape_data[1:]
                    grad.tt = decompress_shape(compressed) * self.grad_raster_time

                    assert len(grad.waveform) == len(grad.tt)
                    t_end = grad.tt[-1]

                grad.shape_id = shape_id
                grad.time_id = time_id
                grad.delay = delay
                grad.shape_dur = t_end
                if len(lib_data) > 5:
                    grad.first = lib_data[4]
                    grad.last = lib_data[5]
            else:
                grad.amplitude = lib_data[0]
                grad.rise_time = lib_data[1]
                grad.flat_time = lib_data[2]
                grad.fall_time = lib_data[3]
                grad.delay = lib_data[4]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time

            setattr(block, grad_channels[i], grad)

    # ADC
    if event_ind[5] > 0:
        lib_data = self.adc_library.data[event_ind[5]]

        adc = SimpleNamespace()
        (
            adc.num_samples,
            adc.dwell,
            adc.delay,
            adc.freq_offset,
            adc.phase_offset,
            adc.dead_time,
        ) = [lib_data[x] for x in range(6)]
        adc.num_samples = int(adc.num_samples)
        adc.type = 'adc'
        block.adc = adc

    # Triggers
    if event_ind[6] > 0:
        # We have extensions - triggers, labels, etc.
        next_ext_id = event_ind[6]
        while next_ext_id != 0:
            ext_data = self.extensions_library.data[next_ext_id]
            # Format: ext_type, ext_id, next_ext_id
            ext_type = self.get_extension_type_string(ext_data[0])

            if ext_type == 'TRIGGERS':
                trigger_types = ['output', 'trigger']
                data = self.trigger_library.data[ext_data[1]]
                trigger = SimpleNamespace()
                trigger.type = trigger_types[int(data[0]) - 1]
                if data[0] == 1:
                    trigger_channels = ['osc0', 'osc1', 'ext1']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                elif data[0] == 2:
                    trigger_channels = ['physio1', 'physio2']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                else:
                    raise ValueError('Unsupported trigger event type')

                trigger.delay = data[2]
                trigger.duration = data[3]
                # Allow for multiple triggers per block
                if hasattr(block, 'trigger'):
                    block.trigger[len(block.trigger)] = trigger
                else:
                    block.trigger = {0: trigger}
            elif ext_type in ['LABELSET', 'LABELINC']:
                label = SimpleNamespace()
                label.type = ext_type.lower()
                supported_labels = get_supported_labels()
                if ext_type == 'LABELSET':
                    data = self.label_set_library.data[ext_data[1]]
                else:
                    data = self.label_inc_library.data[ext_data[1]]

                label.label = supported_labels[int(data[1] - 1)]
                label.value = data[0]
                # Allow for multiple labels per block
                if block.label is not None:
                    block.label[len(block.label)] = label
                else:
                    block.label = {0: label}
            else:
                raise RuntimeError(f'Unknown extension ID {ext_data[0]}')

            next_ext_id = ext_data[2]

    # Reverse the order of labels, because extensions are saved as a reversed linked list
    if block.label is not None:
        block.label = dict(enumerate(reversed(block.label.values())))

    block.block_duration = self.block_durations[block_index]

    # Enter block into the block cache
    if self.use_block_cache:
        self.block_cache[block_index] = block

    return block


def register_adc_event(self, event: EventLibrary) -> int:
    """

    Parameters
    ----------
    event : SimpleNamespace
        ADC event to be registered.

    Returns
    -------
    int
        ID of registered ADC event.
    """
    data = (
        event.num_samples,
        event.dwell,
        event.delay,
        event.freq_offset,
        event.phase_offset,
        event.dead_time,
    )
    adc_id, found = self.adc_library.find_or_insert(new_data=data)

    # Clear block cache because ADC was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return adc_id


def register_control_event(self, event: SimpleNamespace) -> int:
    """

    Parameters
    ----------
    event : SimpleNamespace
        Control event to be registered.

    Returns
    -------
    int
        ID of registered control event.
    """
    event_type = ['output', 'trigger'].index(event.type)
    if event_type == 0:
        # Trigger codes supported by the Siemens interpreter as of May 2019
        event_channel = ['osc0', 'osc1', 'ext1'].index(event.channel)
    elif event_type == 1:
        # Trigger codes supported by the Siemens interpreter as of June 2019
        event_channel = ['physio1', 'physio2'].index(event.channel)
    else:
        raise ValueError('Unsupported control event type')

    data = (event_type + 1, event_channel + 1, event.delay, event.duration)
    control_id, found = self.trigger_library.find_or_insert(new_data=data)

    # Clear block cache because trigger was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return control_id


def register_grad_event(self, event: SimpleNamespace) -> Union[int, Tuple[int, List[int]]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        Gradient event to be registered.

    Returns
    -------
    int, [int, ...]
        For gradient events: ID of registered gradient event, list of shape IDs
    int
        For trapezoid gradient events: ID of registered gradient event
    """
    may_exist = True
    any_changed = False
    if event.type == 'grad':
        amplitude = np.abs(event.waveform).max()
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1  # Workaround for np.sign(0) = 0

        if hasattr(event, 'shape_IDs'):
            shape_IDs = event.shape_IDs
        else:
            shape_IDs = [0, 0]
            if amplitude != 0:
                g = event.waveform / amplitude
            else:
                g = event.waveform
            c_shape = compress_shape(g)
            s_data = np.concatenate(([c_shape.num_samples], c_shape.data))
            shape_IDs[0], found = self.shape_library.find_or_insert(s_data)
            may_exist = may_exist & found
            any_changed = any_changed or found

            # Check whether tt == np.arange(len(event.tt)) * self.grad_raster_time + 0.5
            tt_regular = (np.floor(event.tt / self.grad_raster_time) == np.arange(len(event.tt))).all()

            if not tt_regular:
                c_time = compress_shape(event.tt / self.grad_raster_time)
                t_data = np.concatenate(([c_time.num_samples], c_time.data))
                shape_IDs[1], found = self.shape_library.find_or_insert(t_data)
                may_exist = may_exist & found
                any_changed = any_changed or found

        data = (amplitude, *shape_IDs, event.delay, event.first, event.last)
    elif event.type == 'trap':
        data = (
            event.amplitude,
            event.rise_time,
            event.flat_time,
            event.fall_time,
            event.delay,
        )
    else:
        raise ValueError('Unknown gradient type passed to register_grad_event()')

    if may_exist:
        grad_id, found = self.grad_library.find_or_insert(new_data=data, data_type=event.type[0])
        any_changed = any_changed or found
    else:
        grad_id = self.grad_library.insert(0, data, event.type[0])

    # Clear block cache because grad event or shapes were overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and any_changed:
        self.block_cache.clear()

    if event.type == 'grad':
        return grad_id, shape_IDs
    elif event.type == 'trap':
        return grad_id


def register_label_event(self, event: SimpleNamespace) -> int:
    """
    Parameters
    ----------
    event : SimpleNamespace
        ID of label event to be registered.

    Returns
    -------
    int
        ID of registered label event.
    """
    label_id = get_supported_labels().index(event.label) + 1
    data = (event.value, label_id)
    if event.type == 'labelset':
        label_id, found = self.label_set_library.find_or_insert(new_data=data)
    elif event.type == 'labelinc':
        label_id, found = self.label_inc_library.find_or_insert(new_data=data)
    else:
        raise ValueError('Unsupported label type passed to register_label_event()')

    # Clear block cache because label event was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return label_id


def register_rf_event(self, event: SimpleNamespace) -> Tuple[int, List[int]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        RF event to be registered.

    Returns
    -------
    int, [int, ...]
        ID of registered RF event, list of shape IDs
    """
    mag = np.abs(event.signal)
    amplitude = np.max(mag)
    mag /= amplitude
    # Following line of code is a workaround for numpy's divide functions returning NaN when mathematical
    # edge cases are encountered (eg. divide by 0)
    mag[np.isnan(mag)] = 0
    phase = np.angle(event.signal)
    phase[phase < 0] += 2 * np.pi
    phase /= 2 * np.pi
    may_exist = True

    if hasattr(event, 'shape_IDs'):
        shape_IDs = event.shape_IDs
    else:
        shape_IDs = [0, 0, 0]

        mag_shape = compress_shape(mag)
        data = np.concatenate(([mag_shape.num_samples], mag_shape.data))
        shape_IDs[0], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        phase_shape = compress_shape(phase)
        data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
        shape_IDs[1], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        t_regular = (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all()

        if t_regular:
            shape_IDs[2] = 0
        else:
            time_shape = compress_shape(event.t / self.rf_raster_time)
            data = [time_shape.num_samples, *time_shape.data]
            shape_IDs[2], found = self.shape_library.find_or_insert(data)
            may_exist = may_exist & found

    use = 'u'  # Undefined
    if hasattr(event, 'use'):
        if event.use in [
            'excitation',
            'refocusing',
            'inversion',
            'saturation',
            'preparation',
        ]:
            use = event.use[0]
        else:
            use = 'u'

    data = (amplitude, *shape_IDs, event.delay, event.freq_offset, event.phase_offset)

    if may_exist:
        rf_id, found = self.rf_library.find_or_insert(new_data=data, data_type=use)

        # Clear block cache because RF event was overwritten
        # TODO: Could find only the blocks that are affected by the changes
        if self.use_block_cache and found:
            self.block_cache.clear()
    else:
        rf_id = self.rf_library.insert(key_id=0, new_data=data, data_type=use)

    return rf_id, shape_IDs
