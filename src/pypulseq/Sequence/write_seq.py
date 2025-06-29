import hashlib
from pathlib import Path
from typing import Union

import numpy as np

from pypulseq import __version__
from pypulseq.supported_labels_rf_use import get_supported_labels, get_supported_rf_uses

version_major, version_minor, version_revision = __version__.split('.')[:3]


def write(self, file_name: Union[str, Path], create_signature, remove_duplicates=True) -> Union[str, None]:
    """
    Write the sequence data to the given filename using the open file format for MR sequences.

    See also `pypulseq.Sequence.read_seq.read()`.

    Parameters
    ----------
    file_name : str or Path
        File name of `.seq` file to be written to disk.
    create_signature : bool
    remove_duplicates : bool
        Before writing, remove and remap events that would be duplicates after
        the rounding done during writing

    Returns
    -------
    md5 or None : If create_signature is True, it returns the written .seq file's signature as a string,
    otherwise it returns None. Note that, if remove_duplicates is True, signature belongs to the
    deduplicated sequences signature, and not the Sequence that is stored in the Sequence object.

    Raises
    ------
    RuntimeError
        If an unsupported definition is encountered.
    """
    # `>.0f` for decimals.
    # `>g` to truncate insignificant zeros.
    file_name = Path(file_name)
    if file_name.suffix != '.seq':
        # Append .seq suffix
        file_name = file_name.with_suffix(file_name.suffix + '.seq')

    # If removing duplicates, make a copy of the sequence with the duplicate
    # events removed.
    if remove_duplicates:
        self = self.remove_duplicates()

    with open(file_name, 'w') as output_file:
        output_file.write('# Pulseq sequence file\n')
        output_file.write('# Created by PyPulseq\n\n')

        output_file.write('[VERSION]\n')
        output_file.write(f'major {self.version_major}\n')
        output_file.write(f'minor {self.version_minor}\n')
        output_file.write(f'revision {self.version_revision}\n')
        output_file.write('\n')

        if len(self.definitions) != 0:
            output_file.write('[DEFINITIONS]\n')
            keys = sorted(self.definitions.keys())
            values = [self.definitions[k] for k in keys]
            for block_counter in range(len(keys)):
                output_file.write(f'{keys[block_counter]} ')
                if isinstance(values[block_counter], str):
                    output_file.write(values[block_counter] + ' ')
                elif isinstance(values[block_counter], (int, float)):
                    output_file.write(f'{values[block_counter]:0.9g} ')
                elif isinstance(values[block_counter], (list, tuple, np.ndarray)):  # e.g. [FOV_x, FOV_y, FOV_z]
                    for i in range(len(values[block_counter])):
                        if isinstance(values[block_counter][i], (int, float)):
                            output_file.write(f'{values[block_counter][i]:0.9g} ')
                        else:
                            output_file.write(f'{values[block_counter][i]} ')
                else:
                    raise RuntimeError('Unsupported definition')
                output_file.write('\n')
            output_file.write('\n')

        output_file.write('# Format of blocks:\n')
        output_file.write('# NUM DUR RF  GX  GY  GZ  ADC  EXT\n')
        output_file.write('[BLOCKS]\n')
        id_format_width = '{:' + str(len(str(len(self.block_events)))) + 'd}'
        id_format_str = id_format_width + ' {:3d} {:3d} {:3d} {:3d} {:3d} {:2d} {:2d}\n'
        for block_counter in self.block_events:
            block_duration = self.block_durations[block_counter] / self.block_duration_raster
            block_duration_rounded = round(block_duration)

            if abs(block_duration_rounded - block_duration) >= 1e-6:
                raise ValueError('Inconsistent block duration after rounding')

            s = id_format_str.format(
                *(
                    block_counter,
                    block_duration_rounded,
                    *self.block_events[block_counter][1:],
                )
            )
            output_file.write(s)
        output_file.write('\n')

        if len(self.rf_library.data) != 0:
            output_file.write('# Format of RF events:\n')
            output_file.write('# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n')
            output_file.write('# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n')
            output_file.write(f'# Field "use" is the initial of: {" ".join(get_supported_rf_uses()).strip()}\n')
            output_file.write('[RF]\n')
            id_format_str = (
                '{:.0f} {:12g} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:g} {:g} {:s}\n'  # Refer lines 20-21
            )
            for k in self.rf_library.data:
                lib_data1 = self.rf_library.data[k][0:4]
                lib_data2 = self.rf_library.data[k][6:10]
                center = self.rf_library.data[k][4] * 1e6  # us
                delay = round(self.rf_library.data[k][5] / self.rf_raster_time) * self.rf_raster_time * 1e6
                s = id_format_str.format(k, *lib_data1, center, delay, *lib_data2, self.rf_library.type[k])
                output_file.write(s)
            output_file.write('\n')

        grad_lib_values = np.array(list(self.grad_library.type.values()))
        arb_grad_mask = grad_lib_values == 'g' if self.grad_library.type else False
        trap_grad_mask = grad_lib_values == 't' if self.grad_library.type else False

        if np.any(arb_grad_mask):
            output_file.write('# Format of arbitrary gradients:\n')
            output_file.write(
                '#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n'
            )
            output_file.write('# id amplitude first last amp_shape_id time_shape_id delay\n')
            output_file.write('# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n')
            output_file.write('[GRADIENTS]\n')
            id_format_str = '{:.0f} {:12g} {:12g} {:12g} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
            keys = np.array(list(self.grad_library.data.keys()))
            for k in keys[arb_grad_mask]:
                s = id_format_str.format(
                    k,
                    *self.grad_library.data[k][:5],
                    round(self.grad_library.data[k][5] * 1e6),
                )
                output_file.write(s)
            output_file.write('\n')

        if np.any(trap_grad_mask):
            output_file.write('# Format of trapezoid gradients:\n')
            output_file.write('# id amplitude rise flat fall delay\n')
            output_file.write('# ..      Hz/m   us   us   us    us\n')
            output_file.write('[TRAP]\n')
            keys = np.array(list(self.grad_library.data.keys()))
            id_format_str = '{:2.0f} {:12g} {:3.0f} {:4.0f} {:3.0f} {:3.0f}\n'
            for k in keys[trap_grad_mask]:
                data = np.copy(self.grad_library.data[k])  # Make a copy to leave the original untouched
                data[1:] = np.round(1e6 * data[1:])
                """
                Python & Numpy always round to nearest even value - inconsistent with MATLAB Pulseq's .seq files.
                [1] https://stackoverflow.com/questions/29671945/format-string-rounding-inconsistent
                [2] https://stackoverflow.com/questions/50374779/how-to-avoid-incorrect-rounding-with-numpy-round
                """
                s = id_format_str.format(k, *data)
                output_file.write(s)
            output_file.write('\n')

        if len(self.adc_library.data) != 0:
            output_file.write('# Format of ADC events:\n')
            output_file.write('# id num dwell delay freqPPM phasePPM freq phase phase_id\n')
            output_file.write('# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n')
            output_file.write('[ADC]\n')
            id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:.0f}\n'  # Refer lines 20-21
            for k in self.adc_library.data:
                data = np.multiply(self.adc_library.data[k][0:8], [1, 1e9, 1e6, 1, 1, 1, 1, 1])
                s = id_format_str.format(k, *data)
                output_file.write(s)
            output_file.write('\n')

        if len(self.extensions_library.data) != 0:
            output_file.write('# Format of extension lists:\n')
            output_file.write('# id type ref next_id\n')
            output_file.write('# next_id of 0 terminates the list\n')
            output_file.write('# Extension list is followed by extension specifications\n')
            output_file.write('[EXTENSIONS]\n')
            id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
            for k in self.extensions_library.data:
                s = id_format_str.format(k, *np.round(self.extensions_library.data[k]))
                output_file.write(s)
            output_file.write('\n')

        if len(self.trigger_library.data) != 0:
            output_file.write('# Extension specification for digital output and input triggers:\n')
            output_file.write('# id type channel delay (us) duration (us)\n')
            output_file.write(f'extension TRIGGERS {self.get_extension_type_ID("TRIGGERS")}\n')
            id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
            for k in self.trigger_library.data:
                s = id_format_str.format(k, *np.round(self.trigger_library.data[k] * np.array([1, 1, 1e6, 1e6])))
                output_file.write(s)
            output_file.write('\n')

        if len(self.label_set_library.data) != 0:
            labels = get_supported_labels()

            output_file.write('# Extension specification for setting labels:\n')
            output_file.write('# id set labelstring\n')
            tid = self.get_extension_type_ID('LABELSET')
            output_file.write(f'extension LABELSET {tid}\n')
            id_format_str = '{:.0f} {:.0f} {}\n'  # Refer lines 20-21
            for k in self.label_set_library.data:
                value = self.label_set_library.data[k][0]
                label_id = labels[int(self.label_set_library.data[k][1]) - 1]  # label_id is +1 in add_block()
                s = id_format_str.format(k, value, label_id)
                output_file.write(s)
            output_file.write('\n')

        if len(self.label_inc_library.data) != 0:
            labels = get_supported_labels()

            output_file.write('# Extension specification for setting labels:\n')
            output_file.write('# id set labelstring\n')
            tid = self.get_extension_type_ID('LABELINC')
            output_file.write(f'extension LABELINC {tid}\n')
            id_format_str = '{:.0f} {:.0f} {}\n'  # See comment at the beginning of this method definition
            for k in self.label_inc_library.data:
                value = self.label_inc_library.data[k][0]
                label_id = labels[self.label_inc_library.data[k][1] - 1]  # label_id is +1 in add_block()
                s = id_format_str.format(k, value, label_id)
                output_file.write(s)
            output_file.write('\n')

        if len(self.soft_delay_library.data) != 0:
            output_file.write('# Extension specification for soft delays:\n')
            output_file.write('# id num offset factor hint\n')
            output_file.write('# ..  ..     us     ..   ..\n')

            tid = self.get_extension_type_ID('DELAYS')
            output_file.write(f'extension DELAYS {tid}\n')
            id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {}\n'

            for k in self.soft_delay_library.data:
                data = self.soft_delay_library.data[k]
                s = id_format_str.format(k, data[0], np.round(data[1] * 1e6), data[2], data[3])
                output_file.write(s)
            output_file.write('\n')

        if len(self.rf_shim_library.data) != 0:
            output_file.write('# Extension specification for RF shimming:\n')
            output_file.write('# id num_chan factor magn_c1 phase_c1 magn_c2 phase_c2 ...\n')
            output_file.write(f'extension RF_SHIMS {self.get_extension_type_ID("RF_SHIMS")}\n')

            for k in self.rf_shim_library.data:
                shim_vector_length = len(self.rf_shim_library.data[k])
                id_format_str = '{:.0f} {:0f}' + shim_vector_length * '{:12g}' + '\n'
                s = id_format_str.format(k, int(0.5 * shim_vector_length), *self.rf_shim_library.data[k])
                output_file.write(s)
            output_file.write('\n')

        if len(self.rotation_library.data) != 0:
            output_file.write('# Extension specification for rotation events:\n')
            output_file.write('# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n')
            output_file.write(f'extension ROTATIONS {self.get_extension_type_ID("ROTATIONS")}\n')
            id_format_str = '{:.0f} {:12g} {:12g} {:12g} {:12g}\n'  # Refer lines 20-21
            for k in self.rotation_library.data:
                s = id_format_str.format(k, *self.rotation_library.data[k])
                output_file.write(s)
            output_file.write('\n')

        if len(self.shape_library.data) != 0:
            output_file.write('# Sequence Shapes\n')
            output_file.write('[SHAPES]\n\n')
            for k in self.shape_library.data:
                shape_data = self.shape_library.data[k]
                s = 'shape_id {:.0f}\n'.format(k)
                output_file.write(s)
                s = 'num_samples {:.0f}\n'.format(shape_data[0])
                output_file.write(s)
                s = ('{:.9g}\n' * len(shape_data[1:])).format(*shape_data[1:])
                output_file.write(s)
                output_file.write('\n')

    if create_signature:  # Sign the file
        # Calculate digest
        with open(file_name, 'r') as output_file:
            buffer = output_file.read()

            md5 = hashlib.md5(buffer.encode('utf-8')).hexdigest()

        # Write signature
        with open(file_name, 'a') as output_file:
            output_file.write('\n[SIGNATURE]\n')
            output_file.write(
                '# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n'
            )
            output_file.write(
                '# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n'
            )
            output_file.write(
                '# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for '
                'recalculating/verification)\n'
            )
            output_file.write('Type md5\n')
            output_file.write(f'Hash {md5}\n')

        return md5
