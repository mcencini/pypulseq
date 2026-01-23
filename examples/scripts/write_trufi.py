"""
PyPulseq TrueFISP (bSSFP) demo, event structure and splitting as in MATLAB writeTrufi.m.
Requires: pypulseq >=1.2, with split_gradient_at/add_gradients support.
"""

import numpy as np
import pypulseq as pp
from pypulseq import split_gradient_at, add_gradients, calc_duration, make_trapezoid, make_delay

def main(plot: bool = False, write_seq: bool = False, seq_filename: str = 'trufi_pypulseq.seq'):
    # ---- Sequence and System ----
    fov = 220e-3
    Nx = 256
    Ny = 256
    thick = 4e-3
    sys = pp.Opts(
        max_grad=30,
        grad_unit='mT/m',
        max_slew=140,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
    )
    seq = pp.Sequence(sys)

    # ---- Sequence Hard Parameters ----
    adc_dur_us = 2560
    adc_dur = adc_dur_us * 1e-6
    alpha = 40 * np.pi/180
    rf_dur = 600e-6
    rf_apo = 0.5
    rf_bwt = 1.5

    # ---- RF/Grad/ADC events ----
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=alpha,
        duration=rf_dur,
        slice_thickness=thick,
        apodization=rf_apo,
        time_bw_product=rf_bwt,
        system=sys,
        return_gz=True,
        use="excitation",
    )

    deltak = 1.0 / fov
    gx = make_trapezoid(channel="x", flat_area=Nx*deltak, flat_time=adc_dur, system=sys)
    adc = pp.make_adc(Nx, duration=gx.flat_time, delay=gx.rise_time, system=sys)
    gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, system=sys)
    phase_areas = (np.arange(Ny) - Ny/2) * deltak

    # ---- Event splitting for optimal timing ----
    # a) Gz split: at end of RF (like 'mr.splitGradientAt(gz,mr.calcDuration(rf))')
    rf_len = calc_duration(rf)
    gz_parts = split_gradient_at(gz, rf_len, system=sys)
    gz1 = add_gradients([gz_reph, gz_parts[0]], system=sys)
    gz1.delay = calc_duration(gz_reph)
    rf, _ = pp.align(right=[rf, gz1])   # Ensures RF right edge is aligned with end of gz1

    gz_parts[1].delay = 0
    gz_reph.delay = calc_duration(gz_parts[1])
    gz2 = add_gradients([gz_parts[1], gz_reph], system=sys)

    # b) Gx split: at end of ADC
    adc_len = calc_duration(adc)
    
    # Next line ensures splitting is at a legal raster time boundary
    gx_split_time = np.ceil(adc_len/sys.grad_raster_time) * sys.grad_raster_time
    gx_parts = split_gradient_at(gx, gx_split_time, system=sys)
    gx_parts[0].delay = calc_duration(gx_pre)
    gx1 = add_gradients([gx_pre, gx_parts[0]], system=sys)
    adc.delay = calc_duration(gx_pre)
    gx_parts[1].delay = 0
    gx_pre.delay = calc_duration(gx_parts[1])
    gx2 = add_gradients([gx_parts[1], gx_pre], system=sys)
    gx_pre.delay = 0

    # ---- PE duration and further alignment ----
    pe_dur = calc_duration(gx2)
    gz1.delay = max(calc_duration(gx2) - rf.delay + rf.ringdown_time, 0)
    rf.delay = rf.delay + gz1.delay  # adjust RF delay

    # Calculate TR/TE
    TR = calc_duration(gz1) + calc_duration(gx1)
    TE = TR / 2

    # ---- Alpha/2 preparation pulse ----
    # make RF 0.5*alpha
    rf05 = pp.make_sinc_pulse(
        flip_angle=alpha/2,
        duration=rf_dur,
        slice_thickness=thick,
        apodization=rf_apo,
        time_bw_product=rf_bwt,
        system=sys,
        use="excitation",
    )
    # Block 1: prep
    seq.add_block(rf05, gz1)
    seq.add_block(pp.make_label(type='SET', label='ONCE', value=1))
    
    # Block 2: center delay (approximated as in MATLAB)
    prep_delay = make_delay(np.round((TR/2 - calc_duration(gz1))/sys.grad_raster_time)*sys.grad_raster_time)
    gx1_blank, _, _ = pp.make_extended_trapezoid_area(
        channel='x', 
        grad_start=0.0,
        grad_end=gx2.first,
        area=-gx2.area,
        system=sys
    )
    gy_pre2 = make_trapezoid('y', area=phase_areas[-1], duration=pe_dur, system=sys)
    seq.add_block(*pp.align(left=[prep_delay, gz2, gy_pre2], right=[gx1_blank]))
    seq.add_block(pp.make_label(type='SET', label='ONCE', value=0))  # label end of prep

    # ---- Main Phase Encode Loop ----
    gy_pre2 = None  # Will become previous PE for next loop
    for i in range(Ny):
        rf.phase_offset = np.pi * (i % 2)
        adc.phase_offset = np.pi * (i % 2)
        gy_pre1 = make_trapezoid(channel='y', area=-(phase_areas[i-1] if i > 0 else 0), duration=pe_dur, system=sys)
        gy_pre2 = make_trapezoid(channel='y', area=phase_areas[i], duration=pe_dur, system=sys)
        seq.add_block(rf, gz1, gy_pre1, gx2)
        seq.add_block(gx1, gy_pre2, gz2, adc)

    # ---- Exit/last blocks: ONCE label ----
    seq.add_block(gx2)
    seq.add_block(pp.make_label(type='SET', label='ONCE', value=2))

    # ---- Info ----
    seq.set_definition('FOV', [fov, fov, thick])
    seq.set_definition('Name', 'trufi')

    # ---- Timing check ----
    ok, error_report = seq.check_timing()
    print('Timing check passed successfully' if ok else f'Timing check failed:\n{error_report}')

    # ---- Output ----
    if write_seq:
        seq.write(seq_filename)
        print(f'Sequence written to {seq_filename}')
    if plot:
        seq.plot()

    return seq

if __name__ == '__main__':
    main(plot=True, write_seq=True)