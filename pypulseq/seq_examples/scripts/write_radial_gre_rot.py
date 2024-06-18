import math
import sys

import numpy as np

import pypulseq as pp


def _rotation_matrix(theta):
    
    # R[0] = (R[0][0], R[0][1], R[0][2])
    R0 = np.stack((np.cos(theta), -np.sin(theta), np.zeros_like(theta)), axis=1) # (nangles, 3)
    
    # R[1] = (R[1][0], R[1][1], R[1][2])
    R1 = np.stack((np.sin(theta), np.cos(theta), np.zeros_like(theta)), axis=1) # (nangles, 3)
    
    # R[2] = (R[2][0], R[2][1], R[2][2])
    R2 = np.stack((np.zeros_like(theta), np.zeros_like(theta), np.ones_like(theta)), axis=1) # (nangles, 3)

    return np.stack((R0, R1, R2), axis=1)  # (nangles, 3, 3)

def main(plot: bool, write_seq: bool, seq_filename: str = "gre_pypulseq.seq", use_rot_ext: bool = False):
    # ======
    # SETUP
    # ======
    # Create a new sequence object
    seq = pp.Sequence()
    fov = 256e-3 # In-plane FoV
    Nr = 256 # 1 mm iso in-plane resolution
    slab_thickness = 180e-3  # slice
    Nz = 150 # 1.2 mm slice thickness
    
    # RF specs
    alpha = 10  # flip angle
    rf_spoiling_inc = 117  # RF spoiling increment

    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )

    # ======
    # CREATE EVENTS
    # ======
    rf, gss, _ = pp.make_sinc_pulse(
        flip_angle=alpha * math.pi / 180,
        duration=3e-3,
        slice_thickness=slab_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=True,
    )
    gss_reph = pp.make_trapezoid(
        channel="z", area=-gss.area / 2, duration=1e-3, system=system
    )
    
    # Define other gradients and ADC events
    delta_kr, delta_kz = 1 / fov, 1 / slab_thickness
    gread = pp.make_trapezoid(
        channel="x", flat_area=Nr * delta_kr, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(
        num_samples=Nr, duration=gread.flat_time, delay=gread.rise_time, system=system
    )
    grpre = pp.make_trapezoid(
        channel="x", area=-gread.area / 2, duration=1e-3, system=system
    )
    grrew = pp.scale_grad(grad=grpre, scale=-1)
    grrew.id = seq.register_grad_event(grpre)
    gphase = pp.make_trapezoid(channel="z", area=-delta_kz * Nz, system=system)
    
    # Phase encoding plan and rotation
    pe_steps = ((np.arange(Nz)) - Nz / 2) / Nz * 2
    delta = np.pi / Nr  # Angular increment
    phi = np.arange(Nr) * delta
    
    if use_rot_ext:
        rotmat = _rotation_matrix(phi)

    # gradient spoiling
    gz_spoil = pp.make_trapezoid(channel="z", area=4 / slab_thickness, system=system)

    # Initialize RF phase and increment
    rf_phase = 0
    rf_inc = 0

    # ======
    # CONSTRUCT SEQUENCE
    # ======
    # Loop over phase encodes and define sequence blocks
    for z in range(Nz):
        
        # Pre-register PE events that repeat in the inner loop
        gzpre = pp.scale_grad(grad=gphase, scale=pe_steps[z])
        gzpre.id = seq.register_grad_event(gzpre)
        gzrew = pp.scale_grad(grad=gphase, scale=-pe_steps[z])
        gzrew.id = seq.register_grad_event(gzrew)
        
        for r in range(Nr):
        
            # Compute RF and ADC phase for spoiling and signal demodulation
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
        
            # Add slab-selective excitation
            seq.add_block(rf, gss)
            
            # Slab refocusing gradient
            seq.add_block(gss_reph)
        
            # Read-prewinding and phase encoding gradients
            seq.add_block(grpre, gzpre)
            
            # Add readout
            if use_rot_ext:
                seq.add_block(gread, adc, pp.make_rotation(rotmat[r]))
            else:
                seq.add_block(*pp.rotate(gread, adc, angle=phi[r], axis="z"))
                
            # Rewind
            seq.add_block(grrew, gzrew)
            
            # Spoil
            seq.add_block(gz_spoil)
            
            # Update RF phase and increment
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    # Check whether the timing of the sequence is correct
    ok, error_report = seq.check_timing()
    if ok:
        print("Timing check passed successfully")
    else:
        print("Timing check failed. Error listing follows:")
        [print(e) for e in error_report]

    # ======
    # VISUALIZATION
    # ======
    if plot:
        seq.plot()

    seq.calculate_kspace()

    # Very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within
    # slew-rate limits
    rep = seq.test_report()
    print(rep)

    # =========
    # WRITE .SEQ
    # =========
    if write_seq:
        # Prepare the sequence output for the scanner
        seq.set_definition(key="FOV", value=[fov, fov, slab_thickness / Nz])
        seq.set_definition(key="Name", value="radial_gre")

        seq.write(seq_filename)


if __name__ == "__main__":
    main(plot=False, write_seq=True, use_rot_ext=sys.argv[1])
