from copy import copy, deepcopy
from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq import eps
from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.utils.cumsum import cumsum
from pypulseq.utils.tracing import trace, trace_enabled


def add_gradients(
    grads: List[SimpleNamespace],
    max_grad: int = 0,
    max_slew: int = 0,
    system: Union[Opts, None] = None,
) -> SimpleNamespace:
    """
    Returns the superposition of several gradients.

    Parameters
    ----------
    grads : [SimpleNamespace, ...]
        Gradient events.
    system : Opts, default=Opts()
        System limits.
    max_grad : float, default=0
        Maximum gradient amplitude.
    max_slew : float, default=0
        Maximum slew rate.

    Returns
    -------
    grad : SimpleNamespace
        Superimposition of gradient events from `grads`.
    """
    if system is None:
        system = Opts.default

    if max_grad <= 0:
        max_grad = system.max_grad
    if max_slew <= 0:
        max_slew = system.max_slew

    if len(grads) == 0:
        raise ValueError('No gradients specified')
    if len(grads) == 1:
        # Trapezoids only require a shallow copy
        if grads[0].type == 'trap':
            grad = copy(grads[0])
        else:
            grad = deepcopy(grads[0])

        if trace_enabled():
            grad.trace = trace()
        return grad

    # First gradient defines channel
    channel = grads[0].channel

    # Check if we have a set of traps with the same timing
    if (
        all(g.type == 'trap' for g in grads)
        and all(g.rise_time == grads[0].rise_time for g in grads)
        and all(g.flat_time == grads[0].flat_time for g in grads)
        and all(g.fall_time == grads[0].fall_time for g in grads)
        and all(g.delay == grads[0].delay for g in grads)
    ):
        grad = make_trapezoid(
            grads[0].channel,
            amplitude=sum(g.amplitude for g in grads) + eps,
            rise_time=grads[0].rise_time,
            flat_time=grads[0].flat_time,
            fall_time=grads[0].fall_time,
            delay=grads[0].delay,
            system=system,
        )
        if trace_enabled():
            grad.trace = trace()
        return grad

    # Find out the general delay of all gradients and other statistics
    delays, firsts, lasts, durs, is_trap, is_arb, is_osa = [], [], [], [], [], [], []
    for ii in range(len(grads)):
        if grads[ii].channel != channel:
            raise ValueError('Cannot add gradients on different channels.')

        delays.append(grads[ii].delay)
        durs.append(calc_duration(grads[ii]))
        is_trap.append(grads[ii].type == 'trap')
        if is_trap[-1]:
            is_arb.append(False)
            is_osa.append(False)
            firsts.append(0.0)
            lasts.append(0.0)
        else:
            tt_rast = grads[ii].tt / system.grad_raster_time
            is_arb.append(np.all(np.abs(tt_rast + 0.5 - np.arange(1, len(tt_rast) + 1))) < eps)
            is_osa.append(np.all(np.abs(tt_rast - 0.5 * np.arange(1, len(tt_rast) + 1)) < eps))
            firsts.append(grads[ii].first)
            lasts.append(grads[ii].last)

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    is_etrap = np.logical_and.reduce((np.logical_not(is_trap), np.logical_not(is_arb), np.logical_not(is_osa)))
    if np.all(np.logical_or(is_trap, is_etrap)):
        # Keep shapes still rather simple
        times = []
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == 'trap':
                times.extend(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
            else:
                times.extend(g.delay + g.tt)

        times = np.unique(times)
        dt = times[1:] - times[:-1]
        ieps = np.flatnonzero(dt < eps)
        if np.any(ieps):
            dtx = np.array([times[0], *dt])
            dtx[ieps] = dtx[ieps] + dtx[ieps + 1]  # Assumes that no more than two too similar values can occur
            dtx = np.delete(dtx, ieps + 1)
            times = np.cumsum(dtx)

        amplitudes = np.zeros_like(times)
        for ii in range(len(grads)):
            g = grads[ii]
            if g.type == 'trap':
                if g.flat_time > 0:  # Trapezoid or triangle
                    tt = list(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
                    waveform = [0, g.amplitude, g.amplitude, 0]
                else:
                    tt = list(cumsum(g.delay, g.rise_time, g.fall_time))
                    waveform = [0, g.amplitude, 0]
            else:
                tt = g.delay + g.tt
                waveform = g.waveform

            # Fix rounding for the first and last time points
            i_min = np.argmin(np.abs(tt[0] - times))
            t_min = (np.abs(tt[0] - times))[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = np.argmin(np.abs(tt[-1] - times))
            t_min = (np.abs(tt[-1] - times))[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if abs(waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            amplitudes += np.interp(xp=tt, fp=waveform, x=times, left=0, right=0)

        grad = make_extended_trapezoid(channel=channel, amplitudes=amplitudes, times=times, system=system)

        if trace_enabled():
            grad.trace = trace()

        return grad

    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(firsts), np.array(lasts)
    common_delay = np.min(delays)
    total_duration = np.max(durs)
    durs = np.array(durs)

    # Convert everything to a regularly-sampled waveform
    waveforms = {}
    max_length = 0

    if np.any(is_osa):
        target_raster = 0.5 * system.grad_raster_time
    else:
        target_raster = system.grad_raster_time

    for ii in range(len(grads)):
        g = grads[ii]
        if g.type == 'grad':
            if is_arb[ii] or is_osa[ii]:
                if np.any(is_osa) and is_arb[ii]:  # Porting MATLAB here, maybe a bit ugly
                    # Interpolate missing samples
                    idx = np.arange(0, len(g.waveform) - 0.5 + eps, 0.5)
                    wf = g.waveform
                    interp_waveform = 0.5 * (wf[np.floor(idx).astype(int)] + wf[np.ceil(idx).astype(int)])
                    waveforms[ii] = interp_waveform
                else:
                    waveforms[ii] = g.waveform
            else:
                waveforms[ii] = points_to_waveform(
                    amplitudes=g.waveform,
                    times=g.tt,
                    grad_raster_time=target_raster,
                )
        elif g.type == 'trap':
            if g.flat_time > 0:  # Triangle or trapezoid
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.flat_time,
                        g.delay - common_delay + g.rise_time + g.flat_time + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, g.amplitude, 0])
            else:
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, 0])
            waveforms[ii] = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=target_raster,
            )
        else:
            raise ValueError('Unknown gradient type')

        if g.delay - common_delay > 0:
            # Stop for numpy.arange is not g.delay - common_delay - system.grad_raster_time like in Matlab
            # so as to include the endpoint
            waveforms[ii] = np.concatenate(
                (np.zeros(round((g.delay - common_delay) / system.grad_raster_time)), waveforms[ii])
            )

        num_points = len(waveforms[ii])
        max_length = max(num_points, max_length)

    w = np.zeros(max_length)
    for ii in range(len(grads)):
        wt = np.zeros(max_length)
        wt[0 : len(waveforms[ii])] = waveforms[ii]
        w += wt

    grad = make_arbitrary_grad(
        channel=channel,
        waveform=w,
        system=system,
        max_slew=max_slew,
        max_grad=max_grad,
        delay=common_delay,
        oversampling=np.any(is_osa),
        first=np.sum(firsts[delays == common_delay]),
        last=np.sum(lasts[durs == total_duration]),
    )
    # Fix the first and the last values
    # First is defined by the sum of firsts with the minimal delay (common_delay)
    # Last is defined by the sum of lasts with the maximum duration (total_duration == durs.max())
    grad.first = np.sum(firsts[np.array(delays) == common_delay])
    grad.last = np.sum(lasts[durs == durs.max()])

    if trace_enabled():
        grad.trace = trace()

    return grad
