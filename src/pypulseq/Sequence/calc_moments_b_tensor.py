from typing import Tuple

import numpy as np
from scipy.interpolate import PPoly


def calc_moments_b_tensor(
    self, calcB: bool = True, calcm1: bool = False, calcm2: bool = False, calcm3: bool = False, Ndummy: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute B-tensor and up to 3rd order moments from sequence gradients.

    Parameters
    ----------
    calcB : bool, default=True
        Compute B-tensor.
    calcm1 : bool, default=False
        Whether to compute m1.
    calcm2 : bool, default=False
        Whether to compute m2.
    calcm3 : bool, default=False
        Whether to compute m3.
    Ndummy : int, default=0
        Number of dummy shots to skip.

    Returns
    -------
    B : np.ndarray
        B tensor with shape `(R, 3, 3)`.
    m1 : np.ndarray
        M1 moment with shape `(R,3)`.
    m2 : np.ndarray
        M2 moment with shape `(R,3)`.
    m3 : np.ndarray
        M3 moment with shape `(R,3)`.
    """
    _, _, t_exc, t_ref, _ = self.calculate_kspace()
    gw_pp = self.get_gradients()
    R = len(t_exc)
    gx_l, gy_l, gz_l = gw_pp

    # timings
    t_echo = np.array([2 * t_ref[i] - t_exc[i] for i in range(R)])
    tSeq = np.column_stack([t_exc, t_ref, t_echo]).ravel()
    tn = np.unique(np.concatenate([gx_l.x, gy_l.x, gz_l.x, tSeq]))

    # refit piecewise polynomials onto common tn grid
    gx = PPoly(fill_pp_coefs(gx_l, tn), tn)
    gy = PPoly(fill_pp_coefs(gy_l, tn), tn)
    gz = PPoly(fill_pp_coefs(gz_l, tn), tn)

    # split per repetition, skipping dummies
    gx_pieces = split_pp(gx, t_exc[Ndummy:])
    gy_pieces = split_pp(gy, t_exc[Ndummy:])
    gz_pieces = split_pp(gz, t_exc[Ndummy:])

    B = np.zeros((R, 3, 3))
    m1 = np.zeros((R, 3))
    m2 = np.zeros((R, 3))
    m3 = np.zeros((R, 3))

    def poly_convolve_and_integrate(p1: PPoly, p2: PPoly):
        nb = p1.c.shape[1]
        c_conv = [np.convolve(p1.c[:, i], p2.c[:, i]) for i in range(nb)]
        c_conv = np.stack(c_conv, axis=1)
        return PPoly(c_conv, p1.x).antiderivative()

    for i in range(R):
        te = tSeq[3 * (i + Ndummy) + 2]
        if calcB:
            qp = [gx_pieces[i].antiderivative(), gy_pieces[i].antiderivative(), gz_pieces[i].antiderivative()]
            for q in qp:
                q.c *= 2 * np.pi
            for m in range(3):
                for n in range(3):
                    bb = poly_convolve_and_integrate(qp[m], qp[n])
                    B[i, m, n] = bb(te)

        def calc_moment(order: int):
            arr = np.zeros(3)
            for axis, gpp in enumerate((gx_pieces[i], gy_pieces[i], gz_pieces[i])):  # noqa: B023
                # Multiply polynomial by (t - t0)^order, integrate
                t0 = gpp.x[0]
                # Build polynomial (t - t0)^order coefficients for each interval
                out = PPoly(gpp.c.copy(), gpp.x)
                for _ in range(order):
                    # multiply by (t - t0)
                    new_c = []
                    for seg in range(out.c.shape[1]):
                        seg_coefs = out.c[:, seg]
                        # polynomial multiply by (t - t0): shift + combine
                        deg = len(seg_coefs)
                        conv = np.zeros(deg + 1)
                        conv[:-1] += seg_coefs
                        conv[1:] -= seg_coefs * t0
                        new_c.append(conv)
                    out = PPoly(np.stack(new_c, 1), out.x)
                arr[axis] = out.antiderivative()(te) - out.antiderivative()(t0)  # noqa: B023
            return arr

        if calcm1:
            m1[i] = calc_moment(1)
        if calcm2:
            m2[i] = calc_moment(2)
        if calcm3:
            m3[i] = calc_moment(3)

    return B, m1, m2, m3


def slookup(what: np.ndarray, where: np.ndarray) -> np.ndarray:
    idx = np.zeros_like(what, dtype=int)
    wb = 0
    for i, val in enumerate(what):
        found = np.where(where[wb:] == val)[0]
        if found.size:
            idx[i] = wb + found[0]
            wb = idx[i]
    return idx


def fill_pp_coefs(pp: PPoly, xn: np.ndarray) -> np.ndarray:
    nb = len(xn) - 1
    order = pp.c.shape[0]
    idx1 = slookup(xn[:-1], pp.x[:-1])
    new_coefs = np.zeros((order, nb))
    for i in range(nb):
        ii = idx1[i]
        if ii > 0:
            new_coefs[:, i] = pp.c[:, ii]
        elif i > 0:
            for k in range(order):
                for l in range(k + 1):
                    new_coefs[order - 1 - l, i] += (
                        new_coefs[order - 1 - k, i - 1] * comb(k, l) * (xn[i] - xn[i - 1]) ** (k - l)
                    )
    return new_coefs


def comb(n, k):
    from math import comb as _comb

    return _comb(n, k)


def split_pp(pp: PPoly, t_exc: np.ndarray) -> list[PPoly]:
    """Split a global PPoly into blocks based on excitation times."""
    pieces = []
    x = pp.x
    for start, end in zip(t_exc[:-1], t_exc[1:]):
        mask = (x >= start) & (x <= end)
        if mask.sum():
            xs = x[mask]
            cs = pp.c[:, mask[:-1]]
            pieces.append(PPoly(cs, xs))
    # last block
    mask = x >= t_exc[-1]
    xs = x[mask]
    cs = pp.c[:, mask[:-1]]
    pieces.append(PPoly(cs, xs))
    return pieces
