"""Phase parsing and ZX tensors used by the database and evaluator."""
from fractions import Fraction
import numpy as np

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pi_string_to_fraction(s: str) -> Fraction:
    s = s.replace(" ", "")  # Remove spaces
    if 'π' not in s:
        return Fraction(s)  # No pi, treat as regular number

    if s == 'π':
        return Fraction(1)

    if s.startswith('π/'):
        return Fraction(1, int(s.split('/')[1]))

    if s.endswith('π'):
        return Fraction(int(s[:-1]))

    if 'π/' in s:
        num = s.split('π/')[0]
        denom = s.split('π/')[1]
        return Fraction(int(num), int(denom))

    raise ValueError(f"Unrecognized format: {s}")


def z_spider_tensor(num_legs, phase_radians=0.0):
    shape = (2,) * num_legs
    T = np.zeros(shape, dtype=complex)
    T[(0,)*num_legs] = 1.0
    T[(1,)*num_legs] += np.exp(1j * phase_radians)
    return T


def spider_tensor(num_legs, phase=0.0, basis='Z'):
    """
    phase: the same 'phase' as PyZX's internal: p where PyZX uses phase = pi*p.
           If you pass g.phase(v) (which is p), then multiply by pi below.
    """
    phase_radians = np.pi * phase

    if basis == 'Z':
        return z_spider_tensor(num_legs, phase_radians)
    
    T = z_spider_tensor(num_legs, phase_radians)
    for leg in range(num_legs):
        T = np.tensordot(H, T, axes=([1], [leg]))
        # result has H axis 0 inserted at front; move it to the original leg position
        T = np.moveaxis(T, 0, leg)
    return T


def hadamard_tensor():
    """Hadamard edge node."""
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
