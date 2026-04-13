"""
********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************
"""

import numpy as np
from qrisp import QuantumVariable, x, xxyy, p

def unbalanced_W_state(qv, amplitudes):
    r"""
    Prepare the generalized W state

        |\psi> = \sum_{i=0}^{n-1} a_i |e_i>

    on `qv`, where |e_i> is the computational basis state with a single 1
    at position i.

    This is the Qrisp translation of the Silq implementation:
    - build a heap-style tree of subtree weights |a_i|^2,
    - recurse top-down to distribute one excitation,
    - then imprint the target phases on the leaves.

    Parameters
    ----------
    qv : QuantumVariable
        Freshly allocated quantum register in |0...0>.
    amplitudes : array_like
        Complex target amplitudes, one per qubit.

    Raises
    ------
    ValueError
        If len(amplitudes) != qv.size or the amplitude vector is zero.

    Example
    --------
    ::
        import numpy as np
        from qrisp import QuantumVariable
        a = np.array([1j, 2, 3, 4])
        qv = QuantumVariable(4)
        unbalanced_W_state(qv, a)
        print(qv.qs.statevector())
    """
    n = len(qv)
    a = np.asarray(amplitudes, dtype=complex)

    if len(a) != n:
        raise ValueError(
            f"Length of amplitudes ({len(a)}) must match qv.size ({n})."
        )

    norm = np.linalg.norm(a)
    if norm < 1e-15:
        raise ValueError("Amplitude vector must be non-zero.")
    a = a / norm

    # Heap-style tree of subtree probability weights.
    # We overallocate with 4*n slots, matching the Silq implementation.
    t = np.zeros(4 * n, dtype=float)

    def build(i, l, u):
        if l + 1 == u:
            t[i] = abs(a[l]) ** 2
        else:
            m = (l + u) // 2
            build(2 * i + 1, l, m)
            build(2 * i + 2, m, u)
            t[i] = t[2 * i + 1] + t[2 * i + 2]

    def rec(i, l, u):
        # Invariant:
        # qv[l:u] contains exactly one excitation, initially at qv[l].
        if l + 1 == u or t[i] < 1e-15:
            return

        m = (l + u) // 2
        alpha0 = t[2 * i + 1]
        alpha1 = t[2 * i + 2]

        # alpha0 + alpha1 = t[i] > 0 here
        theta = 2 * np.arccos(np.sqrt(alpha0 / (alpha0 + alpha1)))

        # Split between the left representative qv[l]
        # and the right representative qv[m].
        xxyy(theta, np.pi / 2, qv[l], qv[m])

        rec(2 * i + 1, l, m)
        rec(2 * i + 2, m, u)

    build(0, 0, n)

    # Start from |10...0>
    x(qv[0])

    # Prepare the magnitudes
    rec(0, 0, n)

    # Imprint the target phases on the unique excited qubit
    for i in range(n):
        if abs(a[i]) > 1e-15:
            p(np.angle(a[i]), qv[i])
