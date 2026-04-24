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
import numpy.typing as npt
from qrisp.core import QuantumVariable, Qubit, x, cx
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_W_state
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from qrisp.environments import control
from collections.abc import Sequence
from typing import Any, Callable, TYPE_CHECKING, Union

def foqcs_prep_heisenberg_1D(
    prep_qv: QuantumVariable | Sequence[Qubit],
    L: int,
    g: dict,
    J: dict
) -> None:
    """
    Parameters
    ----------
    prep_qv : QuantumVariable | Sequence[Qubit]
        Ancillae qubits register for PREP.

    L : int
        Number of system qubits in the Heisenberg chain.

    g : dict (length 3)
        Dictionary of local field coefficients. {"X": gx, "Y": gy, "Z": gz}

    J : dict (length 3)
        Dictionary of coupling coefficients for the Heisenberg interaction. {"X": Jx, "Y": Jy, "Z": Jz}
    
    Returns
    -------
    ???

    Raises
    ------
    ValueError
        If ``g`` or ``J`` has length .

    """
    # Check that received g and J dictionaries exactly contain the expected entries.
    req_keys = {"X", "Y", "Z"}

    g_keys = set(g.keys())
    J_keys = set(J.keys())

    if g_keys != req_keys:
        missing = req_keys - g_keys
        extra = g_keys - req_keys
        raise ValueError(
            f"g must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )

    if J_keys != req_keys:
        missing = req_keys - J_keys
        extra = J_keys - req_keys
        raise ValueError(
            f"J must contain exactly keys {sorted(req_keys)}. "
            f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
        )
    
    extra_anc = 6 # Depends on the method and can be potentially decreased
    if len(prep_qv) != L * 2 + extra_anc:
        raise ValueError(
            f"Number of received ancillae qubits must be exactly "
            f"{L * 2 + extra_anc}, but received {len(prep_qv)}"
        )

    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")
    #for q, i in enumerate(req_keys):
    #    _g[q] = np.sqrt(g[i] * L)
    #    _J[q] = np.sqrt(J[i] * (L - 1))

    _g[0] = np.sqrt(g["X"] * L)
    _g[1] = np.sqrt(g["Y"] * L * -1j)
    _g[2] = np.sqrt(g["Z"] * L)
    _J[0] = np.sqrt(J["X"] * (L - 1))
    _J[1] = np.sqrt(J["Y"] * -(L - 1))
    _J[2] = np.sqrt(J["Z"] * (L - 1))

    # Correction for XZ = -iY
    #_J[1] = 1j * _J[1]
    #_g[1] = (1 - 1j) * _g[1] / np.sqrt(2)
    # Normalization for state preparation
    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    # SUBPREP
    unbalanced_W_state(prep_qv[:extra_anc], np.block([_g, _J]))

    # PREP
    fh1 = extra_anc                # First qubit first half
    lh1 = extra_anc + L - 1        # Last qubit first half
    fh2 = extra_anc + L            # First qubit second half
    lh2 = extra_anc + (L * 2) - 1  # Last qubit second half
    # X gx: Balanced(1) on first half
    with control([prep_qv[0]]):
        x(prep_qv[lh1])
        dicke_state(prep_qv[fh1:lh1 + 1], 1)
    # Y gy: Double(1)
    with control([prep_qv[1]]):
        x(prep_qv[lh1])
        dicke_state(prep_qv[fh1:lh1 + 1], 1)
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:])
    # Z gz: Balanced(1) on second half
    with control([prep_qv[2]]):
        x(prep_qv[lh2])
        dicke_state(prep_qv[fh2:], 1)
    # X Jx: Balanced 2NN on first half
    with control([prep_qv[3]]):
        x(prep_qv[lh1 - 1])
        dicke_state(prep_qv[fh1:lh1], 1)
        _cx_ladder(prep_qv[fh1:lh1 + 1], 1)
    # Y Jy: Double 2NN
    with control([prep_qv[4]]):
        x(prep_qv[lh1 - 1])
        dicke_state(prep_qv[fh1:lh1], 1)
        _cx_ladder(prep_qv[fh1:lh1 + 1], 1)
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:])
    # Z Jz: Balanced 2NN on second half
    with control([prep_qv[5]]):
        x(prep_qv[lh2 - 1])
        dicke_state(prep_qv[fh2:lh2], 1)
        _cx_ladder(prep_qv[fh2:], 1)

    return prep_qv

###################################
############# Helpers #############
###################################

def _cx_ladder(qv: QuantumVariable | Sequence[Qubit], k: int = 1) -> None:
    """
    TODO: DOC
    k - how many qubits to drag the control over. (1: neighbour, 2: 1 qubit over the neighbour, etc.)
    """
    n = len(qv)
    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])
