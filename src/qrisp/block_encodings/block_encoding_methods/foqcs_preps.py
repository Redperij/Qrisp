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
from qrisp.core import QuantumVariable, Qubit
from qrisp.core.gate_application_functions import x, cx, ry
from qrisp.alg_primitives.unbalanced_w_state import unbalanced_W_state
from qrisp.alg_primitives.dicke_state_prep import dicke_state
from qrisp.environments import control
from collections.abc import Sequence
from functools import partial
from typing import Any

def foqcs_prep_heisenberg_1D(
    prep_qv: QuantumVariable | Sequence[Qubit],
    L: int,
    g: dict,
    J: dict,
    conjugate: bool = False
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

    conjugate : bool = False
        Indicates whether the prep is PREP_R or PREP_L.
        In case of PREP_L, the conjugates of g and J are used.
        The default is False, which indicates it is PREP_R.

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

    _g = np.zeros((3,), dtype="complex")
    _J = np.zeros((3,), dtype="complex")

    _g[0] = np.sqrt(g["X"] * L)
    _g[1] = np.sqrt(g["Y"] * L * -1j)
    _g[2] = np.sqrt(g["Z"] * L)
    _J[0] = np.sqrt(J["X"] * (L - 1))
    _J[1] = np.sqrt(J["Y"] * -(L - 1))
    _J[2] = np.sqrt(J["Z"] * (L - 1))

    # Normalization for state preparation
    norm = np.linalg.norm(np.block([_g, _J]))
    _g /= norm
    _J /= norm

    if conjugate:
        _g = np.conj(_g)
        _J = np.conj(_J)

    # SUBPREP
    unbalanced_W_state(prep_qv[:extra_anc], np.block([_g, _J]), extra_anc)
    
    # PREP
    fh1 = extra_anc                # First qubit first half
    lh1 = extra_anc + L - 1        # Last qubit first half
    fh2 = extra_anc + L            # First qubit second half
    lh2 = extra_anc + (L * 2) - 1  # Last qubit second half

    # 3 specific CNOTs
    cx(prep_qv[0], prep_qv[lh1]) # from gx
    cx(prep_qv[1], prep_qv[lh1]) # from gy
    cx(prep_qv[2], prep_qv[lh2]) # from gz
    # 2 parallel Gamma gates
    theta = 2 * np.acos(np.sqrt(1 / (L - 1 + 1)))
    _gamma_gate(prep_qv[lh1 - 1], prep_qv[lh1], theta)
    _gamma_gate(prep_qv[lh2 - 1], prep_qv[lh2], theta)
    # 3 specific CNOTs
    cx(prep_qv[3], prep_qv[lh1 - 1]) # from Jx
    cx(prep_qv[4], prep_qv[lh1 - 1]) # from Jy
    cx(prep_qv[5], prep_qv[lh2 - 1]) # from Jz
    # 2 parallel delta gates
    theta_coeffs = []
    for i in range(1, L - 1):
        theta_coeffs.append(2 * np.acos(np.sqrt(1 / (i + 1))))
    _delta_gate(prep_qv[fh1:lh1], theta_coeffs)
    _delta_gate(prep_qv[fh2:lh2], theta_coeffs)
    # One spec CNOT
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # controlled CNOT ladder
    with control(prep_qv[4]): # from Jy
        _cx_ladder(prep_qv[fh1:lh1 + 1], L)
    # One spec CNOT
    cx(prep_qv[3], prep_qv[4]) # from Jx to Jy
    # controlled CNOT ladder
    with control(prep_qv[5]): # from Jz
        _cx_ladder(prep_qv[fh2:], L)
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy
    # controlled Element-wise CNOT
    with control(prep_qv[4]): # from Jy
        cx(prep_qv[fh1:lh1 + 1], prep_qv[fh2:])
    # One spec CNOT
    cx(prep_qv[1], prep_qv[4]) # from gy to Jy

def foqcs_prep_placeholder( qv: QuantumVariable, coeffs: list, some_param: float ) -> None:
    """
    TO-DO DOC
    """
    print("I am doing nothing")

###################################
############# Helpers #############
###################################
    
def get_foqcs_lcu_prep_num_of_ancillae(prep: partial, num_ops: int = 1) -> int:
    r"""
        Constructs a BlockEncoding using the Fast One-Qubit-Controlled Select Linear Combination of Unitaries (FOQCS-LCU) protocol.

        Parameters
        ----------
        prep : partial
            Partially initialised FOQCS-LCU PREP method.
        
        num_ops : int
            Number of operand qubits (L argument for FOQCS-LCU PREP routines).
            The default is 1.

        Returns
        -------
        int
            An integer with number of ancillae required by the received FOQCS-LCU PREP method
    """
    if prep.func == foqcs_prep_heisenberg_1D:
        return num_ops * 2 + 6
    elif prep.func == foqcs_prep_placeholder:
        return 0
    else:
        raise ValueError(f"Received unknown FOQCS-LCU PREP routine: {prep}")

def _cx_ladder(qv: QuantumVariable | Sequence[Qubit], n: int, k: int = 1) -> None:

    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])

def _gamma_gate(qv_rot: Any, qv_x: Any, theta: float):
    with control(qv_x):
        ry(theta, qv_rot)
    cx(qv_rot, qv_x)

def _delta_gate(qv: QuantumVariable | Sequence[Qubit], theta_coeffs):
    if len(theta_coeffs) == 1:
        _gamma_gate(qv[0], qv[1], theta_coeffs[0])
    else:
        _gamma_gate(qv[-2], qv[-1], theta_coeffs[-1])
        _delta_gate(qv[:-1], theta_coeffs[:-1])
