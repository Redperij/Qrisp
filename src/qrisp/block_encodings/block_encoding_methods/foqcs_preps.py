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
from qrisp import QuantumVariable, x, cx
from typing import Any, Callable, TYPE_CHECKING, Union

def foqcs_prep_heisenberg_1D(
    coeffs: npt.NDArray[np.number],
    unitaries: list[Callable[..., Any]],
) -> QuantumVariable:
    
    raise NotImplementedError


###################################
############# Helpers #############
###################################

# Special Dicke states
def dicke1_state(qv, coeffs=None, nn=False):
    """
    TODO DOC
    """
    n = len(qv)

    # Check coeffs, if None or same -> Check nn.
    # Check nn, if False -> dicke_state() with k = 1

    # If have coeffs -> flag unbalanced

    # unbalanced: coeffs
    # -> NOT n^th qubit -> calc theta for all coeffs -> calculate phases -> put in gamma gate ladder -> put in the phase gates.
    # balanced nn: no coeffs + nn
    # -> NOT (n - 1)^th qubit -> call dicke_state(qv(n - 1), 1) -> call CNOT ladder from top to bottom from last qubit to first.
    # unbalanced nn: coeffs + nn
    # -> NOT (n - 1)^th qubit -> calc theta for coeffs except last? -> put in gamma gate ladder -> put in the phase gates -> call CNOT ladder from top to bottom from last qubit to first
    pass

def cx_ladder(qv, k = 1):
    """
    TODO: DOC
    len - how many qubits to drag the control over. (1: neighbour, 2: 1 qubit over the neighbour, etc.)
    """
    n = len(qv)
    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])

def ecx(qv, split_index):
    
    n =  len(qv)
    cx(qv[:split_index], qv[split_index:])
