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
from qrisp import Qubit, QuantumVariable, x, dicke_state, cx, unbalanced_W_state
from collections.abc import Sequence

def _cx_ladder(qv: QuantumVariable | Sequence[Qubit], k: int = 1) -> None:
    """
    TODO: DOC
    k - how many qubits to drag the control over. (1: neighbour, 2: 1 qubit over the neighbour, etc.)
    """
    n = len(qv)
    for i in reversed(range(0, n - k)):
        cx(qv[i], qv[i + k])

def test_dicke_state_balanced():
    expected = QuantumVariable(3)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "001": amp,
        "010": amp,
        "100": amp
    }, method="qswitch")
    qv = QuantumVariable(3)

    # Prepare Balanced Dicke state.
    x(qv[2])
    dicke_state(qv, 1)

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_unbalanced():
    expected = QuantumVariable(3)
    expected.init_state({
        "001": 0.25,
        "010": 0.375,
        "100": 0.375
    }, method="qswitch")
    qv = QuantumVariable(3)

    amps = np.array([0.25, 0.375, 0.375], dtype=complex)

    # Prepare Unbalanced Dicke state.
    unbalanced_W_state(qv, amps, True)

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_balanced_2kN():
    n = 5
    k = 2
    expected = QuantumVariable(n)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "00101": amp,
        "01010": amp,
        "10100": amp
    }, method="qswitch")
    qv = QuantumVariable(n)

    # Prepare Balanced Dicke state on all qubits except last k.
    x(qv[n - k - 1])
    dicke_state(qv[:n - k], 1)

    # Make it 2kN by using CNOT ladder
    _cx_ladder(qv, k)

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_unbalanced_2kN():
    n = 6
    k = 2
    expected = QuantumVariable(n)
    expected.init_state({
        "000101": 0.5,
        "001010": 0.2,
        "010100": 0.3,
        "101000": 0.1
    }, method="qswitch")
    qv = QuantumVariable(n)

    amps = np.array([0.1, 0.3, 0.2, 0.5], dtype=complex)

    # Prepare Unbalanced Dicke state on all qubits except last k.
    unbalanced_W_state(qv[:n - k], amps)

    # Make it 2kN by using CNOT ladder
    _cx_ladder(qv, k)

    print(f"\nDepth = {qv.qs.depth()}")
    print(f"CNOT count = {qv.qs.cnot_count()}")

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_balanced_double():
    expected = QuantumVariable(6)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "001001": amp,
        "010010": amp,
        "100100": amp
    }, method="qswitch")
    qv = QuantumVariable(6)

    # Prepare Balanced Dicke state on first half of qubits.
    x(qv[2])
    dicke_state(qv[:3], 1)

    # Make it double by dragging down control to the second half.
    cx(qv[:3], qv[3:])

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_unbalanced_double():
    expected = QuantumVariable(6)
    expected.init_state({
        "001001": 0.25,
        "010010": 0.375,
        "100100": 0.375
    }, method="qswitch")
    qv = QuantumVariable(6)

    amps = np.array([0.375, 0.375, 0.25], dtype=complex)

    # Prepare Unbalanced Dicke state on first half of qubits.
    unbalanced_W_state(qv[:3], amps)

    # Make it double by dragging down control to the second half.
    cx(qv[:3], qv[3:])

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_balanced_double_2kN():
    n = 6
    k = 1
    expected = QuantumVariable(n)
    expected.init_state({
        "011011": 0.5,
        "110110": 0.5,
    }, method="qswitch")
    qv = QuantumVariable(n)

    # Prepare NN Dicke state on first half of qubits.
    x(qv[n // 2 - k - 1])
    dicke_state(qv[:n // 2 - k], 1)
    _cx_ladder(qv[:n // 2], k)

    # Make it double by dragging down control to the second half.
    cx(qv[:n // 2], qv[n // 2:])

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_unbalanced_double_2kN():
    n = 6
    k = 1
    expected = QuantumVariable(n)
    expected.init_state({
        "011011": 0.6,
        "110110": 0.4,
    }, method="qswitch")
    qv = QuantumVariable(n)

    amps = np.array([0.4, 0.6], dtype=complex)

    # Prepare Unbalanced Dicke state on first half of qubits.
    unbalanced_W_state(qv[:n // 2 - k], amps)
    _cx_ladder(qv[:n // 2], k)

    # Make it double by dragging down control to the second half.
    cx(qv[:n // 2], qv[n // 2:])

    print(f"Prepared: {qv}")
    print(f"Expected: {expected}")
    assert qv.get_measurement() == expected.get_measurement()
