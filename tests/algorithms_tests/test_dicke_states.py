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
from qrisp import QuantumVariable, x, dicke_state, cx, multi_measurement, prepare

def test_dicke_state_balanced():
    expected = QuantumVariable(3)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "001": amp,
        "010": amp,
        "100": amp
    }, method="qswitch")

    qv = QuantumVariable(3)
    x(qv[2])
    dicke_state(qv, 1)

    print(f"Prepared: {qv}")
    print(f"Against: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_unbalanced():
    expected = QuantumVariable(3)
    expected.init_state({
        "001": 0.25,
        "010": 0.375,
        "100": 0.375
    }, method="qswitch")

    qv = QuantumVariable(3)

    prep_vec = np.zeros(8, dtype=complex)
    prep_vec[1] = 0.25
    prep_vec[2] = 0.375
    prep_vec[4] = 0.375

    prepare(qv, prep_vec, reversed=True, method="qswitch")

    # Variable, num of |1> qubits, coefficients distribution for unbalanced
    #dicke_state(qv, 1, coeffs) # Just make an unbalanced modification to the existing
    # dicke state preparation routine. If possible, of course.
    # Otherwise implement it under the same function as switch.

    print(f"Prepared: {qv}")
    print(f"Against: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_2NN():
    expected = QuantumVariable(3)
    expected.init_state({
        "011": 0.5,
        "110": 0.5
    }, method="qswitch")
    qv = QuantumVariable(3)
    # Variable, num of |1> qubits, coefficients distribution for unbalanced, Should qubits be NNs.
    #dicke_state(qv, 2, coeffs, true) # Allow unbalanced as well?

    print(f"Prepared: {qv}")
    print(f"Against: {expected}")
    assert qv.get_measurement() == expected.get_measurement()

def test_dicke_state_balanced_double():
    expected = QuantumVariable(6)
    qv = QuantumVariable(6)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "001001": amp,
        "010010": amp,
        "100100": amp
    }, method="qswitch")

    # Want to prepare it on a single variable.
    # For now just utilise the existing Dicke state preparation routine.
    qv1 = QuantumVariable(3)
    qv2 = QuantumVariable(3)
    x(qv1[2])
    dicke_state(qv1, 1)

    cx(qv1, qv2)

    prep = {
        "".join(bits): p
        for bits, p in multi_measurement([qv1, qv2]).items()
    }
    # Variable, num of |1> qubits
    #dicke_state_double(qv, 1)

    print(f"Prepared: {prep}")
    print(f"Against: {expected}")
    assert prep == expected.get_measurement()

def test_dicke_state_unbalanced_double():
    expected = QuantumVariable(6)
    qv = QuantumVariable(6)
    amp = 1 / np.sqrt(3)
    expected.init_state({
        "001001": 0.25,
        "010010": 0.375,
        "100100": 0.375
    }, method="qswitch")

    # Want to prepare it on a single variable.
    qv1 = QuantumVariable(3)
    qv2 = QuantumVariable(3)
    #x(qv1[2])
    # Variable, num of |1> qubits, coefficients distribution for unbalanced
    #dicke_state_double(qv, 1, coeffs)

    cx(qv1, qv2)

    prep = {
        "".join(bits): p
        for bits, p in multi_measurement([qv1, qv2]).items()
    }

    print(f"Prepared: {prep}")
    print(f"Against: {expected}")
    assert prep == expected.get_measurement()

def test_dicke_state_double_2NN():
    expected = QuantumVariable(6)
    qv = QuantumVariable(6)
    expected.init_state({
        "011011": 0.5,
        "110110": 0.5,
    }, method="qswitch")

    # Want to prepare it on a single variable.
    qv1 = QuantumVariable(3)
    qv2 = QuantumVariable(3)
    x(qv1[2])
    # Variable, num of |1> qubits, coefficients distribution for unbalanced, Should qubits be NNs.
    #dicke_state_double(qv, 2, true)

    cx(qv1, qv2)

    prep = {
        "".join(bits): p
        for bits, p in multi_measurement([qv1, qv2]).items()
    }

    print(f"Prepared: {prep}")
    print(f"Against: {expected}")
    assert prep == expected.get_measurement()
