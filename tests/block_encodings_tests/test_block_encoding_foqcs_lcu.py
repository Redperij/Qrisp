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
from qrisp.block_encodings import BlockEncoding
from qrisp import QuantumVariable, terminal_sampling, QuantumFloat, multi_measurement
from qrisp.block_encodings import foqcs_prep_heisenberg_1D, foqcs_prep_different
from functools import partial

def heisenberg_from_def(L: int, g: dict, J: dict):

    assert len(J) == 3, "J must be a list of length 3."
    assert len(g) == 3, "g must be list a of length 3."
    sigma_list = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]
    H = np.zeros((2**L, 2**L))
    for k, sigma in enumerate(sigma_list):
        sisj = np.kron(sigma, sigma)
        for i in range(L):
            if i < L-1:
                H = H + J[k] * np.kron(np.identity(2**i), np.kron(sisj, np.identity(2**(L-i-2))))
            H = H + g[k] * np.kron(np.identity(2**i), np.kron(sigma, np.identity(2**(L-i-1))))
    return H

def test_foqcs_lcu_prep():
    """
    TODO: DOC
    """

    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    #g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    #J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    d_state = QuantumVariable(L * 2 + 6)

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    foqcs_prep_heisenberg_1D(d_state, L, heis_g, heis_J)
    #sv = d_state.qs.statevector("function")
    # Take out the statevector from the compiled circuit
    qc = d_state.qs.compile()
    statev = qc.statevector_array()

    # Modify the original coefficients for the manual state building.
    g[0] = np.sqrt(g[0] * L)
    g[1] = np.sqrt(g[1] * L * -1j)
    g[2] = np.sqrt(g[2] * L)
    J[0] = np.sqrt(J[0] * (L - 1))
    J[1] = np.sqrt(J[1] * -(L - 1))
    J[2] = np.sqrt(J[2] * (L - 1))

    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # Build Dicke states manually, give them all unique weights from g[0] to J[2]: base, double, 2NN, 2NN double...
    dicke_1_norm = 1 / (np.sqrt(L))
    dicke_1 = np.zeros(2**L)
    for i in range(L):
        dicke_1[2**i] = dicke_1_norm

    dicke_double = np.zeros(4**L)
    for i in range(L):
        dicke_double[2**i + 2 ** (L + i)] = dicke_1_norm

    dicke_2NN_norm = 1 / (np.sqrt(L - 1))
    dicke_2NN = np.zeros((2**L,), dtype="complex")
    for i in range(L - 1):
        dicke_2NN[2**i + 2 ** (i + 1)] = dicke_2NN_norm

    dicke_2NN_double = np.zeros((4**L,), dtype="complex")
    for i in range(L - 1):
        dicke_2NN_double[
            2**i + 2 ** (i + 1) + 2 ** (i + L) + 2 ** (i + L + 1)
        ] = dicke_2NN_norm

    ref_state = np.zeros((2 ** (6 + 2 * L),), dtype="complex")
    zero_n = np.array([1] + [0] * (2**L - 1))

    # g[0]
    ref_state += g[0] * np.kron(
            [1 if i == 2 ** (6 - 1) else 0 for i in range(2**6)],
            np.kron(dicke_1, zero_n),
        )
    
    # g[1]
    ref_state += g[1] * np.kron(
        [1 if i == 2 ** (6 - 2) else 0 for i in range(2**6)], dicke_double
    )
    
    # g[2]
    ref_state += g[2] * np.kron(
        [1 if i == 2 ** (6 - 3) else 0 for i in range(2**6)],
        np.kron(zero_n, dicke_1),
    )

    # J[0]
    ref_state += J[0] * np.kron(
        [1 if i == 2 ** (6 - 4) else 0 for i in range(2**6)],
        np.kron(dicke_2NN, zero_n),
    )

    # J[1]
    ref_state += J[1] * np.kron(
        [1 if i == 2 ** (6 - 5) else 0 for i in range(2**6)], dicke_2NN_double
    )

    # J[2]
    ref_state += J[2] * np.kron(
        [1 if i == 2 ** (6 - 6) else 0 for i in range(2**6)],
        np.kron(zero_n, dicke_2NN),
    )

    #comp_arr = []
    #comp_arr_ref = []

    # print("State:")
    # for i in range(2**14):
    #     bits = format(i, "014b")
    #     amp = sv({d_state: bits})
    #     if np.isclose(amp, 0j, atol=1e-6) == False:
    #         print(bits, amp)
    #         comp_arr.append(amp)

    # print()
    # print("Ref state:")
    # init_ref = {}
    # q = 0
    # for i in ref_state:
    #     bits = format(q, "014b")
    #     if i != 0:
    #         init_ref[bits] = i
    #         print(bits, i)
    #         comp_arr_ref.append(i)
    #     q += 1

    # print(f"Array: {comp_arr}")
    # print(f"Array REF: {comp_arr_ref}")
    # print(f"Comparison of shortened arrays: {np.allclose(comp_arr, comp_arr_ref, atol=1e-06)}")

    # print(init_ref)

    # expected = QuantumVariable(14)
    # expected.init_state(init_ref, method="qiskit")

    # print(f"State:\n{d_state}")
    # print(f"Expected state:\n{expected}")

    # Add a zero qubit to the reference state to match the compiled circuit
    zero = np.array([1, 0], dtype=complex)
    ref_state_padded = np.kron(ref_state, zero)

    #Zero out entries that are close to zero
    #statev[np.isclose(statev, 0j, atol=1e-6)] = 0

    #for i in range(0, len(statev)):
    #    if statev[i] != 0:
    #        print(f"s[{i}] = {statev[i]}")
    #    if ref_state_padded[i] != 0:
    #        print(f"r[{i}] = {ref_state_padded[i]}")

    assert np.allclose(statev, ref_state_padded, atol=1e-06)

def test_block_encoding_from_foqcs_lcu_prep():
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    #d_state = foqcs_prep_heisenberg_1D(L, heis_g, heis_J)
    prep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
    )

    be = BlockEncoding.from_foqcs_lcu_prep(prep, L)

    qv = QuantumVariable(4)

    psi = np.random.uniform(-1, 1, 2 ** (L)) + 1j * np.random.uniform(
            -1, 1, 2 ** (L)
        )
    psi /= np.linalg.norm(psi)

    qv.init_state(psi, method="qswitch")

    def main(BE):
        operand = qv
        ancillas = BE.apply(operand)
        return operand, ancillas

    operand, ancillas = main(be)

    qc = operand.qs.compile()
    sv = qc.statevector_array()
    res_ops = []

    for i in range(0, 2 ** L):

        ind = i << (len(ancillas[0]) + 1)
        res_ops.append(sv[ind])

    res_dict = multi_measurement([operand] + ancillas)

    # print(f"Res dict = {res_dict}")
    H = heisenberg_from_def(L, g, J)

    ref_state = H @ psi

    print(f"Ref state = {ref_state}")
    print(f"Resulting operands = {res_ops}")

    # Filtering only zero ancillae entries
    zero_anc = "0" * len(ancillas[0])
    filtered = {
        key: value
        for key, value in res_dict.items()
        if key[1] == zero_anc
    }

    print(f"\n\nFiltered dict = {filtered}")

    # operand = QuantumVariable(4)
    # result = be.apply(operand)
    
    # print(f"Res = {result[0]}")
    # print(f"Op = {operand}")
    # print(operand.qs)
    # print(f"Op measurement = {operand.get_measurement()}")
    # print(f"Res measurement = {result[0].get_measurement()}")

    assert False



def test_xxyy_prep_conjugate_jasp():
    import numpy as np
    from qrisp import QuantumVariable, conjugate, xxyy
    from qrisp.jasp import terminal_sampling

    coeffs = np.ones(6, dtype=complex)

    def prep(qv):
        xxyy(coeffs[0], np.pi / 2, qv[0], qv[1])

    @terminal_sampling
    def main():
        qv = QuantumVariable(6)
        with conjugate(prep)(qv):
            pass
        #prep(qv)
        return qv

    main()

"""
def test_block_encoding_from_foqcs_lcu_prep_jax():
    # Initialize variables + their values
    L = 4
    g = np.array(np.random.uniform(-1, 1, 3), dtype="complex")
    J = np.array(np.random.uniform(-1, 1, 3), dtype="complex")

    # Fix coefficients for debugging
    g = np.array([0.80054361+0.j,  0.50905072+0.j, -0.89045545+0.j])
    J = np.array([0.98167489+0.j, -0.32435597+0.j,  0.42262456+0.j])

    # Normalize
    norm = np.linalg.norm(np.block([g, J]))
    g /= norm
    J /= norm

    # Construct dictionary input expected by foqcs_prep_heisenberg_1D()
    heis_g = {"X": g[0], "Y": g[1], "Z": g[2]}
    heis_J = {"X": J[0], "Y": J[1], "Z": J[2]}

    # Prep base state using qv + Do FOQCS-LCU magic using prep function
    #d_state = foqcs_prep_heisenberg_1D(L, heis_g, heis_J)
    prep = partial(
        foqcs_prep_heisenberg_1D,
        L=L,
        g=heis_g,
        J=heis_J,
    )

    be = BlockEncoding.from_foqcs_lcu_prep(prep, L)

    # xxyy gate doesn't work
    @terminal_sampling
    def main():
        return be.apply_rus(lambda: QuantumVariable(4))()
    
    result = main()

    print(result)

    assert False
"""