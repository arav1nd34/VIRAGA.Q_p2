from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.algorithms import QSVC

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel
)


def build_qsvc(n_qubits=4):

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2
    )

    kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    return QSVC(
        quantum_kernel=kernel
    )


def build_vqc(
    n_qubits=4
):
    return None