from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes

from qiskit_machine_learning.algorithms import (
    QSVC,
    VQC
)

from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel
)

from qiskit_algorithms.optimizers import COBYLA

from qiskit.primitives import StatevectorSampler


def build_qsvc(
    n_qubits=4
):

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2
    )

    kernel = FidelityQuantumKernel(
        feature_map=feature_map
    )

    return QSVC(
        quantum_kernel=kernel,
        probability=True
    )


def build_vqc(
    n_qubits=4
):

    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=2
    )

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=4
    )

    optimizer = COBYLA(
        maxiter=600
    )

    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=StatevectorSampler()
    )