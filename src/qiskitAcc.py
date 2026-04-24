from qiskit_ibm_runtime import QiskitRuntimeService
# Salve sua conta uma vez
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # Substitua pela sua chave
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/c32bc959b9024b8da06b6bbc92713040:7cfb645a-ab7c-404b-ab40-5d0dc662a74a::" #CRN
)
