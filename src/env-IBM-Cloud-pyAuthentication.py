from qiskit_ibm_runtime import QiskitRuntimeService

# Execute esta célula UMA VEZ para salvar suas credenciais
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="xxxxxxxxxxxxxxxxxxxxxxxx",  # Cole o token que você gerou
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/c32bc959b9024b8da06b6bbc92713040:7cfb645a-ab7c-404b-ab40-5d0dc662a74a::"
)

print("Conta configurada com sucesso!")
