# test_conexao_final.py
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()
backends = service.backends()
print("✅ Backends disponíveis:")
for backend in backends:
    print(f"  • {backend.name}")