# Célula de teste para verificar a instalação do Aer
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import SamplerV2 as Sampler
    print("✅ qiskit-aer instalado com sucesso!")
    print("✅ Módulos AerSimulator e SamplerV2 importados.")
    
    # Teste básico de funcionamento
    backend = AerSimulator()
    print(f"✅ Backend do simulador criado: {backend}")
    
except ImportError as e:
    print(f"❌ Erro na importação: {e}")