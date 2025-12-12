# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO - ATUALIZADA
# ===========================================================================

import numpy as np
import time
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import warnings
warnings.filterwarnings('ignore')

# Importações IBM Quantum atualizadas
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session  # ALTERAÇÃO: Session adicionada
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    IBM_QUANTUM_AVAILABLE = True
    print("✅ IBM Quantum Runtime disponível")
except ImportError as e:
    IBM_QUANTUM_AVAILABLE = False
    print(f"⚠️  IBM Runtime não disponível: {e}")

print("🚀 TESTE RÁPIDO NO HARDWARE IBM QUANTUM - 3 MINUTOS")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO PARA TESTE RÁPIDO
# ===========================================================================

PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 6,
    'NUM_SELECIONAR': 3,
    'TIPO_DADOS': 'sintetico',
    'QAOA_CAMADAS': 2,
    'PARAMETROS_FIXOS': [0.7, 0.3, 0.5, 0.5],
    'PENALIDADE_FACTOR': 15.0,
    'MODO_EXECUCAO': 'ibm_quantum',
    'BACKEND_IBM': 'ibm_fez',
    'NUM_SHOTS': 512,
    'OPTIMIZATION_LEVEL': 1,
}

print("✅ CONFIGURADO PARA TESTE RÁPIDO (3 MINUTOS)")

# ===========================================================================
# CÉLULA 3: GERADOR DE DADOS DE PORTFÓLIO (SIMPLIFICADO)
# ===========================================================================

def gerar_dados_portfolio(config):
    """Gera dados de portfólio simplificados para teste rápido"""
    np.random.seed(42)
    n = config['NUM_ATIVOS']
    
    returns = np.random.uniform(0.05, 0.20, n)
    
    covariance = np.eye(n) * 0.1
    for i in range(n):
        for j in range(i+1, n):
            covariance[i, j] = covariance[j, i] = np.random.uniform(0.01, 0.05)
    
    return returns, covariance

def setup_portfolio_problem(returns, covariance, num_assets_to_select):
    """Configuração rápida do problema"""
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"📊 CONFIGURAÇÃO DO PORTFÓLIO:")
    print(f"• Ativos disponíveis: {N}")
    print(f"• Ativos a selecionar: {budget}")
    print(f"• Retornos: {[f'{r:.3f}' for r in mu]}")
    
    return mu, cov, budget

# Gerar dados do portfólio
returns, covariance = gerar_dados_portfolio(PORTFOLIO_CONFIG)
mu, cov, budget = setup_portfolio_problem(
    returns, covariance, PORTFOLIO_CONFIG['NUM_SELECIONAR']
)

# ===========================================================================
# CÉLULA 4: HAMILTONIANO SIMPLIFICADO PARA TESTE
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO PARA TESTE")
print("=" * 70)

def build_test_hamiltonian(mu, cov, budget, penalty_factor):
    """Hamiltoniano simplificado para teste rápido"""
    N = len(mu)
    
    print(f"⚙️  CONFIGURAÇÃO DO HAMILTONIANO:")
    print(f"   • Número de qubits: {N}")
    print(f"   • Penalidade: {penalty_factor}")
    
    pauli_terms = []
    coefficients = []
    
    # Termos lineares (Z_i)
    for i in range(N):
        coeff = -0.5 * (cov[i, i] - mu[i])
        coeff += penalty_factor * (1 - 2 * budget)
        if abs(coeff) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_terms.append(''.join(pauli_str))
            coefficients.append(coeff)
    
    # Termos quadráticos (Z_i Z_j)
    for i in range(N):
        for j in range(i+1, N):
            coeff = 0.25 * cov[i, j] + 0.5 * penalty_factor
            if abs(coeff) > 1e-10:
                pauli_str = ['I'] * N
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_terms.append(''.join(pauli_str))
                coefficients.append(coeff)
    
    # Termo constante
    constant_term = penalty_factor * budget**2 + 0.5 * np.sum(np.diag(cov))
    pauli_terms.append('I' * N)
    coefficients.append(constant_term)
    
    hamiltonian = SparsePauliOp(pauli_terms, coefficients)
    
    max_coeff = np.max(np.abs(coefficients))
    if max_coeff > 10:
        print(f"⚠️  Normalizando Hamiltoniano: dividindo por {max_coeff:.2f}")
        hamiltonian = hamiltonian / max_coeff
    
    print(f"✅ Hamiltoniano construído: {len(hamiltonian)} termos")
    return hamiltonian

# Construir Hamiltoniano
hamiltonian = build_test_hamiltonian(mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR'])

# ===========================================================================
# CÉLULA 5: EXECUÇÃO RÁPIDA NO HARDWARE IBM - CORRIGIDA + CORRIGIDA PARA OPEN PLAN
# ===========================================================================

def executar_teste_rapido_no_hardware(hamiltonian, config):
    """
    🚀 EXECUÇÃO RÁPIDA: Circuito único com parâmetros fixos
    SEGUINDO EXATAMENTE O PADRÃO DA DOCUMENTAÇÃO OFICIAL
    """
    if not IBM_QUANTUM_AVAILABLE:
        print("❌ IBM Runtime não disponível.")
        return None

    try:
        print("\n🔧 INICIANDO TESTE RÁPIDO NO HARDWARE IBM...")
        
        # 1. CONECTAR AO IBM QUANTUM (Igual à documentação)
        service = QiskitRuntimeService()
        
        # 2. ESCOLHER BACKEND (Igual à documentação - "least_busy")
        print("🔍 Buscando backends disponíveis...")
        backend = service.least_busy(simulator=False, operational=True)
        print(f"🎯 Backend selecionado: {backend.name}")
        print(f"   • Qubits: {backend.configuration().n_qubits}")
        
        # 3. CONSTRUIR CIRCUITO QAOA (p=1)
        print("\n🔨 Construindo circuito QAOA...")
        circuit = QAOAAnsatz(
            cost_operator=hamiltonian,
            reps=config['QAOA_CAMADAS'],
            initial_state=None
        )
        circuit.measure_all()
        
        circuito_fixo = circuit.assign_parameters(config['PARAMETROS_FIXOS'])
        print(f"✅ Circuito construído com {circuit.num_qubits} qubits")
        
        # 4. TRANSPILAR para o hardware (Igual à documentação)
        print("\n🔄 Transpilando circuito...")
        pm = generate_preset_pass_manager(
            optimization_level=config['OPTIMIZATION_LEVEL'],
            backend=backend
        )
        isa_circuit = pm.run(circuito_fixo)
        print(f"✅ Circuito transpilado. Profundidade: {isa_circuit.depth()}")
        
        # 5. EXECUTAR COM SAMPLER - CORREÇÃO DEFINITIVA AQUI
        print("\n⚡ Executando no hardware...")
        
        # 🔥 CORREÇÃO: Usar EXATAMENTE o padrão da documentação
        # Documentação mostra: estimator = Estimator(mode=backend)
        # Para Sampler é o mesmo: sampler = Sampler(mode=backend)
        sampler = Sampler(mode=backend)  # ⬅️ FORMA CORRETA
        
        sampler.options.default_shots = config['NUM_SHOTS']
        print(f"   • Shots: {config['NUM_SHOTS']}")
        print(f"   • Backend: {backend.name}")
        print(f"   • Submetido em: {time.strftime('%H:%M:%S')}")
        
        start_time = time.time()
        
        # Executar o circuito (igual à documentação)
        job = sampler.run([isa_circuit])
        print(f"   • Job ID: {job.job_id()}")
        
        # Aguardar resultados
        print("   ⏳ Aguardando resultados (timeout: 180s)...")
        result = job.result(timeout=180)
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 EXECUÇÃO CONCLUÍDA EM {elapsed_time:.1f} SEGUNDOS!")
        
        # 6. PROCESSAR RESULTADOS
        counts = result[0].data.meas.get_counts()
        total_shots = sum(counts.values())
        
        print(f"\n📊 RESULTADOS DO HARDWARE:")
        print(f"• Backend: {backend.name}")
        print(f"• Shots executados: {total_shots}")
        
        print(f"\n🏆 TOP 5 ESTADOS MAIS PROVÁVEIS:")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:5]:
            prob = count / total_shots
            x = np.array([int(bit) for bit in state])
            num_assets = sum(x)
            valid = "✅" if num_assets == budget else "❌"
            print(f"   {valid} {state}: {prob:.2%} ({num_assets} ativos)")
        
        return counts
            
    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===========================================================================
# CÉLULA 6: EXECUÇÃO PRINCIPAL - SIMPLIFICADA
# ===========================================================================

print("\n" + "="*70)
print("🚀 EXECUTANDO TESTE RÁPIDO NO HARDWARE IBM QUANTUM")
print("="*70)

print("📋 RESUMO DA CONFIGURAÇÃO:")
print(f"• Problema: {PORTFOLIO_CONFIG['NUM_ATIVOS']} ativos, selecionar {PORTFOLIO_CONFIG['NUM_SELECIONAR']}")
print(f"• QAOA: p={PORTFOLIO_CONFIG['QAOA_CAMADAS']}, parâmetros fixos")
print(f"• Hardware: {PORTFOLIO_CONFIG['BACKEND_IBM']}")
print(f"• Shots: {PORTFOLIO_CONFIG['NUM_SHOTS']}")
print(f"• Tempo alvo: < 3 minutos")

# Executar teste rápido
if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum':
    if IBM_QUANTUM_AVAILABLE:
        counts = executar_teste_rapido_no_hardware(hamiltonian, PORTFOLIO_CONFIG)
        
        if counts is not None:
            print("\n" + "="*70)
            print("✅ TESTE CONCLUÍDO COM SUCESSO!")
            print("="*70)
            
            # Análise rápida dos resultados
            total_shots = sum(counts.values())
            valid_shots = 0
            
            for state, count in counts.items():
                x = np.array([int(bit) for bit in state])
                if sum(x) == PORTFOLIO_CONFIG['NUM_SELECIONAR']:
                    valid_shots += count
            
            print(f"\n📈 ESTATÍSTICAS:")
            print(f"• Total de shots: {total_shots}")
            print(f"• Shots em soluções válidas: {valid_shots} ({valid_shots/total_shots*100:.1f}%)")
            print(f"• Soluções distintas encontradas: {len(counts)}")
            
            # Mostrar melhor solução
            if valid_shots > 0:
                valid_states = [(s, c) for s, c in counts.items() 
                              if sum(np.array([int(b) for b in s])) == PORTFOLIO_CONFIG['NUM_SELECIONAR']]
                if valid_states:
                    best_state, best_count = max(valid_states, key=lambda x: x[1])
                    print(f"\n🏆 MELHOR SOLUÇÃO ENCONTRADA:")
                    print(f"• Estado: {best_state}")
                    print(f"• Probabilidade: {best_count/total_shots*100:.1f}%")
                    
                    # Decodificar ativos
                    assets = [i for i, bit in enumerate(best_state) if bit == '1']
                    print(f"• Ativos selecionados: {assets}")
        else:
            print("\n❌ TESTE NÃO CONCLUÍDO - Verifique conexão e créditos")
    else:
        print("\n⚠️  IBM Quantum Runtime não disponível")
else:
    print("\n💻 Modo de execução: simulação (não é hardware real)")

print("\n" + "="*70)
print("🎯 FIM DO TESTE RÁPIDO")
print("="*70)