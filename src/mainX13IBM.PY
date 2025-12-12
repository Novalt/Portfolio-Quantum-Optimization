# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES (MANTIDO)
# ===========================================================================

import numpy as np
import time
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    IBM_QUANTUM_AVAILABLE = True
    print("✅ IBM Quantum Runtime disponível")
except ImportError:
    IBM_QUANTUM_AVAILABLE = False

print("🚀 PROBLEMA DE PORTFÓLIO - 6 ATIVOS NO HARDWARE IBM")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO OTIMIZADA
# ===========================================================================

PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 6,
    'NUM_SELECIONAR': 3,
    'TIPO_DADOS': 'sintetico',
    'QAOA_CAMADAS': 2,
    'PARAMETROS_FIXOS': [0.6, 0.4, 0.5, 0.5,],
    'PENALIDADE_FACTOR': 35.0,  # ⬅️ PENALIDADE AUMENTADA
    'MODO_EXECUCAO': 'ibm_quantum',
    'BACKEND_IBM': 'ibm_fez',
    'NUM_SHOTS': 1024,
    'OPTIMIZATION_LEVEL': 1,
}

print("✅ CONFIGURADO COM PENALIDADE REFORÇADA")

# ===========================================================================
# CÉLULA 3: GERAÇÃO DE DADOS (MANTIDO)
# ===========================================================================

def gerar_dados_portfolio(config):
    np.random.seed(42)
    n = config['NUM_ATIVOS']
    returns = np.random.uniform(0.05, 0.20, n)
    covariance = np.eye(n) * 0.1
    for i in range(n):
        for j in range(i+1, n):
            covariance[i, j] = covariance[j, i] = np.random.uniform(0.01, 0.05)
    return returns, covariance

def setup_portfolio_problem(returns, covariance, num_assets_to_select):
    N = len(returns)
    mu, cov, budget = np.array(returns), np.array(covariance), num_assets_to_select
    print(f"📊 PORTFÓLIO: {N} ativos, selecionar {budget}")
    return mu, cov, budget

returns, covariance = gerar_dados_portfolio(PORTFOLIO_CONFIG)
mu, cov, budget = setup_portfolio_problem(returns, covariance, PORTFOLIO_CONFIG['NUM_SELECIONAR'])

# ===========================================================================
# CÉLULA 4: HAMILTONIANO COM PENALIDADE FORTE
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO COM PENALIDADE FORTE")
print("=" * 70)

def build_strong_penalty_hamiltonian(mu, cov, budget, penalty_factor=50.0):
    """Hamiltoniano com penalidade reforçada para forçar restrição"""
    N = len(mu)
    
    print(f"⚙️  CONFIGURAÇÃO:")
    print(f"   • Qubits: {N}")
    print(f"   • Selecionar: {budget}")
    print(f"   • Penalidade: {penalty_factor} (ALTA)")
    
    pauli_terms = []
    coefficients = []
    
    # 1. Termo objetivo (risco - retorno)
    for i in range(N):
        coeff = -0.5 * (cov[i, i] - mu[i])
        if abs(coeff) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_terms.append(''.join(pauli_str))
            coefficients.append(coeff)
    
    # 2. PENALIDADE FORTE para ∑x_i = budget
    lambda_penalty = penalty_factor * 10.0
    
    # Termos quadráticos Z_i Z_j
    for i in range(N):
        for j in range(i+1, N):
            coeff = 0.25 * lambda_penalty
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_terms.append(''.join(pauli_str))
            coefficients.append(coeff)
    
    # Termos lineares Z_i
    for i in range(N):
        coeff = -0.5 * lambda_penalty * (1 - 2*budget/N)
        found = False
        for idx, pauli in enumerate(pauli_terms):
            if pauli == ('I'*i + 'Z' + 'I'*(N-i-1)):
                coefficients[idx] += coeff
                found = True
                break
        if not found and abs(coeff) > 1e-10:
            pauli_str = ['I'] * N
            pauli_str[i] = 'Z'
            pauli_terms.append(''.join(pauli_str))
            coefficients.append(coeff)
    
    # Termo constante
    constant = lambda_penalty * budget**2 / 4.0
    pauli_terms.append('I' * N)
    coefficients.append(constant)
    
    # 3. Covariância entre ativos
    for i in range(N):
        for j in range(i+1, N):
            coeff = 0.25 * cov[i, j]
            if abs(coeff) > 1e-10:
                pauli_str = ['I'] * N
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                term_str = ''.join(pauli_str)
                if term_str in pauli_terms:
                    idx = pauli_terms.index(term_str)
                    coefficients[idx] += coeff
                else:
                    pauli_terms.append(term_str)
                    coefficients.append(coeff)
    
    hamiltonian = SparsePauliOp(pauli_terms, coefficients)
    
    # Normalização
    max_coeff = np.max(np.abs(coefficients))
    if max_coeff > 5:
        print(f"⚠️  Normalizando: dividindo por {max_coeff:.2f}")
        hamiltonian = hamiltonian / max_coeff
    
    print(f"✅ Hamiltoniano com {len(hamiltonian)} termos")
    return hamiltonian

hamiltonian = build_strong_penalty_hamiltonian(mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR'])

# ===========================================================================
# CÉLULA 5: EXECUÇÃO COM ANÁLISE DETALHADA
# ===========================================================================

def executar_com_analise_detalhada(hamiltonian, config):
    """Execução com análise completa dos resultados"""
    if not IBM_QUANTUM_AVAILABLE:
        print("❌ IBM Runtime não disponível.")
        return None

    try:
        print("\n🔧 INICIANDO EXECUÇÃO NO HARDWARE...")
        
        service = QiskitRuntimeService()
        backend = service.least_busy(simulator=False, operational=True)
        print(f"🎯 Backend: {backend.name} ({backend.configuration().n_qubits} qubits)")
        
        # Construir circuito
        circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=config['QAOA_CAMADAS'])
        circuit.measure_all()
        circuito_fixo = circuit.assign_parameters(config['PARAMETROS_FIXOS'])
        
        # Transpilar
        pm = generate_preset_pass_manager(backend=backend, optimization_level=config['OPTIMIZATION_LEVEL'])
        isa_circuit = pm.run(circuito_fixo)
        print(f"📏 Circuito transpilado: {isa_circuit.depth()} de profundidade")
        
        # Executar
        print(f"\n⚡ Executando {config['NUM_SHOTS']} shots...")
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = config['NUM_SHOTS']
        
        start_time = time.time()
        job = sampler.run([isa_circuit])
        print(f"   • Job ID: {job.job_id()}")
        print(f"   • Horário: {time.strftime('%H:%M:%S')}")
        
        result = job.result(timeout=180)
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 CONCLUÍDO EM {elapsed_time:.1f} SEGUNDOS!")
        
        # Análise detalhada
        counts = result[0].data.meas.get_counts()
        total_shots = sum(counts.values())
        
        print(f"\n📊 RESULTADOS DO HARDWARE:")
        print(f"• Backend: {backend.name}")
        print(f"• Shots totais: {total_shots}")
        
        # Distribuição por número de ativos
        distribution = {}
        valid_states = []
        
        for state, count in counts.items():
            num_assets = sum(int(bit) for bit in state)
            distribution[num_assets] = distribution.get(num_assets, 0) + count
            
            if num_assets == config['NUM_SELECIONAR']:
                valid_states.append((state, count))
        
        print(f"\n📈 DISTRIBUIÇÃO POR NÚMERO DE ATIVOS:")
        for num_assets in sorted(distribution.keys()):
            perc = distribution[num_assets] / total_shots * 100
            marker = "✅" if num_assets == config['NUM_SELECIONAR'] else "❌"
            print(f"   {marker} {num_assets} ativos: {perc:.1f}% ({distribution[num_assets]} shots)")
        
        # Top estados válidos
        if valid_states:
            valid_states.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🏆 TOP 5 SOLUÇÕES VÁLIDAS:")
            for i, (state, count) in enumerate(valid_states[:5]):
                prob = count / total_shots * 100
                assets = [idx for idx, bit in enumerate(state) if bit == '1']
                print(f"   {i+1}. {state}: {prob:.1f}% → ativos {assets}")
        
        return counts, distribution
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        return None, None

# ===========================================================================
# CÉLULA 6: EXECUÇÃO PRINCIPAL
# ===========================================================================

print("\n" + "="*70)
print("🚀 EXECUTANDO PROBLEMA DE 6 ATIVOS")
print("="*70)

if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum' and IBM_QUANTUM_AVAILABLE:
    counts, distribution = executar_com_analise_detalhada(hamiltonian, PORTFOLIO_CONFIG)
    
    if counts is not None:
        print("\n" + "="*70)
        print("✅ ANÁLISE COMPLETA")
        print("="*70)
        
        total_shots = sum(counts.values())
        target = PORTFOLIO_CONFIG['NUM_SELECIONAR']
        shots_validos = distribution.get(target, 0)
        perc_valido = shots_validos / total_shots * 100
        
        print(f"\n📊 MÉTRICAS FINAIS:")
        print(f"• Soluções válidas ({target} ativos): {perc_valido:.1f}%")
        print(f"• Shots válidos: {shots_validos}/{total_shots}")
        
        # Avaliação de desempenho
        if perc_valido > 40:
            print("• Desempenho: 🎉 EXCELENTE")
        elif perc_valido > 25:
            print("• Desempenho: ✅ BOM")
        elif perc_valido > 15:
            print("• Desempenho: ⚠️  ACEITÁVEL")
        else:
            print("• Desempenho: 🔴 PRECISA MELHORAR")
            print("  💡 Sugestão: Aumente PENALIDADE_FACTOR para 70-100")

print("\n" + "="*70)
print("🎯 EXECUÇÃO CONCLUÍDA")
print("="*70)