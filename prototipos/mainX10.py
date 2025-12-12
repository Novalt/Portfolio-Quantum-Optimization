# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO - ATUALIZADO PARA QISKIT 1.0+
# ===========================================================================

import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import warnings
warnings.filterwarnings('ignore')

# Importações IBM Quantum atualizadas para primitivas V2
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, Session
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    IBM_QUANTUM_AVAILABLE = True
except ImportError:
    IBM_QUANTUM_AVAILABLE = False
    print("⚠️  IBM Runtime não disponível. Executando em modo simulação.")

# Para simulação local (fallback)
from qiskit.primitives import StatevectorSampler

print("🚀 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - QISKIT 1.0+")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO FLEXÍVEL - ATUALIZADA
# ===========================================================================

PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 6,              # ⚡ Recomendado: 5-6 qubits para início
    'NUM_SELECIONAR': 3,          # ⚡ Quantos ativos selecionar
    'TIPO_DADOS': 'sintetico',    # ⚡ 'sintetico' ou 'manual'
    
    # 💰 DADOS MANUAIS (se TIPO_DADOS = 'manual')
    'RETORNOS_MANUAL': [0.12, 0.10, 0.14, 0.07, 0.15, 0.11],
    'COVARIANCIA_MANUAL': [
        [0.1, 0.02, 0.01, 0.03, 0.02, 0.01],
        [0.02, 0.15, 0.05, 0.02, 0.03, 0.02],
        [0.01, 0.05, 0.2, 0.04, 0.02, 0.03],
        [0.03, 0.02, 0.04, 0.1, 0.02, 0.01],
        [0.02, 0.03, 0.02, 0.02, 0.12, 0.02],
        [0.01, 0.02, 0.03, 0.01, 0.02, 0.08]
    ],
    
    # ⚙️ CONFIGURAÇÃO DO ALGORITMO - OTIMIZADA
    'PENALIDADE_FACTOR': 1000.0,   # ⚡ 300-500 para IBM Quantum
    'QAOA_CAMADAS': 1,            # ⚡ 2-3 camadas (recomendado)
    'QAOA_ITERACOES': 200,        # ⚡ 50-100 iterações (custo)
    'OTIMIZADOR': 'SPSA',         # ⚡ 'SPSA' recomendado para hardware real/COBYLA (mais estável)
    
    # 🎯 MODO DE EXECUÇÃO - ATUALIZADO
    'MODO_EXECUCAO': 'simulacao',  # ⚡ 'simulacao' ou 'ibm_quantum'
    'BACKEND_IBM': 'ibmq_qasm_simulator',  # ⚡ Simulador para teste
    # 'BACKEND_IBM': 'ibm_brisbane',  # ⚡ Hardware real (descomente depois)
    
    # 🔐 CONFIGURAÇÃO DE RUNTIME ATUALIZADA
    'NUM_SHOTS': 1024,            # ⚡ Número de execuções
    'RESILIENCE_LEVEL': 1,        # ⚡ Nível de resiliência
    'OPTIMIZATION_LEVEL': 3,      # ⚡ Nível de otimização da transpilação
}

print("✅ CONFIGURADO PARA QISKIT 1.0+ E IBM QUANTUM CLOUD")

# ===========================================================================
# CÉLULA 3: GERADOR DE DADOS DE PORTFÓLIO (MANTIDO)
# ===========================================================================

def gerar_dados_portfolio(config):
    """
    Gera dados de portfólio baseado na configuração
    """
    if config['TIPO_DADOS'] == 'manual':
        returns = np.array(config['RETORNOS_MANUAL'])
        covariance = np.array(config['COVARIANCIA_MANUAL'])
    else:
        # Geração de dados sintéticos realistas
        np.random.seed(42)
        n = config['NUM_ATIVOS']
        
        # Retornos entre 5% e 20%
        returns = np.random.uniform(0.05, 0.20, n)
        
        # Matriz de covariância realista
        covariance = np.random.uniform(0.01, 0.15, (n, n))
        covariance = (covariance + covariance.T) / 2  # Torna simétrica
        np.fill_diagonal(covariance, np.random.uniform(0.08, 0.25, n))
        
        # Garantir que seja positiva definida
        for i in range(n):
            covariance[i, i] = np.sum(np.abs(covariance[i])) * 1.1
    
    return returns, covariance

def setup_portfolio_problem(returns, covariance, num_assets_to_select):
    """
    Configuração automática para dados reais de portfólio
    """
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"📊 CONFIGURAÇÃO DO PORTFÓLIO:")
    print(f"• Ativos disponíveis: {N}")
    print(f"• Ativos a selecionar: {budget}")
    print(f"• Retornos esperados: {[f'{r:.3f}' for r in mu]}")
    
    # Análise das soluções clássicas
    solutions = []
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        solutions.append((comb, value, x))
    
    solutions.sort(key=lambda x: x[1])
    best_sol = solutions[0]
    worst_sol = solutions[-1]
    
    print(f"🎯 Melhor portfólio clássico: {best_sol[0]} (valor: {best_sol[1]:.6f})")
    print(f"📉 Pior portfólio clássico: {worst_sol[0]} (valor: {worst_sol[1]:.6f})")
    
    return mu, cov, budget, best_sol

# Gerar dados do portfólio
returns, covariance = gerar_dados_portfolio(PORTFOLIO_CONFIG)
mu, cov, budget, classical_solution = setup_portfolio_problem(
    returns, covariance, PORTFOLIO_CONFIG['NUM_SELECIONAR']
)

classical_comb, classical_val, classical_vec = classical_solution

# ===========================================================================
# CÉLULA 4: HAMILTONIANO OTIMIZADO (MANTIDO)
# ===========================================================================

print("\n🔧 CONFIGURANDO HAMILTONIANO OTIMIZADO")
print("=" * 70)

def build_scalable_hamiltonian(mu, cov, budget, penalty_factor):
    """
    Hamiltoniano escalável para diferentes tamanhos de portfólio
    """
    N = len(mu)
    
    # Cálculo automático da penalidade baseado no tamanho
    max_obj_val = 0.0
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        max_obj_val = max(max_obj_val, abs(value))
    
    penalty = penalty_factor * max_obj_val
    
    print(f"⚙️  CONFIGURAÇÃO DO HAMILTONIANO:")
    print(f"   • Número de qubits: {N}")
    print(f"   • Escala do problema: {max_obj_val:.6f}")
    print(f"   • Penalidade aplicada: {penalty:.2f}")
    
    # Construção do QUBO otimizada
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    Q += cov  # Termo de risco
    q -= mu   # Termo de retorno
    
    # Restrição de budget otimizada
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * budget)
        q[i] += 2 * penalty * budget
    
    for i in range(N):
        for j in range(i+1, N):
            Q[i, j] += 2 * penalty
            Q[j, i] += 2 * penalty
    
    # Conversão para Ising otimizada
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        h[i] += -0.5 * q[i]
        for j in range(N):
            if i == j:
                h[i] += -0.5 * Q[i, j]
            else:
                h[i] += -0.25 * Q[i, j]
    
    for i in range(N):
        for j in range(i+1, N):
            J[i, j] = 0.25 * Q[i, j]
    
    # Operador Pauli eficiente
    pauli_terms = ["I" * N]
    coefficients = [penalty * budget**2 + 0.5 * np.sum(q) + 0.25 * np.sum(Q)]
    
    for i in range(N):
        if abs(h[i]) > 1e-10:
            pauli_str = ["I"] * N
            pauli_str[i] = "Z"
            pauli_terms.append("".join(pauli_str))
            coefficients.append(h[i])
    
    for i in range(N):
        for j in range(i+1, N):
            if abs(J[i, j]) > 1e-10:
                pauli_str = ["I"] * N
                pauli_str[i] = "Z"
                pauli_str[j] = "Z"
                pauli_terms.append("".join(pauli_str))
                coefficients.append(J[i, j])
    
    hamiltonian = SparsePauliOp(pauli_terms, coefficients)
    
    print(f"✅ Hamiltoniano construído:")
    print(f"   • Termos Pauli: {len(hamiltonian)}")
    print(f"   • Complexidade: {N} qubits")
    
    return hamiltonian

hamiltonian = build_scalable_hamiltonian(
    mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR']
)

# ===========================================================================
# CÉLULA 5: CONEXÃO E EXECUÇÃO IBM QUANTUM - ATUALIZADA PARA PRIMITIVAS V2
# ===========================================================================

def setup_ibm_quantum_service():
    """
    Configura e autentica com IBM Quantum - Atualizada
    """
    try:
        if not IBM_QUANTUM_AVAILABLE:
            raise ImportError("IBM Quantum Runtime não instalado")
            
        service = QiskitRuntimeService()
        
        # Listar backends disponíveis
        backends = service.backends()
        print("🔧 BACKENDS DISPONÍVEIS:")
        for i, backend in enumerate(backends):
            status = backend.status()
            qubits = backend.configuration().n_qubits
            print(f"  {i+1}. {backend.name} - {qubits} qubits - {status.status.value}")
        
        return service
    
    except Exception as e:
        print(f"❌ ERRO NA CONEXÃO IBM QUANTUM: {e}")
        print("\n💡 SOLUÇÃO:")
        print("1. Execute: pip install qiskit-ibm-runtime")
        print("2. Acesse: https://quantum-computing.ibm.com/")
        print("3. Crie uma conta e obtenha seu token")
        print("4. Execute no terminal:")
        print("   from qiskit_ibm_runtime import QiskitRuntimeService")
        print('   QiskitRuntimeService.save_account(channel="ibm_quantum", token="SEU_TOKEN")')
        return None

# 🔥 NOVA FUNÇÃO: Avaliação eficiente de estados (do tutorial)
_PARITY = np.array(
    [-1 if bin(i).count("1") % 2 else 1 for i in range(256)],
    dtype=np.complex128,
)

def evaluate_sparse_pauli(state: int, observable: SparsePauliOp) -> complex:
    """Utility for the evaluation of the expectation value of a measured state."""
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(
        state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8
    )
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])

def cost_func_estimator_moderno(params, ansatz, hamiltonian, estimator):
    """
    Função de custo atualizada para primitivas V2 - Baseada no tutorial
    """
    # Publicar o circuito com parâmetros
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    
    results = job.result()[0]
    cost = results.data.evs
    
    # Armazenar histórico (se necessário)
    if hasattr(cost_func_estimator_moderno, 'history'):
        cost_func_estimator_moderno.history.append(cost)
    
    return cost

def run_ibm_quantum_moderno(hamiltonian, config):
    """
    🚀 EXECUÇÃO ATUALIZADA: Seguindo padrão do tutorial IBM com primitivas V2
    """
    try:
        service = QiskitRuntimeService()
        
        # Selecionar backend baseado na configuração
        if config['BACKEND_IBM'] == 'auto':
            backend = service.least_busy(operational=True, simulator=False)
        else:
            backend = service.get_backend(config['BACKEND_IBM'])
        
        print(f"🎯 CONECTADO AO BACKEND: {backend.name}")
        print(f"📊 Status: {backend.status().status.value}")
        print(f"🔢 Qubits: {backend.configuration().n_qubits}")
        
        # 1. CONSTRUIR CIRCUITO QAOA (como no tutorial)
        num_assets = len(mu)
        reps = min(config['QAOA_CAMADAS'], 3)
        
        circuit = QAOAAnsatz(
            cost_operator=hamiltonian, 
            reps=reps,
            initial_state=None
        )
        circuit.measure_all()
        
        print("🔧 CIRCUITO CONSTRUÍDO:")
        print(f"   • Qubits: {circuit.num_qubits}")
        print(f"   • Parâmetros: {len(circuit.parameters)}")
        print(f"   • Profundidade: {circuit.depth()}")
        
        # 2. TRANSPILAÇÃO EXPLÍCITA (CRÍTICO - como no tutorial)
        pm = generate_preset_pass_manager(
            optimization_level=config['OPTIMIZATION_LEVEL'], 
            backend=backend
        )
        candidate_circuit = pm.run(circuit)
        
        print("✅ CIRCUITO TRANSPILADO:")
        print(f"   • Qubits físicos: {candidate_circuit.num_qubits}")
        print(f"   • Profundidade final: {candidate_circuit.depth()}")
        
        # 3. APLICAR LAYOUT AO HAMILTONIANO (CRÍTICO)
        isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout)
        
        # 4. CONFIGURAR PRIMITIVAS V2
        with Session(backend=backend) as session:
            estimator = Estimator(session=session)
            
            # Configurações modernas (como no tutorial)
            estimator.options.default_shots = config['NUM_SHOTS']
            estimator.options.resilience_level = config['RESILIENCE_LEVEL']
            
            # 🔥 SUPRESSÃO DE ERROS (como no tutorial)
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"
            
            print("⚙️  CONFIGURAÇÃO DE EXECUÇÃO:")
            print(f"   • Shots: {estimator.options.default_shots}")
            print(f"   • Resilience: {estimator.options.resilience_level}")
            print(f"   • Dynamical Decoupling: {estimator.options.dynamical_decoupling.enable}")
            
            # 5. CONFIGURAR OTIMIZAÇÃO
            if config['OTIMIZADOR'] == 'SPSA':
                optimizer = SPSA(maxiter=config['QAOA_ITERACOES'])
            else:
                optimizer = COBYLA(maxiter=config['QAOA_ITERACOES'])
            
            # Ponto inicial otimizado
            num_params = 2 * reps
            initial_points = {
                2: [0.7, 0.5],                    # p=1
                4: [0.7, 0.5, 0.8, 0.3],         # p=2
                6: [1.0, 0.8, 1.2, 0.6, 1.1, 0.7] # p=3
            }
            initial_point = initial_points.get(num_params, np.random.uniform(0, 1, num_params))
            
            # Inicializar histórico
            cost_func_estimator_moderno.history = []
            
            print("⚡ INICIANDO EXECUÇÃO NA IBM QUANTUM...")
            print(f"• Backend: {backend.name}")
            print(f"• Camadas QAOA: p={reps}")
            print(f"• Parâmetros: {num_params}")
            print(f"• Iterações: {config['QAOA_ITERACOES']}")
            
            # 6. EXECUTAR OTIMIZAÇÃO
            result = optimizer.minimize(
                cost_func_estimator_moderno,
                initial_point,
                args=(candidate_circuit, isa_hamiltonian, estimator)
            )
            
            print("✅ EXECUÇÃO CONCLUÍDA NA IBM QUANTUM!")
            print(f"• Valor final: {result.fun:.6f}")
            print(f"• Iterações executadas: {result.nfev}")
            
            # 7. PREPARAR RESULTADO COMPATÍVEL
            class QAOResult:
                def __init__(self, opt_result, circuit, hamiltonian):
                    self.optimal_value = opt_result.fun
                    self.optimal_parameters = opt_result.x
                    self.optimal_point = opt_result.x
                    self.cost_history = cost_func_estimator_moderno.history
                    # Criar eigenstate simulado para compatibilidade
                    self.eigenstate = self._simulate_eigenstate(circuit, opt_result.x)
                    
                def _simulate_eigenstate(self, circuit, params):
                    """Simular distribuição para compatibilidade"""
                    try:
                        from qiskit.primitives import StatevectorSampler
                        sampled_circuit = circuit.assign_parameters(params)
                        sampler = StatevectorSampler()
                        job = sampler.run([sampled_circuit])
                        result = job.result()[0]
                        return result.data.meas.get_counts()
                    except:
                        # Fallback simples
                        return {'0' * circuit.num_qubits: 1.0}
            
            qaoa_result = QAOResult(result, candidate_circuit, hamiltonian)
            return qaoa_result
            
    except Exception as e:
        print(f"❌ ERRO NA EXECUÇÃO IBM MODERNA: {e}")
        print("🔄 Voltando para simulação local...")
        return run_local_qaoa(hamiltonian, {**config, 'MODO_EXECUCAO': 'simulacao'})

def get_optimizer_config(mode, num_assets, qaoa_camadas=None):
    """
    Retorna configuração otimizada com ponto inicial CORRETO
    """
    # Define reps baseado no parâmetro ou no tamanho
    if qaoa_camadas:
        reps = qaoa_camadas
    else:
        if num_assets <= 6:
            reps = 2
        elif num_assets <= 8:
            reps = 3
        else:
            reps = 3
    
    presets = {
        'simulacao_pequena': {  # 4-6 ativos
            'reps': reps,
            'max_iter': 150,
            'optimizer': COBYLA,
        },
        'simulacao_media': {    # 7-8 ativos
            'reps': reps,
            'max_iter': 200,
            'optimizer': COBYLA,
        },
        'simulacao_grande': {   # 9-10 ativos
            'reps': reps,
            'max_iter': 250,
            'optimizer': SPSA,
        }
    }
    
    # Seleciona preset baseado no tamanho
    if num_assets <= 6:
        return presets['simulacao_pequena']
    elif num_assets <= 8:
        return presets['simulacao_media']
    else:
        return presets['simulacao_grande']

def run_local_qaoa(hamiltonian, config):
    """
    Execução do QAOA local com ponto inicial CORRETO (fallback)
    """
    num_assets = len(mu)
    qaoa_config = get_optimizer_config(config['MODO_EXECUCAO'], num_assets, config['QAOA_CAMADAS'])
    
    # Override com configuração manual
    if config['QAOA_CAMADAS']:
        qaoa_config['reps'] = config['QAOA_CAMADAS']
    if config['QAOA_ITERACOES']:
        qaoa_config['max_iter'] = config['QAOA_ITERACOES']
    if config['OTIMIZADOR'] == 'SPSA':
        qaoa_config['optimizer'] = SPSA
    else:
        qaoa_config['optimizer'] = COBYLA
    
    print("⚙️  CONFIGURAÇÃO DE PERFORMANCE (SIMULAÇÃO LOCAL):")
    print(f"• Algoritmo: QAOA")
    print(f"• Camadas: p={qaoa_config['reps']}")
    print(f"• Iterações: {qaoa_config['max_iter']}")
    print(f"• Otimizador: {qaoa_config['optimizer'].__name__}")
    print(f"• Qubits: {num_assets}")
    
    sampler = StatevectorSampler()
    optimizer = qaoa_config['optimizer'](maxiter=qaoa_config['max_iter'])
    
    # Ponto inicial com dimensão CORRETA
    num_params = 2 * qaoa_config['reps']
    initial_points = {
        2: [0.7, 0.5],                    # p=1
        4: [0.7, 0.5, 0.8, 0.3],         # p=2
        6: [0.7, 0.5, 0.8, 0.3, 0.6, 0.4] # p=3
    }
    
    if num_params in initial_points:
        initial_point = initial_points[num_params]
        print(f"• Ponto inicial: {initial_point}")
    else:
        initial_point = np.random.uniform(0, 1, num_params)
        print(f"• Ponto inicial: Aleatório ({num_params} parâmetros)")
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=qaoa_config['reps'],
        initial_point=initial_point
    )
    
    print("⚡ Executando otimização quântica local...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

def run_sampler_moderno(circuit, config, backend=None):
    """
    Execução moderna do Sampler para obter distribuição final
    """
    if config['MODO_EXECUCAO'] == 'ibm_quantum' and backend is not None:
        with Session(backend=backend) as session:
            sampler = Sampler(session=session)
            sampler.options.default_shots = config['NUM_SHOTS']
            
            # Configurações de supressão de erro
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            
            pub = (circuit,)
            job = sampler.run([pub])
            result = job.result()[0]
            return result.data.meas.get_counts()
    else:
        # Fallback para simulação local
        sampler = StatevectorSampler()
        job = sampler.run([circuit])
        result = job.result()[0]
        return result.data.meas.get_counts()

# ===========================================================================
# CÉLULA 6: EXECUÇÃO PRINCIPAL ATUALIZADA
# ===========================================================================

print("\n🚀 EXECUTANDO ALGORITMO QUÂNTICO ESCALÁVEL - ATUALIZADO")
print("=" * 70)

# Testar conexão IBM Quantum (opcional)
if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum':
    service = setup_ibm_quantum_service()

# Execução com a configuração atual
if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum' and IBM_QUANTUM_AVAILABLE:
    print("🌐 MODO: IBM QUANTUM CLOUD (PRIMITIVAS V2)")
    result = run_ibm_quantum_moderno(hamiltonian, PORTFOLIO_CONFIG)
else:
    print("💻 MODO: SIMULAÇÃO LOCAL")
    if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum' and not IBM_QUANTUM_AVAILABLE:
        print("⚠️  IBM Quantum não disponível. Executando localmente.")
    result = run_local_qaoa(hamiltonian, PORTFOLIO_CONFIG)


# ===========================================================================
# CÉLULA 6.5: CORREÇÃO DA FUNÇÃO DE AVALIAÇÃO PARA CDF
# ===========================================================================

def evaluate_portfolio_objective(bitstring, mu, cov, budget):
    """
    Avalia o valor objetivo ORIGINAL do problema de portfólio
    (sem as penalidades do Hamiltoniano)
    """
    x = np.array([int(bit) for bit in bitstring])
    
    # Verificar se é solução válida
    num_assets = sum(x)
    if num_assets != budget:
        # Retornar um valor alto para soluções inválidas
        return float('inf')
    
    # Calcular o valor objetivo original: xᵀΣx - μᵀx
    risk = x @ cov @ x
    returns = mu @ x
    objective_value = risk - returns
    
    return objective_value

def samples_to_objective_values_corrected(samples, mu, cov, budget):
    """Converte amostras para valores do objetivo ORIGINAL"""
    objective_values = {}
    for bit_str, prob in samples.items():
        # Converter string binária para vetor
        if len(bit_str) != len(mu):
            continue
            
        # Avaliar objetivo original
        fval = evaluate_portfolio_objective(bit_str, mu, cov, budget)
        objective_values[fval] = objective_values.get(fval, 0) + prob
    
    return objective_values

def analyze_penalty_effectiveness(analysis, mu, cov, budget, penalty_factor):
    """Analisa se a penalidade está funcionando"""
    print("\n🔍 ANÁLISE DE EFETIVIDADE DA PENALIDADE:")
    
    counts = analysis.get('probability_distribution', {})
    valid_count = 0
    invalid_counts = {}
    
    for bitstring, prob in counts.items():
        x = np.array([int(bit) for bit in bitstring])
        num_assets = sum(x)
        
        if num_assets == budget:
            valid_count += prob
        else:
            if num_assets not in invalid_counts:
                invalid_counts[num_assets] = 0
            invalid_counts[num_assets] += prob
    
    print(f"✅ Probabilidade em válidas: {valid_count:.4f} ({valid_count*100:.2f}%)")
    print(f"❌ Probabilidade em inválidas: {1-valid_count:.4f} ({(1-valid_count)*100:.2f}%)")
    
    if invalid_counts:
        print("\n📊 DISTRIBUIÇÃO DE INVÁLIDOS:")
        for num_assets, prob in sorted(invalid_counts.items()):
            print(f"   • {num_assets} ativos: {prob:.4f} ({prob*100:.2f}%)")
    
    # Calcular violação média
    avg_violation = 0
    for num_assets, prob in invalid_counts.items():
        violation = abs(num_assets - budget)
        avg_violation += violation * prob
    
    print(f"⚠️  Violação média: {avg_violation:.2f} ativos")
    
    # Recomendações
    if valid_count < 0.5:
        needed_penalty = penalty_factor * (0.5 / valid_count) if valid_count > 0 else penalty_factor * 10
        print(f"🚨 RECOMENDAÇÃO: Aumente PENALIDADE_FACTOR para {needed_penalty:.0f}")

def debug_detailed_solutions(analysis, mu, cov, budget):
    """Análise detalhada das soluções"""
    print("\n🔎 ANÁLISE DETALHADA DAS SOLUÇÕES:")
    
    valid_solutions = analysis.get('all_valid_solutions', [])
    
    if valid_solutions:
        print(f"🏆 TOP 10 SOLUÇÕES VÁLIDAS (de {len(valid_solutions)}):")
        for i, (bitstr, value, prob, assets) in enumerate(valid_solutions[:10]):
            print(f"   {i+1:2d}. {assets} → valor: {value:.6f}, prob: {prob:.4f}")
    else:
        print("   ❌ Nenhuma solução válida encontrada!")
    
    # Análise das soluções inválidas mais prováveis
    counts = analysis.get('probability_distribution', {})
    invalid_solutions = []
    
    for bitstr, prob in counts.items():
        x = np.array([int(bit) for bit in bitstr])
        num_assets = sum(x)
        if num_assets != budget and prob > 0.01:
            invalid_solutions.append((bitstr, prob, num_assets, list(np.where(x == 1)[0])))
    
    invalid_solutions.sort(key=lambda x: x[1], reverse=True)
    
    if invalid_solutions:
        print(f"\n📉 TOP 5 SOLUÇÕES INVÁLIDAS:")
        for i, (bitstr, prob, num_assets, assets) in enumerate(invalid_solutions[:5]):
            print(f"   {i+1:2d}. {assets} → {num_assets} ativos, prob: {prob:.4f}")


# ===========================================================================
# CÉLULA 7: ANÁLISE ESCALÁVEL DOS RESULTADOS - ATUALIZADA
# ===========================================================================

print("\n📊 RELATÓRIO DE RESULTADOS ESCALÁVEL - ATUALIZADO")
print("=" * 70)

def analyze_scalable_results(result, classical_comb, classical_val, mu, cov, budget):
    """
    Análise escalável para diferentes tamanhos de problema - Atualizada
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        return {"status": "ERRO", "message": "Sem resultados para análise"}
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Coletar soluções
    valid_solutions = []
    invalid_high_prob = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != len(mu):
            continue
            
        x = np.array([int(bit) for bit in bitstring])
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        assets = [i for i, val in enumerate(x) if val == 1]
        
        if num_assets == budget:  # Solução válida
            valid_solutions.append((bitstring, objective_value, prob, assets))
        elif prob > 0.01:  # Soluções inválidas com alta probabilidade
            invalid_high_prob.append((bitstring, prob, assets))
    
    # Ordenar soluções válidas
    valid_solutions.sort(key=lambda x: x[1])
    
    if valid_solutions:
        best_solution = valid_solutions[0]
        total_valid_prob = sum(sol[2] for sol in valid_solutions)
        optimal_prob = next((sol[2] for sol in valid_solutions if sol[3] == list(classical_comb)), 0.0)
        
        print("🎯 SOLUÇÃO QUÂNTICA ENCONTRADA:")
        print(f"• Portfólio recomendado: {best_solution[3]}")
        print(f"• Valor da solução: {best_solution[1]:.6f}")
        print(f"• Solução ótima clássica: {list(classical_comb)}")
        
        print(f"\n📈 EFICÁCIA DO ALGORITMO:")
        print(f"• Probabilidade na solução ótima: {optimal_prob:.4f} ({optimal_prob*100:.2f}%)")
        print(f"• Eficiência total em válidas: {total_valid_prob:.4f} ({total_valid_prob*100:.2f}%)")
        print(f"• Número de soluções válidas: {len(valid_solutions)}")
        print(f"• Precisão (gap): {abs(best_solution[1] - classical_val)/abs(classical_val)*100:.2f}%")
        
        # Análise de problemas
        if invalid_high_prob:
            print(f"\n⚠️  ALERTAS:")
            for bitstring, prob, assets in invalid_high_prob[:3]:
                print(f"   • Estado {bitstring}: {prob:.3f} (ativos: {assets})")
        
        # Avaliação de usabilidade escalável
        if optimal_prob > 0.1 and total_valid_prob > 0.3:
            usability = "✅ ALTA - Pronta para uso"
            score = "A"
        elif optimal_prob > 0.05 and total_valid_prob > 0.15:
            usability = "✅ MÉDIA - Funcional"
            score = "B"
        elif optimal_prob > 0.02 and total_valid_prob > 0.08:
            usability = "⚠️  BÁSICA - Experimental"
            score = "C"
        elif optimal_prob > 0:
            usability = "🔴 LIMITADA - Demonstração"
            score = "D"
        else:
            usability = "💥 CRÍTICA - Não funcional"
            score = "F"
        
        print(f"\n🏆 AVALIAÇÃO FINAL: {score}")
        print(f"• Usabilidade: {usability}")
        
        return {
            'portfolio': best_solution[3],
            'value': best_solution[1],
            'optimal_probability': optimal_prob,
            'efficiency': total_valid_prob,
            'usability_score': score,
            'usability': usability,
            'num_valid_solutions': len(valid_solutions),
            'all_valid_solutions': valid_solutions,
            'probability_distribution': probabilities
        }
    
    return {"status": "SEM_SOLUCOES", "message": "Nenhuma solução válida encontrada"}

# Análise dos resultados
analysis = analyze_scalable_results(result, classical_comb, classical_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 7.5: CHAMADAS DE DEBUG CORRIGIDAS
# ===========================================================================

print("\n🔍 ANÁLISE DE DEBUG DETALHADA")
print("=" * 70)

# Executar análises de debug APÓS a análise principal
if 'portfolio' in analysis:
    analyze_penalty_effectiveness(analysis, mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR'])
    debug_detailed_solutions(analysis, mu, cov, budget)
else:
    print("❌ Não foi possível executar análise de debug - sem resultados válidos")


# ===========================================================================
# CÉLULA 8: ANÁLISE CDF E VISUALIZAÇÕES (NOVO - DO TUTORIAL)
# ===========================================================================

print("\n📈 ANÁLISE ESTATÍSTICA AVANÇADA - CDF")
print("=" * 70)

def samples_to_objective_values(samples, hamiltonian):
    """Convert the samples to values of the objective function."""
    objective_values = {}
    for bit_str, prob in samples.items():
        candidate_sol = int(bit_str, 2)  # Converter binário para inteiro
        fval = evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        objective_values[fval] = objective_values.get(fval, 0) + prob
    return objective_values

def plot_cdf(objective_values, title="Função de Distribuição Cumulativa"):
    """Plot Cumulative Distribution Function - Baseado no tutorial"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x_vals = sorted(objective_values.keys(), reverse=True)
    y_vals = np.cumsum([objective_values[x] for x in x_vals])
    
    ax.plot(x_vals, y_vals, color='tab:purple', linewidth=2)
    
    if x_vals:
        min_val = min(x_vals)
        ax.axvline(x=min_val, color='tab:red', linestyle='--', 
                   label=f'Melhor valor: {min_val:.4f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Valor da Função Objetivo", fontsize=12)
    ax.set_ylabel("Função de Distribuição Cumulativa", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Gerar análise CDF se temos resultados
if 'probability_distribution' in analysis:
    result_dist = samples_to_objective_values_corrected(
    analysis['probability_distribution'], 
    mu, cov, budget
    )
    
    print("📊 ESTATÍSTICAS DA DISTRIBUIÇÃO:")
    if result_dist:
        min_val = min(result_dist.keys())
        max_val = max(result_dist.keys())
        mean_val = sum(k*v for k, v in result_dist.items()) / sum(result_dist.values())
        
        print(f"• Melhor valor: {min_val:.6f}")
        print(f"• Pior valor: {max_val:.6f}") 
        print(f"• Valor médio: {mean_val:.6f}")
        print(f"• Número de valores únicos: {len(result_dist)}")
        
        # Plotar CDF
        plot_cdf(result_dist, "Desempenho do Algoritmo Quântico")
        
        # Análise de concentração
        top_prob = sum(sorted(result_dist.values(), reverse=True)[:3])
        print(f"• Probabilidade concentrada nos top 3: {top_prob:.3f} ({top_prob*100:.1f}%)")

# Plotar histórico de convergência se disponível
if hasattr(result, 'cost_history') and result.cost_history:
    plt.figure(figsize=(12, 6))
    plt.plot(result.cost_history, 'b-', linewidth=2)
    plt.xlabel("Iteração")
    plt.ylabel("Valor da Função Custo")
    plt.title("Convergência do QAOA")
    plt.grid(alpha=0.3)
    plt.show()

# ===========================================================================
# CÉLULA 9: RELATÓRIO EXECUTIVO COMPLETO ATUALIZADO
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - PORTFÓLIO QUÂNTICO ATUALIZADO")
print("=" * 70)

if 'portfolio' in analysis:
    print(f"\n✅ RESULTADO OBTIDO:")
    print(f"   • Portfólio recomendado: {analysis['portfolio']}")
    print(f"   • Valor da solução: {analysis['value']:.6f}")
    print(f"   • Nota: {analysis['usability_score']}/A")
    
    print(f"\n📊 PERFORMANCE DO ALGORITMO:")
    print(f"   • Eficiência: {analysis['efficiency']:.3f} ({analysis['efficiency']*100:.1f}%)")
    print(f"   • Precisão ótima: {analysis['optimal_probability']:.3f} ({analysis['optimal_probability']*100:.1f}%)")
    print(f"   • Soluções válidas: {analysis['num_valid_solutions']}")
    
    print(f"\n🎯 STATUS:")
    print(f"   {analysis['usability']}")
    
    print(f"\n🚀 RECOMENDAÇÕES PARA MELHORIA:")
    if analysis['usability_score'] in ['D', 'F']:
        print("1. 🔼 Aumentar PENALIDADE_FACTOR para 500-800")
        print("2. 🔼 Aumentar QAOA_CAMADAS para 3-4")
        print("3. 🔄 Mudar para otimizador SPSA")
        print("4. 📉 Reduzir número de ativos para teste")
    elif analysis['usability_score'] == 'C':
        print("1. 🔼 Aumentar PENALIDADE_FACTOR para 400-600")
        print("2. 🔼 Aumentar QAOA_ITERACOES para 250+")
        print("3. ✅ Manter configuração atual para estudos")
    elif analysis['usability_score'] == 'B':
        print("1. ✅ Configuração adequada para protótipos")
        print("2. 🔄 Testar com dados reais")
        print("3. 🌐 Considerar execução em IBM Quantum")
    else:  # A
        print("1. 🎉 Configuração excelente!")
        print("2. 🌐 Preparar para execução em hardware real")
        print("3. 📊 Integrar com análise clássica")

print(f"\n🔧 CONFIGURAÇÃO ATUAL:")
print(f"   • Ativos: {PORTFOLIO_CONFIG['NUM_ATIVOS']} | Selecionar: {PORTFOLIO_CONFIG['NUM_SELECIONAR']}")
print(f"   • Penalidade: {PORTFOLIO_CONFIG['PENALIDADE_FACTOR']}")
print(f"   • Camadas QAOA: {PORTFOLIO_CONFIG['QAOA_CAMADAS']}")
print(f"   • Iterações: {PORTFOLIO_CONFIG['QAOA_ITERACOES']}")
print(f"   • Otimizador: {PORTFOLIO_CONFIG['OTIMIZADOR']}")
print(f"   • Modo: {PORTFOLIO_CONFIG['MODO_EXECUCAO']}")
print(f"   • Nível Otimização: {PORTFOLIO_CONFIG['OPTIMIZATION_LEVEL']}")

print(f"\n🆕 NOVAS FUNCIONALIDADES IMPLEMENTADAS:")
print(f"   ✅ Primitivas Qiskit Runtime V2")
print(f"   ✅ Transpilação explícita com Pass Manager") 
print(f"   ✅ Aplicação de layout ao Hamiltoniano")
print(f"   ✅ Supressão de erros: Dynamical Decoupling")
print(f"   ✅ Supressão de erros: Gate Twirling")
print(f"   ✅ Análise CDF (Função Distribuição Cumulativa)")
print(f"   ✅ Avaliação eficiente de estados")

print(f"\n💡 PRESETS RECOMENDADOS ATUALIZADOS:")

print(f"\n🖥️  PARA MÁQUINA COMUM:")
print("   • 4-6 ativos: PENALIDADE=300, CAMADAS=2, ITERAÇÕES=150")
print("   • 7-8 ativos: PENALIDADE=400, CAMADAS=3, ITERAÇÕES=200") 
print("   • 9-10 ativos: PENALIDADE=500, CAMADAS=3, ITERAÇÕES=250")

print(f"\n☁️  PARA IBM QUANTUM (PRIMITIVAS V2):")
print("   • 4-6 ativos: PENALIDADE=400, CAMADAS=2, ITERAÇÕES=100")
print("   • 7-8 ativos: PENALIDADE=500, CAMADAS=2, ITERAÇÕES=150")
print("   • OTIMIZATION_LEVEL=3, RESILIENCE_LEVEL=1")

print(f"\n🔧 PARA USAR IBM QUANTUM COM PRIMITIVAS V2:")
print("1. Altere MODO_EXECUCAO para 'ibm_quantum'")
print("2. Configure seu token IBM:")
print("   from qiskit_ibm_runtime import QiskitRuntimeService")
print('   QiskitRuntimeService.save_account(token="SEU_TOKEN")')
print("3. Escolha um backend disponível")

print(f"\n🌟 FERRAMENTA DE OTIMIZAÇÃO QUÂNTICA - ATUALIZADA PARA QISKIT 1.0+!")
print("✅ Pronta para IBM Quantum com primitivas modernas V2")
print("✅ Alinhada com tutorial oficial do Qiskit")
print("✅ Inclui supressão de erros e análise estatística avançada")

# ===========================================================================
# CÉLULA 10: FUNÇÕES AUXILIARES ADICIONAIS (NOVO)
# ===========================================================================

def best_solution(samples, hamiltonian):
    """Encontra a solução com menor custo - Baseado no tutorial"""
    min_cost = float('inf')
    min_sol = None
    for bit_str in samples.keys():
        candidate_sol = int(bit_str, 2)  # Converter binário para inteiro
        fval = evaluate_sparse_pauli(candidate_sol, hamiltonian).real
        if fval <= min_cost:
            min_cost = fval
            min_sol = candidate_sol
    return min_sol

def to_bitstring(integer, num_bits):
    """Converte inteiro para bitstring - Baseado no tutorial"""
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

# Exemplo de uso das funções auxiliares
if 'probability_distribution' in analysis:
    best_sol = best_solution(analysis['probability_distribution'], hamiltonian)
    best_sol_bitstring = to_bitstring(best_sol, len(mu))
    best_sol_bitstring.reverse()  # Qiskit usa little endian
    print(f"\n🔍 MELHOR SOLUÇÃO (ANÁLISE AUXILIAR):")
    print(f"   • Bitstring: {best_sol_bitstring}")
    print(f"   • Valor: {evaluate_sparse_pauli(best_sol, hamiltonian).real:.6f}")

print(f"\n🎯 EXECUÇÃO CONCLUÍDA COM SUCESSO!")