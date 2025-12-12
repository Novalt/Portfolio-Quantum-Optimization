# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO - IBM QUANTUM CLOUD
# ===========================================================================

import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options, Sampler
import warnings
warnings.filterwarnings('ignore')

print("🚀 PORTFÓLIO QUÂNTICO - OTIMIZADO PARA IBM QUANTUM CLOUD")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO OTIMIZADA PARA IBM QUANTUM
# ===========================================================================

PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 8,             # ⚡ Otimizado para IBM Quantum (10 qubits)
    'NUM_SELECIONAR': 2,          # ⚡ Problema não trivial mas executável
    'TIPO_DADOS': 'realista',     # ⚡ Dados baseados em mercado real
    
    # ⚙️ CONFIGURAÇÃO OTIMIZADA PARA IBM QUANTUM
    'PENALIDADE_FACTOR': 1.5,     # ⚡ Penalidade reduzida para melhor convergência
    'QAOA_CAMADAS': 2,            # ⚡ 2 camadas para evitar deep circuits
    'QAOA_ITERACOES': 80,        # ⚡ Balance entre custo e performance
    'OTIMIZADOR': 'SPSA',         # ⚡ SPSA recomendado para hardware real
    
    # 🎯 CONFIGURAÇÃO IBM QUANTUM
    'MODO_EXECUCAO': 'ibm_quantum',  # ⚡ Executar na nuvem
    'BACKEND_IBM': 'ibmq_qasm_simulator',  # ⚡ Simulador para testes
    #'BACKEND_IBM': 'ibm_fez',  # ⚡ Hardware real
    # 'BACKEND_IBM': 'ibm_torino',   # ⚡ Backend AVANÇADO (127 qubits) - para problemas maiores
    # 'BACKEND_IBM': 'ibm_marrakesh', # ⚡ Backend básico (5 qubits)

    
    # 🔐 CONFIGURAÇÃO DE RUNTIME OTIMIZADA
    'NUM_SHOTS': 1024,            # ⚡ Shots padrão para boa estatística
    'RESILIENCE_LEVEL': 1,        # ⚡ Resilience level 1 para mitigação básica
    'OPTIMIZATION_LEVEL': 1,      # ⚡ Otimização básica do circuito
    
    'SEED': 42,
}

print("🎯 CONFIGURADO PARA IBM QUANTUM CLOUD")
print("💡 10 ativos, 4 selecionar - Problema ideal para hardware atual")

# ===========================================================================
# CÉLULA 3: VERIFICAÇÃO E CONEXÃO COM IBM QUANTUM
# ===========================================================================

def setup_ibm_quantum_service():
    """
    Configura e verifica conexão com IBM Quantum - VERSÃO CORRIGIDA
    """
    try:
        print("🔗 Conectando com IBM Quantum...")
        service = QiskitRuntimeService()
        
        # Verificar backends disponíveis - FORMA CORRIGIDA
        backends = service.backends()
        print("✅ Conectado com IBM Quantum Cloud")
        print("🔧 Backends disponíveis:")
        
        available_backends = []
        for i, backend in enumerate(backends):
            # 🔥 CORREÇÃO: Nova forma de verificar status
            status_obj = backend.status()
            status_value = status_obj.status if hasattr(status_obj, 'status') else 'unknown'
            qubits = backend.configuration().n_qubits
            
            # Usar operational como critério principal
            if status_obj.operational:
                available_backends.append(backend)
                print(f"  {i+1}. {backend.name} - {qubits} qubits - {status_value}")
        
        # Verificar se o backend configurado está disponível
        target_backend = PORTFOLIO_CONFIG['BACKEND_IBM']
        backend_available = any(backend.name == target_backend for backend in available_backends)
        
        if not backend_available and available_backends:
            print(f"⚠️  Backend {target_backend} não disponível. Usando alternativo...")
            # Usar primeiro backend disponível com pelo menos 10 qubits
            for backend in available_backends:
                if backend.configuration().n_qubits >= 10:
                    PORTFOLIO_CONFIG['BACKEND_IBM'] = backend.name
                    print(f"🎯 Backend alternativo selecionado: {backend.name}")
                    break
        
        return service
    
    except Exception as e:
        print(f"❌ ERRO na conexão IBM Quantum: {e}")
        print("🔄 Alternando para simulação local...")
        return None

# ===========================================================================
# CÉLULA 4: DADOS REALISTAS OTIMIZADOS
# ===========================================================================

def gerar_dados_reais_otimizados():
    """
    Gera dados realistas otimizados para execução quântica
    """
    np.random.seed(42)
    n = PORTFOLIO_CONFIG['NUM_ATIVOS']
    
    print("📊 GERANDO DADOS DE MERCADO REALISTAS...")
    
    # Retornos anuais realistas (5% a 25%)
    returns = np.array([0.08, 0.12, 0.15, 0.06, 0.18, 0.09, 0.11, 0.14, 0.07, 0.10])
    
    # Matriz de covariância realista e bem condicionada
    base_cov = np.array([
        [0.040, 0.008, 0.012, 0.004, 0.015, 0.006, 0.009, 0.011, 0.005, 0.007],
        [0.008, 0.090, 0.010, 0.007, 0.020, 0.008, 0.012, 0.015, 0.006, 0.009],
        [0.012, 0.010, 0.120, 0.009, 0.025, 0.010, 0.015, 0.018, 0.008, 0.012],
        [0.004, 0.007, 0.009, 0.030, 0.006, 0.012, 0.008, 0.007, 0.015, 0.006],
        [0.015, 0.020, 0.025, 0.006, 0.150, 0.012, 0.020, 0.022, 0.009, 0.016],
        [0.006, 0.008, 0.010, 0.012, 0.012, 0.060, 0.010, 0.009, 0.008, 0.011],
        [0.009, 0.012, 0.015, 0.008, 0.020, 0.010, 0.080, 0.014, 0.007, 0.013],
        [0.011, 0.015, 0.018, 0.007, 0.022, 0.009, 0.014, 0.100, 0.008, 0.015],
        [0.005, 0.006, 0.008, 0.015, 0.009, 0.008, 0.007, 0.008, 0.040, 0.006],
        [0.007, 0.009, 0.012, 0.006, 0.016, 0.011, 0.013, 0.015, 0.006, 0.070]
    ])
    
    # Adicionar pequeno ruído para tornar única
    noise = np.random.normal(0, 0.001, (n, n))
    noise = (noise + noise.T) / 2  # Manter simetria
    covariance = base_cov + noise
    
    # Garantir positiva definida
    min_eig = np.min(np.real(np.linalg.eigvals(covariance)))
    if min_eig < 0:
        covariance -= min_eig * np.eye(n) + 0.001
    
    print(f"✅ Dados gerados: {n} ativos")
    print(f"📈 Retornos: {[f'{r:.3f}' for r in returns]}")
    print(f"📊 Variâncias: {[f'{covariance[i,i]:.3f}' for i in range(3)]}...")
    
    return returns, covariance

# ===========================================================================
# CÉLULA 5: CONFIGURAÇÃO DO PROBLEMA
# ===========================================================================

def setup_problema_ibm_otimizado(returns, covariance, num_assets_to_select):
    """
    Configura problema otimizado para IBM Quantum
    """
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"\n📊 CONFIGURAÇÃO DO PORTFÓLIO:")
    print(f"• Ativos: {N}, Selecionar: {budget}")
    
    # Amostrar soluções para análise (não todas para performance)
    sample_solutions = []
    all_combinations = list(combinations(range(N), budget))
    
    # Amostrar 500 combinações para análise
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_combinations), 
                                    size=min(500, len(all_combinations)), 
                                    replace=False)
    
    for idx in sample_indices:
        comb = all_combinations[idx]
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        sample_solutions.append((comb, value, x))
    
    sample_solutions.sort(key=lambda x: x[1])
    best_sample = sample_solutions[0]
    
    sample_values = [sol[1] for sol in sample_solutions]
    
    print(f"🎯 Melhor portfólio na amostra: {best_sample[0]}")
    print(f"📈 Estatísticas da amostra:")
    print(f"  • Valor: {best_sample[1]:.6f}")
    print(f"  • Média: {np.mean(sample_values):.6f}")
    print(f"  • Std: {np.std(sample_values):.6f}")
    
    return mu, cov, budget, best_sample, sample_solutions

# Gerar dados
returns, covariance = gerar_dados_reais_otimizados()
mu, cov, budget, reference_solution, sample_solutions = setup_problema_ibm_otimizado(
    returns, covariance, PORTFOLIO_CONFIG['NUM_SELECIONAR']
)

reference_comb, reference_val, reference_vec = reference_solution

# ===========================================================================
# CÉLULA 6: HAMILTONIANO OTIMIZADO PARA IBM QUANTUM
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO OTIMIZADO")
print("=" * 70)

def build_hamiltonian_ibm_otimizado(mu, cov, budget, penalty_factor, sample_solutions):
    """
    Hamiltoniano otimizado para execução no IBM Quantum
    """
    N = len(mu)
    
    # Calcular escala de forma inteligente
    sample_values = [sol[1] for sol in sample_solutions]
    avg_val = np.mean(sample_values)
    std_val = np.std(sample_values)
    
    # Penalidade adaptativa - CRÍTICO para IBM Quantum
    penalty = penalty_factor * std_val
    
    print(f"⚙️  CONFIGURAÇÃO IBM OTIMIZADA:")
    print(f"   • Qubits: {N}")
    print(f"   • Escala do problema: {std_val:.6f}")
    print(f"   • Penalidade calculada: {penalty:.6f}")
    
    # Construção eficiente do QUBO
    Q = cov.copy()
    q = -mu.copy()
    
    # Restrição de budget otimizada
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * budget)
    
    for i in range(N):
        for j in range(i+1, N):
            Q[i, j] += 2 * penalty
            Q[j, i] = Q[i, j]  # Manter simetria
    
    # Conversão direta para Ising
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        h[i] = -0.5 * q[i] - 0.5 * Q[i, i]
        for j in range(i+1, N):
            h[i] += -0.25 * Q[i, j]
            J[i, j] = 0.25 * Q[i, j]
    
    # Construção do operador Pauli otimizada
    pauli_list = []
    coeffs_list = []
    
    # Termo constante
    const = penalty * budget**2 + 0.5 * np.sum(q) + 0.25 * np.sum(Q)
    if abs(const) > 1e-10:
        pauli_list.append("I" * N)
        coeffs_list.append(const)
    
    # Termos Z
    for i in range(N):
        if abs(h[i]) > 1e-10:
            pauli_str = ["I"] * N
            pauli_str[i] = "Z"
            pauli_list.append("".join(pauli_str))
            coeffs_list.append(h[i])
    
    # Termos ZZ
    for i in range(N):
        for j in range(i+1, N):
            if abs(J[i, j]) > 1e-10:
                pauli_str = ["I"] * N
                pauli_str[i] = "Z"
                pauli_str[j] = "Z"
                pauli_list.append("".join(pauli_str))
                coeffs_list.append(J[i, j])
    
    hamiltonian = SparsePauliOp(pauli_list, coeffs_list)
    
    # Análise crítica da escala
    coeffs_array = np.array(coeffs_list)
    norm_hamiltonian = np.linalg.norm(coeffs_array)
    
    print(f"✅ Hamiltoniano IBM otimizado:")
    print(f"   • Termos Pauli: {len(hamiltonian)}")
    print(f"   • Norma: {norm_hamiltonian:.2f}")
    
    # Verificação importante para IBM Quantum
    if norm_hamiltonian > 50:
        print("⚠️  ALERTA: Hamiltoniano pode estar com escala alta")
    else:
        print("✅ Escala adequada para IBM Quantum")
    
    return hamiltonian, penalty

hamiltonian, actual_penalty = build_hamiltonian_ibm_otimizado(
    mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR'], sample_solutions
)

# ===========================================================================
# CÉLULA 7: EXECUÇÃO OTIMIZADA PARA IBM QUANTUM
# ===========================================================================

print("\n🚀 INICIANDO EXECUÇÃO NO IBM QUANTUM CLOUD")
print("=" * 70)

def run_ibm_quantum_otimizado(hamiltonian, config):
    """
    Execução otimizada para IBM Quantum Cloud
    """
    try:
        service = QiskitRuntimeService()
        backend = service.get_backend(config['BACKEND_IBM'])
        
        print(f"🎯 CONECTADO AO BACKEND: {backend.name}")
        print(f"📊 Status: {backend.status().status.value}")
        print(f"🔢 Qubits disponíveis: {backend.configuration().n_qubits}")
        
        # Configuração de runtime otimizada
        options = Options()
        options.optimization_level = config['OPTIMIZATION_LEVEL']
        options.resilience_level = config['RESILIENCE_LEVEL']
        options.execution.shots = config['NUM_SHOTS']
        
        # Configuração QAOA para IBM Quantum
        reps = config['QAOA_CAMADAS']
        
        if config['OTIMIZADOR'] == 'SPSA':
            optimizer = SPSA(maxiter=config['QAOA_ITERACOES'])
        else:
            optimizer = COBYLA(maxiter=config['QAOA_ITERACOES'])
        
        # Pontos iniciais otimizados para IBM
        if reps == 2:
            initial_point = [0.8, 0.4, 0.6, 0.3]  # Valores testados
        elif reps == 3:
            initial_point = [0.8, 0.3, 0.7, 0.4, 0.5, 0.5]
        else:
            initial_point = np.random.uniform(0.1, 0.9, 2 * reps)
        
        print("⚙️  CONFIGURAÇÃO DE EXECUÇÃO:")
        print(f"• Backend: {backend.name}")
        print(f"• Shots: {config['NUM_SHOTS']}")
        print(f"• Camadas QAOA: p={reps}")
        print(f"• Iterações: {config['QAOA_ITERACOES']}")
        print(f"• Otimizador: {optimizer.__class__.__name__}")
        print(f"• Ponto inicial: {initial_point}")
        
        # Execução com Session
        with Session(service=service, backend=backend) as session:
            sampler = Sampler(session=session, options=options)
            
            qaoa = QAOA(
                sampler=sampler,
                optimizer=optimizer,
                reps=reps,
                initial_point=initial_point
            )
            
            print("⚡ Executando no IBM Quantum Cloud...")
            print("   ⏳ Isso pode levar alguns minutos...")
            result = qaoa.compute_minimum_eigenvalue(hamiltonian)
            
        print("✅ EXECUÇÃO CONCLUÍDA NO IBM QUANTUM!")
        return result
        
    except Exception as e:
        print(f"❌ ERRO na execução IBM Quantum: {e}")
        print("🔄 Alternando para simulação local...")
        return run_simulacao_local(hamiltonian, config)

def run_simulacao_local(hamiltonian, config):
    """
    Fallback para simulação local
    """
    print("💻 EXECUTANDO EM SIMULAÇÃO LOCAL...")
    
    reps = config['QAOA_CAMADAS']
    
    if config['OTIMIZADOR'] == 'SPSA':
        optimizer = SPSA(maxiter=config['QAOA_ITERACOES'])
    else:
        optimizer = COBYLA(maxiter=config['QAOA_ITERACOES'])
    
    if reps == 2:
        initial_point = [0.8, 0.4, 0.6, 0.3]
    else:
        initial_point = np.random.uniform(0.1, 0.9, 2 * reps)
    
    sampler = StatevectorSampler()
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point
    )
    
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    return result

# Execução principal
service = setup_ibm_quantum_service()

if PORTFOLIO_CONFIG['MODO_EXECUCAO'] == 'ibm_quantum' and service is not None:
    result = run_ibm_quantum_otimizado(hamiltonian, PORTFOLIO_CONFIG)
else:
    print("🔄 Executando em simulação local...")
    PORTFOLIO_CONFIG['MODO_EXECUCAO'] = 'simulacao'
    result = run_simulacao_local(hamiltonian, PORTFOLIO_CONFIG)

# ===========================================================================
# CÉLULA 8: ANÁLISE DE RESULTADOS PARA IBM QUANTUM
# ===========================================================================

print("\n📊 ANALISANDO RESULTADOS DA EXECUÇÃO")
print("=" * 70)

def analise_resultados_ibm(result, reference_comb, reference_val, mu, cov, budget):
    """
    Análise otimizada para resultados do IBM Quantum
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        print("❌ Nenhum resultado para analisar")
        return {"status": "ERRO"}
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    N = len(mu)
    total_combinacoes = len(list(combinations(range(N), budget)))
    prob_aleatoria = 1 / total_combinacoes
    
    print(f"📈 CONTEXTO DO PROBLEMA:")
    print(f"• Combinações totais: {total_combinacoes:,}")
    print(f"• Probabilidade aleatória: {prob_aleatoria:.6f}%")
    
    # Analisar soluções
    valid_solutions = []
    all_solutions = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != N:
            continue
            
        x = np.array([int(bit) for bit in bitstring[::-1]])  # Inverter para little-endian
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        assets = [i for i, val in enumerate(x) if val == 1]
        
        solution_data = {
            'bitstring': bitstring,
            'value': objective_value,
            'probability': prob,
            'assets': assets,
            'num_assets': num_assets
        }
        
        all_solutions.append(solution_data)
        if num_assets == budget:
            valid_solutions.append(solution_data)
    
    # Ordenar soluções
    valid_solutions.sort(key=lambda x: x['value'])
    all_solutions.sort(key=lambda x: x['probability'], reverse=True)
    
    if valid_solutions:
        best_solution = valid_solutions[0]
        
        # Verificar solução de referência
        referencia_encontrada = False
        prob_referencia = 0.0
        for sol in valid_solutions:
            if sol['assets'] == list(reference_comb):
                referencia_encontrada = True
                prob_referencia = sol['probability']
                break
        
        total_valid_prob = sum(sol['probability'] for sol in valid_solutions)
        
        print(f"\n🎯 RESULTADOS OBTIDOS:")
        print(f"• Melhor portfólio: {best_solution['assets']}")
        print(f"• Valor objetivo: {best_solution['value']:.6f}")
        print(f"• Probabilidade: {best_solution['probability']:.4f} ({best_solution['probability']*100:.2f}%)")
        
        print(f"\n📊 EFICÁCIA DO ALGORITMO:")
        print(f"• Solução de referência: {'✅ ENCONTRADA' if referencia_encontrada else '❌ NÃO ENCONTRADA'}")
        if referencia_encontrada:
            print(f"• Probabilidade na referência: {prob_referencia:.4f} ({prob_referencia*100:.2f}%)")
        print(f"• Eficiência em válidas: {total_valid_prob:.3f} ({total_valid_prob*100:.1f}%)")
        print(f"• Soluções válidas: {len(valid_solutions)}")
        
        # Top soluções
        print(f"\n🏆 MELHORES SOLUÇÕES:")
        for i, sol in enumerate(valid_solutions[:3]):
            marcador = "🎯" if sol['assets'] == list(reference_comb) else "  "
            print(f"  {marcador} {i+1}. {sol['assets']} - valor: {sol['value']:.4f} - prob: {sol['probability']:.4f}")
        
        # Estados mais prováveis (pode incluir inválidos)
        print(f"\n🔍 ESTADOS MAIS PROVÁVEIS:")
        for i, sol in enumerate(all_solutions[:3]):
            valido = "✅" if sol['num_assets'] == budget else "❌"
            print(f"  {valido} {sol['bitstring']}: {sol['probability']:.3f} (ativos: {sol['assets']})")
        
        # Avaliação para IBM Quantum
        if best_solution['probability'] > 0.05:
            avaliacao = "🎉 EXCELENTE"
            nota = "A+"
        elif best_solution['probability'] > 0.02:
            avaliacao = "✅ MUITO BOM"
            nota = "A"
        elif best_solution['probability'] > 0.01:
            avaliacao = "👍 BOM" 
            nota = "B"
        elif best_solution['probability'] > 0.005:
            avaliacao = "⚠️  RAZOÁVEL"
            nota = "C"
        elif best_solution['probability'] > prob_aleatoria:
            avaliacao = "🔴 BÁSICO"
            nota = "D"
        else:
            avaliacao = "💥 INSUFICIENTE"
            nota = "F"
        
        melhoria = best_solution['probability'] / prob_aleatoria
        
        print(f"\n🏅 AVALIAÇÃO IBM QUANTUM: {nota}")
        print(f"• {avaliacao}")
        print(f"• Melhoria sobre aleatório: {melhoria:.1f}x")
        
        return {
            'portfolio': best_solution['assets'],
            'value': best_solution['value'],
            'probability': best_solution['probability'],
            'found_reference': referencia_encontrada,
            'reference_probability': prob_referencia,
            'efficiency': total_valid_prob,
            'usability_score': nota,
            'num_valid_solutions': len(valid_solutions),
            'improvement_over_random': melhoria,
            'backend': PORTFOLIO_CONFIG['BACKEND_IBM']
        }
    
    return {"status": "SEM_SOLUCOES"}

analysis = analise_resultados_ibm(result, reference_comb, reference_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 9: RELATÓRIO EXECUTIVO IBM QUANTUM
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - IBM QUANTUM CLOUD")
print("=" * 70)

if 'portfolio' in analysis:
    print(f"\n✅ RESULTADO DA EXECUÇÃO NA NUVEM:")
    print(f"   • Backend utilizado: {analysis['backend']}")
    print(f"   • Portfólio recomendado: {analysis['portfolio']}")
    print(f"   • Valor objetivo: {analysis['value']:.6f}")
    print(f"   • Probabilidade: {analysis['probability']:.4f} ({analysis['probability']*100:.2f}%)")
    print(f"   • Nota: {analysis['usability_score']}")
    
    print(f"\n📈 DESEMPENHO NA NUVEM:")
    print(f"   • Melhoria sobre aleatório: {analysis['improvement_over_random']:.1f}x")
    print(f"   • Eficiência em válidas: {analysis['efficiency']*100:.1f}%")
    print(f"   • Soluções válidas encontradas: {analysis['num_valid_solutions']}")
    print(f"   • Referência encontrada: {'✅ SIM' if analysis['found_reference'] else '❌ NÃO'}")

print(f"\n🔧 CONFIGURAÇÃO APLICADA:")
print(f"   • Ativos: {PORTFOLIO_CONFIG['NUM_ATIVOS']} | Selecionar: {PORTFOLIO_CONFIG['NUM_SELECIONAR']}")
print(f"   • Backend: {PORTFOLIO_CONFIG['BACKEND_IBM']}")
print(f"   • Modo: {PORTFOLIO_CONFIG['MODO_EXECUCAO']}")
print(f"   • Camadas QAOA: {PORTFOLIO_CONFIG['QAOA_CAMADAS']}")
print(f"   • Iterações: {PORTFOLIO_CONFIG['QAOA_ITERACOES']}")

print(f"\n💡 PRÓXIMOS PASSOS:")
if analysis.get('usability_score', 'F') in ['A', 'A+', 'B']:
    print("1. 🎉 Sucesso! Resultados promissores obtidos")
    print("2. 🌐 Testar em hardware real (ibm_brisbane)")
    print("3. 📊 Aumentar número de shots para melhor estatística")
    print("4. 🔄 Testar com mais camadas QAOA")
else:
    print("1. 🔧 Ajustar penalidade (experimente 1.0 a 3.0)")
    print("2. 🔄 Mudar otimizador (COBYLA para SPSA ou vice-versa)")
    print("3. 📈 Aumentar número de iterações")
    print("4. 🎯 Verificar escala do Hamiltoniano")

print(f"\n🎯 EXPECTATIVAS REALISTAS PARA IBM QUANTUM:")
print("   • Probabilidade ótima: 1-5% (50-250x sobre aleatório)")
print("   • Eficiência válida: 10-30%")
print("   • Tempo execução: 2-10 minutos")

print(f"\n🌟 EXECUÇÃO CONCLUÍDA - PRONTO PARA PRÓXIMOS EXPERIMENTOS!")