# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO
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
import warnings
warnings.filterwarnings('ignore')

print("🚀 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - VERSÃO ESCALÁVEL")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO FLEXÍVEL - ALTERE AQUI!
# ===========================================================================

# 🎯 CONFIGURAÇÃO DO PORTFÓLIO - ALTERE ESTES VALORES!
PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 8,              # ⚡ Número total de ativos (4-8 para simulação)
    'NUM_SELECIONAR': 4,          # ⚡ Quantos ativos selecionar
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
    
    # ⚙️ CONFIGURAÇÃO DO ALGORITMO
    'PENALIDADE_FACTOR': 800.0,   # ⚡ 200-800 (maior = mais soluções válidas)
    'QAOA_CAMADAS': 4,            # ⚡ 2-4 (mais camadas = mais preciso)
    'QAOA_ITERACOES': 300,        # ⚡ 100-300
    'OTIMIZADOR': 'SPSA',       # ⚡ 'COBYLA' ou 'SPSA'
    
    # 🎯 MODO DE EXECUÇÃO
    'MODO_EXECUCAO': 'simulacao'  # ⚡ 'simulacao' ou 'ibm_quantum'
}

# ===========================================================================
# CÉLULA 3: GERADOR DE DADOS DE PORTFÓLIO
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
# CÉLULA 4: HAMILTONIANO OTIMIZADO PARA DIFERENTES TAMANHOS
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
# CÉLULA 5: QAOA ESCALÁVEL - PRESETS PARA DIFERENTES CONFIGURAÇÕES +  PONTO INICIAL CORRETO
# ===========================================================================

print("\n🚀 EXECUTANDO ALGORITMO QUÂNTICO ESCALÁVEL")
print("=" * 70)

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

def run_scalable_qaoa(hamiltonian, config):
    """
    Execução do QAOA com ponto inicial CORRETO
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
    
    print("⚙️  CONFIGURAÇÃO DE PERFORMANCE:")
    print(f"• Algoritmo: QAOA")
    print(f"• Camadas: p={qaoa_config['reps']}")
    print(f"• Iterações: {qaoa_config['max_iter']}")
    print(f"• Otimizador: {qaoa_config['optimizer'].__name__}")
    print(f"• Qubits: {num_assets}")
    print(f"• Parâmetros esperados: {2 * qaoa_config['reps']}")
    
    sampler = StatevectorSampler()
    optimizer = qaoa_config['optimizer'](maxiter=qaoa_config['max_iter'])
    
    # 🔥 CORREÇÃO CRÍTICA: Ponto inicial com dimensão CORRETA
    num_params = 2 * qaoa_config['reps']
    
    # Pontos iniciais pré-otimizados por número de parâmetros
    initial_points = {
        4: [0.7, 0.5, 0.8, 0.3],                    # p=2
        6: [0.7, 0.5, 0.8, 0.3, 0.6, 0.4],          # p=3  
        8: [0.7, 0.5, 0.8, 0.3, 0.6, 0.4, 0.9, 0.2] # p=4
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
    
    print("⚡ Executando otimização quântica...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

# Execução com a configuração atual
result = run_scalable_qaoa(hamiltonian, PORTFOLIO_CONFIG)

# ===========================================================================
# CÉLULA 6: ANÁLISE ESCALÁVEL DOS RESULTADOS
# ===========================================================================

print("\n📊 RELATÓRIO DE RESULTADOS ESCALÁVEL")
print("=" * 70)

def analyze_scalable_results(result, classical_comb, classical_val, mu, cov, budget):
    """
    Análise escalável para diferentes tamanhos de problema
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
            'num_valid_solutions': len(valid_solutions)
        }
    
    return {"status": "SEM_SOLUCOES", "message": "Nenhuma solução válida encontrada"}

# Análise dos resultados
analysis = analyze_scalable_results(result, classical_comb, classical_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 7: RELATÓRIO EXECUTIVO COMPLETO
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - PORTFÓLIO QUÂNTICO")
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

print(f"\n💡 PRESETS RECOMENDADOS:")

print(f"\n🖥️  PARA MÁQUINA COMUM:")
print("   • 4-6 ativos: PENALIDADE=300, CAMADAS=2, ITERAÇÕES=150")
print("   • 7-8 ativos: PENALIDADE=400, CAMADAS=3, ITERAÇÕES=200") 
print("   • 9-10 ativos: PENALIDADE=500, CAMADAS=3, ITERAÇÕES=250")

print(f"\n☁️  PARA IBM QUANTUM (FUTURO):")
print("   • 4-6 ativos: PENALIDADE=400, CAMADAS=2, ITERAÇÕES=100")
print("   • 7-8 ativos: PENALIDADE=500, CAMADAS=2, ITERAÇÕES=150")

print(f"\n🎲 EXEMPLOS DE CONFIGURAÇÃO PARA TESTE:")

print(f"\n📈 PORTFÓLIO PEQUENO (Rápido):")
print("   NUM_ATIVOS = 4, NUM_SELECIONAR = 2")
print("   PENALIDADE_FACTOR = 300, QAOA_CAMADAS = 2")

print(f"\n📊 PORTFÓLIO MÉDIO (Balanceado):")
print("   NUM_ATIVOS = 6, NUM_SELECIONAR = 3") 
print("   PENALIDADE_FACTOR = 400, QAOA_CAMADAS = 3")

print(f"\n📉 PORTFÓLIO GRANDE (Avançado):")
print("   NUM_ATIVOS = 8, NUM_SELECIONAR = 4")
print("   PENALIDADE_FACTOR = 500, QAOA_CAMADAS = 3")

print(f"\n🌟 FERRAMENTA DE OTIMIZAÇÃO QUÂNTICA - PRONTA PARA EXPERIMENTOS!")