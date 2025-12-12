# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO
# ===========================================================================

import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

print("✅ Bibliotecas importadas com sucesso!")

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO PRÁTICA PARA DADOS REAIS
# ===========================================================================

print("🎯 CONFIGURAÇÃO PRÁTICA - OTIMIZAÇÃO DE PORTFÓLIO")
print("=" * 60)

def setup_portfolio_problem(returns, covariance, num_assets_to_select):
    """
    Configuração prática para dados reais de portfólio
    """
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"📊 CONFIGURAÇÃO DO PROBLEMA:")
    print(f"• Número de ativos: {N}")
    print(f"• Ativos a selecionar: {budget}")
    print(f"• Retornos esperados: {mu}")
    print(f"• Matriz de covariância: {cov.shape}")
    
    # Análise rápida das soluções
    print(f"\n🔍 ANÁLISE RÁPIDA DAS SOLUÇÕES:")
    solutions = []
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        solutions.append((comb, value, x))
    
    solutions.sort(key=lambda x: x[1])
    best_sol = solutions[0]
    worst_sol = solutions[-1]
    
    print(f"🎯 Melhor portfólio: {best_sol[0]} (valor: {best_sol[1]:.6f})")
    print(f"📉 Pior portfólio: {worst_sol[0]} (valor: {worst_sol[1]:.6f})")
    
    return mu, cov, budget, best_sol

# DADOS DE EXEMPLO (substitua por dados reais)
returns = [0.12, 0.10, 0.14, 0.07]
covariance = [
    [0.1, 0.02, 0.01, 0.03],
    [0.02, 0.15, 0.05, 0.02],
    [0.01, 0.05, 0.2, 0.04],
    [0.03, 0.02, 0.04, 0.1]
]
num_selected = 2

mu, cov, budget, classical_solution = setup_portfolio_problem(
    returns, covariance, num_selected
)

classical_comb, classical_val, classical_vec = classical_solution

# ===========================================================================
# CÉLULA 3: HAMILTONIANO PRÁTICO E ROBUSTO
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO PRÁTICO")
print("=" * 60)

def build_practical_hamiltonian(mu, cov, budget, penalty_factor=800.0):
    """
    Hamiltoniano prático e robusto para dados reais
    """
    N = len(mu)
    
    # Penalidade automática baseada nos dados
    max_obj_val = 0.0
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        max_obj_val = max(max_obj_val, abs(value))
    
    penalty = penalty_factor * max_obj_val
    
    print(f"⚙️  PARÂMETROS AUTOMÁTICOS:")
    print(f"   • Escala do problema: {max_obj_val:.6f}")
    print(f"   • Penalidade: {penalty:.2f}")
    
    # Construção direta do QUBO
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    # Termos do problema de Markowitz
    Q += cov  # Risco
    q -= mu   # Retorno (negativo para minimização)
    
    # Restrição de budget
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * budget)
        q[i] += 2 * penalty * budget
    
    for i in range(N):
        for j in range(i+1, N):
            Q[i, j] += 2 * penalty
            Q[j, i] += 2 * penalty
    
    # Conversão para Ising
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
    
    # Operador Pauli
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
    print(f"   • Termos: {len(hamiltonian)}")
    print(f"   • Qubits: {N}")
    
    return hamiltonian

hamiltonian = build_practical_hamiltonian(mu, cov, budget, penalty_factor=800.0)

# ===========================================================================
# CÉLULA 4: QAOA PRÁTICO - CONFIGURAÇÃO ROBUSTA
# ===========================================================================

print("\n🚀 EXECUTANDO QAOA PRÁTICO")
print("=" * 60)

def run_practical_qaoa(hamiltonian, reps=3, max_iter=200):
    """
    Execução prática do QAOA para dados reais
    """
    print("⚙️  CONFIGURAÇÃO DO QAOA:")
    print(f"• Camadas (p): {reps}")
    print(f"• Iterações: {max_iter}")
    print(f"• Otimizador: COBYLA")
    
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=max_iter)
    
    # Ponto inicial robusto
    if reps == 2:
        initial_point = [0.7, 0.5, 0.8, 0.3]
    elif reps == 3:
        initial_point = [0.7, 0.5, 0.8, 0.3, 0.6, 0.4]
    else:
        initial_point = np.random.uniform(0, 1, 2 * reps)
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point
    )
    
    print("⚡ Executando...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

# Execução prática
result = run_practical_qaoa(hamiltonian, reps=3, max_iter=200)

# ===========================================================================
# CÉLULA 5: ANÁLISE PRÁTICA DOS RESULTADOS
# ===========================================================================

print("\n📊 ANÁLISE PRÁTICA DOS RESULTADOS")
print("=" * 60)

def analyze_practical_results(result, classical_comb, classical_val, mu, cov, budget):
    """
    Análise prática e robusta dos resultados
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        print("❌ Nenhum resultado para analisar")
        return
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Coletar todas as soluções
    valid_solutions = []
    all_solutions = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != len(mu):
            continue
            
        x = np.array([int(bit) for bit in bitstring])
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        is_valid = num_assets == budget
        assets = [i for i, val in enumerate(x) if val == 1]
        
        all_solutions.append((bitstring, objective_value, prob, assets, is_valid))
        
        if is_valid:
            valid_solutions.append((bitstring, objective_value, prob, assets))
    
    # Análise das soluções válidas
    if valid_solutions:
        valid_solutions.sort(key=lambda x: x[1])  # Ordenar por qualidade
        best_solution = valid_solutions[0]
        
        print("🎯 SOLUÇÕES VÁLIDAS ENCONTRADAS:")
        print("=" * 70)
        for i, (bitstring, obj_val, prob, assets) in enumerate(valid_solutions[:6]):
            is_optimal = "🌟" if assets == list(classical_comb) else "  "
            print(f"{is_optimal} {bitstring}: {prob:.3f} | Valor: {obj_val:.6f} | Ativos: {assets}")
        
        # Métricas práticas
        total_valid_prob = sum(sol[2] for sol in valid_solutions)
        optimal_prob = next((sol[2] for sol in valid_solutions if sol[3] == list(classical_comb)), 0.0)
        gap = abs(best_solution[1] - classical_val) / abs(classical_val) * 100
        
        print(f"\n📈 MÉTRICAS PRÁTICAS:")
        print(f"• Probabilidade em soluções válidas: {total_valid_prob:.3f} ({total_valid_prob*100:.1f}%)")
        print(f"• Probabilidade na solução ótima: {optimal_prob:.3f} ({optimal_prob*100:.1f}%)")
        print(f"• Melhor solução encontrada: {best_solution[3]} (valor: {best_solution[1]:.6f})")
        print(f"• Gap para solução clássica: {gap:.2f}%")
        
        # Avaliação prática
        if gap < 1.0 and optimal_prob > 0.05:
            status = "✅ EXCELENTE - Pronto para uso prático"
        elif gap < 5.0 and total_valid_prob > 0.3:
            status = "✅ BOM - Funcional para aplicações"
        elif gap < 20.0:
            status = "⚠️  ACEITÁVEL - Pode ser usado com cautela"
        else:
            status = "🔴 NECESSITA AJUSTES"
        
        print(f"• Status: {status}")
        
        return {
            'best_solution': best_solution[3],
            'best_value': best_solution[1],
            'optimal_probability': optimal_prob,
            'total_valid_probability': total_valid_prob,
            'gap_percent': gap,
            'status': status
        }
    
    else:
        print("❌ Nenhuma solução válida encontrada")
        return None

# Análise prática
analysis = analyze_practical_results(result, classical_comb, classical_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 6: RELATÓRIO FINAL PRÁTICO
# ===========================================================================

print("\n🏁 RELATÓRIO FINAL - FERRAMENTA PRÁTICA")
print("=" * 60)

if analysis:
    print(f"🎯 RESULTADO: {analysis['status']}")
    print(f"📊 Melhor portfólio quântico: {analysis['best_solution']}")
    print(f"💰 Valor da solução: {analysis['best_value']:.6f}")
    print(f"📈 Eficiência do algoritmo: {analysis['total_valid_probability']:.3f}")
    print(f"🎯 Precisão: {analysis['gap_percent']:.2f}% de gap")
    
    print(f"\n💡 PRÓXIMOS PASSOS:")
    if analysis['status'].startswith("✅"):
        print("• A ferramenta está PRONTA para uso com dados reais")
        print("• Teste com seus próprios dados de portfólio")
        print("• Considere executar em hardware quântico real")
    else:
        print("• Ajuste a penalidade no Hamiltoniano")
        print("• Aumente o número de camadas do QAOA")
        print("• Teste com mais iterações")

print(f"\n🔧 PARA USO COM DADOS REAIS:")
print("1. Substitua 'returns' e 'covariance' por seus dados")
print("2. Ajuste 'num_selected' para o número de ativos desejado")
print("3. Execute e analise os resultados")

print(f"\n🎉 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - PRONTA!")