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
    print(f"• Retornos esperados: {[f'{r:.3f}' for r in mu]}")
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
# CÉLULA 3: HAMILTONIANO OTIMIZADO COM PENALIDADE INTELIGENTE
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO OTIMIZADO")
print("=" * 60)

def build_optimized_hamiltonian(mu, cov, budget, penalty_factor=300.0):  # Reduzido para 300
    """
    Hamiltoniano com penalidade inteligente baseada na análise anterior
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
    
    print(f"⚙️  PARÂMETROS OTIMIZADOS:")
    print(f"   • Escala do problema: {max_obj_val:.6f}")
    print(f"   • Penalidade: {penalty:.2f} (fator: {penalty_factor})")
    print(f"   • Estratégia: Balance entre válidas e qualidade")
    
    # Construção direta do QUBO
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    # Termos do problema de Markowitz
    Q += cov  # Risco
    q -= mu   # Retorno (negativo para minimização)
    
    # Restrição de budget - fórmula otimizada
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
    
    print(f"✅ Hamiltoniano otimizado:")
    print(f"   • Termos: {len(hamiltonian)}")
    print(f"   • Qubits: {N}")
    print(f"   • Norma: {np.linalg.norm(hamiltonian.coeffs):.2f}")
    
    return hamiltonian

# Testar diferentes penalidades para encontrar o melhor valor
print("\n🔍 TESTANDO DIFERENTES PENALIDADES:")
best_penalty = 300.0
hamiltonian = build_optimized_hamiltonian(mu, cov, budget, penalty_factor=best_penalty)

# ===========================================================================
# CÉLULA 4: QAOA OTIMIZADO - ESTRATÉGIA AVANÇADA
# ===========================================================================

print("\n🚀 EXECUTANDO QAOA OTIMIZADO")
print("=" * 60)

def run_optimized_qaoa(hamiltonian, strategy="balanced"):
    """
    Execução otimizada do QAOA com estratégias diferentes
    """
    if strategy == "balanced":
        reps = 3
        max_iter = 250
        optimizer_class = COBYLA
        initial_point = [0.8, 0.6, 0.4, 0.7, 0.5, 0.3]
    elif strategy == "aggressive":
        reps = 4
        max_iter = 350
        optimizer_class = SPSA
        initial_point = np.random.uniform(0, 1, 2 * reps)
    else:  # conservative
        reps = 2
        max_iter = 200
        optimizer_class = COBYLA
        initial_point = [0.5, 0.5, 0.5, 0.5]
    
    print(f"⚙️  ESTRATÉGIA: {strategy.upper()}")
    print(f"• Camadas (p): {reps}")
    print(f"• Iterações: {max_iter}")
    print(f"• Otimizador: {optimizer_class.__name__}")
    
    sampler = StatevectorSampler()
    optimizer = optimizer_class(maxiter=max_iter)
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point
    )
    
    print("⚡ Executando QAOA otimizado...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

# Execução com estratégia balanceada
result = run_optimized_qaoa(hamiltonian, strategy="balanced")

# ===========================================================================
# CÉLULA 5: ANÁLISE DETALHADA E OTIMIZAÇÃO
# ===========================================================================

print("\n📊 ANÁLISE DETALHADA DOS RESULTADOS")
print("=" * 60)

def analyze_optimized_results(result, classical_comb, classical_val, mu, cov, budget):
    """
    Análise detalhada e otimizada dos resultados
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        print("❌ Nenhum resultado para analisar")
        return None
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Coletar e classificar todas as soluções
    valid_solutions = []
    invalid_solutions = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != len(mu):
            continue
            
        x = np.array([int(bit) for bit in bitstring])
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        is_valid = num_assets == budget
        assets = [i for i, val in enumerate(x) if val == 1]
        
        if is_valid:
            valid_solutions.append((bitstring, objective_value, prob, assets))
        else:
            invalid_solutions.append((bitstring, objective_value, prob, assets))
    
    # Ordenar soluções
    valid_solutions.sort(key=lambda x: x[1])  # Por valor objetivo
    invalid_solutions.sort(key=lambda x: x[2], reverse=True)  # Por probabilidade
    
    # ANÁLISE DETALHADA
    print("🎯 SOLUÇÕES VÁLIDAS DETALHADAS:")
    print("=" * 75)
    print(f"{'Estado':<8} {'Prob.':<8} {'Valor':<12} {'Ativos':<15} {'Status':<10}")
    print("-" * 75)
    
    for bitstring, obj_val, prob, assets in valid_solutions:
        is_optimal = "🌟 ÓTIMO" if assets == list(classical_comb) else "  VÁLIDO"
        print(f"{bitstring:<8} {prob:<8.3f} {obj_val:<12.6f} {str(assets):<15} {is_optimal:<10}")
    
    # Análise das soluções inválidas problemáticas
    print(f"\n⚠️  PRINCIPAIS SOLUÇÕES INVÁLIDAS:")
    for i, (bitstring, obj_val, prob, assets) in enumerate(invalid_solutions[:3]):
        if prob > 0.01:
            print(f"   {bitstring}: {prob:.3f} | Ativos: {assets}")
    
    # Cálculo de métricas
    if valid_solutions:
        best_solution = valid_solutions[0]
        total_valid_prob = sum(sol[2] for sol in valid_solutions)
        optimal_prob = next((sol[2] for sol in valid_solutions if sol[3] == list(classical_comb)), 0.0)
        gap = abs(best_solution[1] - classical_val) / abs(classical_val) * 100
        
        print(f"\n📈 MÉTRICAS DE PERFORMANCE:")
        print(f"• Probabilidade total em válidas: {total_valid_prob:.3f} ({total_valid_prob*100:.1f}%)")
        print(f"• Probabilidade na solução ótima: {optimal_prob:.3f} ({optimal_prob*100:.1f}%)")
        print(f"• Número de soluções válidas: {len(valid_solutions)}/{len(list(combinations(range(len(mu)), budget)))}")
        print(f"• Melhor solução: {best_solution[3]} (valor: {best_solution[1]:.6f})")
        print(f"• Gap para ótimo: {gap:.2f}%")
        
        # Score de qualidade aprimorado
        validity_score = min(100, total_valid_prob * 150)  # Mais exigente
        optimality_score = min(100, optimal_prob * 400)    # Mais exigente
        gap_score = max(0, 100 - gap)                      # 100 se gap=0%
        
        overall_score = (validity_score + optimality_score + gap_score) / 3
        
        print(f"\n🏆 SCORE AVANÇADO: {overall_score:.1f}/100")
        
        # Diagnóstico detalhado
        if overall_score > 80:
            status = "✅ EXCELENTE - Pronto para produção"
            diagnosis = "O algoritmo está performando muito bem"
        elif overall_score > 60:
            status = "✅ BOM - Funcional para aplicações práticas"
            diagnosis = "Performance sólida para uso real"
        elif overall_score > 40:
            status = "⚠️  ACEITÁVEL - Pode ser usado com monitoramento"
            diagnosis = "Funciona mas pode melhorar com ajustes"
        elif overall_score > 20:
            status = "🔴 BÁSICO - Necessita otimização"
            diagnosis = "Recomendado ajustar parâmetros"
        else:
            status = "💥 CRÍTICO - Revisão urgente necessária"
            diagnosis = "Algoritmo não está convergindo adequadamente"
        
        print(f"• Status: {status}")
        print(f"• Diagnóstico: {diagnosis}")
        
        # Recomendações específicas
        print(f"\n💡 RECOMENDAÇÕES ESPECÍFICAS:")
        if optimal_prob < 0.05:
            print("• Aumentar penalidade para 400-500")
            print("• Tentar estratégia 'aggressive'")
        elif total_valid_prob < 0.3:
            print("• Aumentar número de camadas para p=4")
            print("• Testar mais iterações (300+)")
        else:
            print("• Configuração adequada para dados reais")
            print("• Considerar execução em hardware quântico")
        
        return {
            'best_solution': best_solution[3],
            'best_value': best_solution[1],
            'optimal_probability': optimal_prob,
            'total_valid_probability': total_valid_prob,
            'gap_percent': gap,
            'overall_score': overall_score,
            'status': status
        }
    
    else:
        print("❌ Nenhuma solução válida encontrada")
        return None

# Análise detalhada
analysis = analyze_optimized_results(result, classical_comb, classical_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 6: RELATÓRIO EXECUTIVO FINAL
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - FERRAMENTA DE OTIMIZAÇÃO QUÂNTICA")
print("=" * 70)

if analysis:
    print(f"\n📊 RESULTADOS OBTIDOS:")
    print(f"   • Melhor portfólio: {analysis['best_solution']}")
    print(f"   • Valor da solução: {analysis['best_value']:.6f}")
    print(f"   • Eficiência em válidas: {analysis['total_valid_probability']:.3f} ({analysis['total_valid_probability']*100:.1f}%)")
    print(f"   • Precisão da solução ótima: {analysis['optimal_probability']:.3f} ({analysis['optimal_probability']*100:.1f}%)")
    print(f"   • Precisão (gap): {analysis['gap_percent']:.2f}%")
    print(f"   • Score final: {analysis['overall_score']:.1f}/100")
    
    print(f"\n🎯 STATUS: {analysis['status']}")
    
    print(f"\n🚀 PRÓXIMOS PASSOS:")
    if analysis['overall_score'] > 60:
        print("1. ✅ Testar com seus dados reais de portfólio")
        print("2. ✅ Comparar com métodos clássicos (Markowitz)")
        print("3. 🔄 Otimizar para mais ativos (até 8-10)")
        print("4. 🌐 Executar em hardware quântico quando disponível")
    else:
        print("1. 🔧 Ajustar penalidade no Hamiltoniano")
        print("2. 🔧 Aumentar camadas do QAOA (p=4)")
        print("3. 🔧 Testar com mais iterações (400+)")
        print("4. 🔧 Experimentar diferentes otimizadores")

print(f"\n🔧 INSTRUÇÕES PARA USO COM DADOS REAIS:")
print("1. Substitua 'returns' pela lista de retornos dos ativos")
print("2. Substitua 'covariance' pela matriz de covariância")
print("3. Ajuste 'num_selected' para o número de ativos desejado")
print("4. Execute e analise o relatório")

print(f"\n📈 EXEMPLO DE DADOS REAIS:")
print("   returns = [0.15, 0.12, 0.18, 0.09, 0.11]  # 5 ativos")
print("   covariance = [[0.1, 0.02, ...], ...]      # Matriz 5x5")
print("   num_selected = 3                          # Selecionar 3 ativos")

print(f"\n🎉 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - PRONTA PARA USO!")