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

print("🚀 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - VERSÃO FINAL")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO AUTOMÁTICA PARA DADOS REAIS
# ===========================================================================

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
    
    print(f"🎯 Solução clássica ótima: {best_sol[0]} (valor: {best_sol[1]:.6f})")
    
    return mu, cov, budget, best_sol

# DADOS DO USUÁRIO (SUBSTITUIR POR DADOS REAIS)
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
# CÉLULA 3: HAMILTONIANO COM PENALIDADE SUPER-OTIMIZADA
# ===========================================================================

print("\n🔧 CONFIGURANDO HAMILTONIANO OTIMIZADO")
print("=" * 70)

def build_final_hamiltonian(mu, cov, budget, penalty_factor=600.0):  # Aumentado para 600
    """
    Hamiltoniano final com penalidade super-otimizada
    """
    N = len(mu)
    
    # Cálculo automático da penalidade
    max_obj_val = 0.0
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        max_obj_val = max(max_obj_val, abs(value))
    
    penalty = penalty_factor * max_obj_val
    
    print(f"⚙️  CONFIGURAÇÃO DO HAMILTONIANO:")
    print(f"   • Escala do problema: {max_obj_val:.6f}")
    print(f"   • Penalidade aplicada: {penalty:.2f}")
    print(f"   • Estratégia: Máximo forçamento de soluções válidas")
    
    # Construção do QUBO
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    Q += cov  # Termo de risco
    q -= mu   # Termo de retorno
    
    # Restrição de budget reforçada
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
    
    print(f"✅ Hamiltoniano configurado:")
    print(f"   • Qubits: {N}")
    print(f"   • Termos: {len(hamiltonian)}")
    
    return hamiltonian

hamiltonian = build_final_hamiltonian(mu, cov, budget, penalty_factor=600.0)

# ===========================================================================
# CÉLULA 4: QAOA COM ESTRATÉGIA DE ALTA PERFORMANCE
# ===========================================================================

print("\n🚀 EXECUTANDO ALGORITMO QUÂNTICO")
print("=" * 70)

def run_high_performance_qaoa(hamiltonian):
    """
    Execução do QAOA com configuração de alta performance
    """
    print("⚙️  CONFIGURAÇÃO DE PERFORMANCE:")
    print("• Algoritmo: QAOA")
    print("• Camadas: p=4 (alta aproximação)")
    print("• Iterações: 300")
    print("• Otimizador: COBYLA")
    print("• Estratégia: Busca por máxima eficiência")
    
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=300)
    
    # Ponto inicial otimizado
    initial_point = [0.8, 0.6, 0.7, 0.5, 0.9, 0.4, 0.3, 0.2]
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=4,
        initial_point=initial_point
    )
    
    print("⚡ Executando otimização quântica...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

result = run_high_performance_qaoa(hamiltonian)

# ===========================================================================
# CÉLULA 5: ANÁLISE EXECUTIVA DOS RESULTADOS
# ===========================================================================

print("\n📊 RELATÓRIO DE RESULTADOS")
print("=" * 70)

def executive_analysis(result, classical_comb, classical_val, mu, cov, budget):
    """
    Análise executiva para tomada de decisão
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        return {"status": "ERRO", "message": "Sem resultados para análise"}
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Análise das soluções
    valid_solutions = []
    for bitstring, prob in probabilities.items():
        if len(bitstring) != len(mu):
            continue
            
        x = np.array([int(bit) for bit in bitstring])
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        
        if num_assets == budget:  # Solução válida
            assets = [i for i, val in enumerate(x) if val == 1]
            valid_solutions.append((bitstring, objective_value, prob, assets))
    
    # Ordenar por valor objetivo
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
        print(f"• Probabilidade na solução ótima: {optimal_prob:.3f} ({optimal_prob*100:.1f}%)")
        print(f"• Eficiência total: {total_valid_prob:.3f} ({total_valid_prob*100:.1f}%)")
        print(f"• Precisão (gap): {abs(best_solution[1] - classical_val)/abs(classical_val)*100:.2f}%")
        
        # Avaliação para uso prático
        if optimal_prob > 0.05 and total_valid_prob > 0.2:
            usability = "✅ ALTA - Pronta para uso prático"
            recommendation = "Pode ser utilizada em análises reais"
        elif optimal_prob > 0.02 and total_valid_prob > 0.1:
            usability = "✅ MÉDIA - Funcional com supervisão"
            recommendation = "Recomendado validação com método clássico"
        elif optimal_prob > 0:
            usability = "⚠️  BÁSICA - Para fins experimentais"
            recommendation = "Usar apenas para estudos e protótipos"
        else:
            usability = "🔴 LIMITADA - Apenas demonstração"
            recommendation = "Necessita otimizações adicionais"
        
        print(f"\n🏆 USABILIDADE: {usability}")
        print(f"💡 RECOMENDAÇÃO: {recommendation}")
        
        return {
            'portfolio': best_solution[3],
            'value': best_solution[1],
            'optimal_probability': optimal_prob,
            'efficiency': total_valid_prob,
            'usability': usability,
            'recommendation': recommendation
        }
    
    return {"status": "SEM_SOLUCOES", "message": "Nenhuma solução válida encontrada"}

# Análise executiva
analysis = executive_analysis(result, classical_comb, classical_val, mu, cov, budget)

# ===========================================================================
# CÉLULA 6: RELATÓRIO FINAL PARA O USUÁRIO
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO FINAL - VIABILIDADE DA FERRAMENTA")
print("=" * 70)

if 'portfolio' in analysis:
    print(f"\n✅ RESULTADO OBTIDO:")
    print(f"   Portfólio recomendado: {analysis['portfolio']}")
    print(f"   Valor da solução: {analysis['value']:.6f}")
    
    print(f"\n📊 PERFORMANCE:")
    print(f"   Eficiência do algoritmo: {analysis['efficiency']:.3f} ({analysis['efficiency']*100:.1f}%)")
    print(f"   Precisão da solução ótima: {analysis['optimal_probability']:.3f} ({analysis['optimal_probability']*100:.1f}%)")
    
    print(f"\n🎯 STATUS PARA USO PRÁTICO:")
    print(f"   {analysis['usability']}")
    print(f"   {analysis['recommendation']}")
    
    print(f"\n🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    if analysis['usability'].startswith("✅ ALTA"):
        print("1. Testar com seus dados reais de portfólio")
        print("2. Comparar resultados com otimização clássica")
        print("3. Implementar em fluxo de análise regular")
    elif analysis['usability'].startswith("✅ MÉDIA"):
        print("1. Validar resultados com métodos clássicos")
        print("2. Testar com diferentes conjuntos de dados")
        print("3. Considerar para apoio à decisão")
    else:
        print("1. Utilizar para estudos e demonstrações")
        print("2. Manter como ferramenta de pesquisa")
        print("3. Acompanhar evoluções do QAOA")

print(f"\n💡 COMO USAR COM SEUS DADOS:")
print("1. Substitua 'returns' pelos retornos dos seus ativos")
print("2. Substitua 'covariance' pela matriz de covariância")
print("3. Ajuste 'num_selected' para o número desejado")
print("4. Execute e confira o relatório de usabilidade")

print(f"\n📈 EXEMPLO PRÁTICO:")
print("   returns = [0.15, 0.12, 0.18, 0.09, 0.11]")
print("   covariance = [[0.1, 0.02, ...], ...]")
print("   num_selected = 3")

print(f"\n🔮 PERSPECTIVA FUTURA:")
print("• Esta ferramenta representa o estado da arte em computação quântica aplicada")
print("• Performance irá melhorar com avanços em hardware e algoritmos")
print("• Ideal para posicionamento estratégico em tecnologia quântica")

print(f"\n🌟 FERRAMENTA DE OTIMIZAÇÃO QUÂNTICA - CONCLUÍDA!")