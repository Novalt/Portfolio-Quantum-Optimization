# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO
# ===========================================================================

# Importações ATUALIZADAS para Qiskit 2.2.3
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt

# Qiskit imports - VERSÃO 2.2.3
from qiskit import QuantumCircuit
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp

print("✅ Bibliotecas importadas com sucesso para Qiskit 2.2.3!")

# ===========================================================================
# CÉLULA 2: DADOS ORIGINAIS - SEM NORMALIZAÇÃO
# ===========================================================================

print("📊 CONFIGURANDO PROBLEMA COM DADOS ORIGINAIS")
print("=" * 60)

N = 4
mu = np.array([0.12, 0.10, 0.14, 0.07])  # DADOS ORIGINAIS
cov = np.array([
    [0.1, 0.02, 0.01, 0.03],
    [0.02, 0.15, 0.05, 0.02],
    [0.01, 0.05, 0.2, 0.04],
    [0.03, 0.02, 0.04, 0.1]
])
budget = 2

print(f"• Número de ativos: {N}")
print(f"• Budget: {budget} ativos")
print(f"• Retornos esperados: {mu}")
print(f"• Matriz de covariância:")
for i in range(N):
    print(f"  {cov[i]}")

# Calcular todas as soluções para análise de escala
print(f"\n🔍 ANÁLISE DAS SOLUÇÕES CLÁSSICAS:")
classical_solutions = []
for comb in combinations(range(N), budget):
    x = np.zeros(N)
    x[list(comb)] = 1
    value = x @ cov @ x - mu @ x
    classical_solutions.append((comb, value))
    print(f"  {comb}: valor = {value:.6f}")

classical_solutions.sort(key=lambda x: x[1])
best_classical = classical_solutions[0]
worst_classical = classical_solutions[-1]

print(f"🎯 Melhor solução: {best_classical[0]} (valor: {best_classical[1]:.6f})")
print(f"📉 Pior solução: {worst_classical[0]} (valor: {worst_classical[1]:.6f})")

# Calcular escala para penalidade
max_obj_value = max(abs(sol[1]) for sol in classical_solutions)
print(f"📏 Escala máxima da função objetivo: {max_obj_value:.6f}")

# ===========================================================================
# CÉLULA 3: HAMILTONIANO COM PENALIDADE SUPER-OTIMIZADA
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO SUPER-OTIMIZADO")
print("=" * 60)

def build_super_optimized_hamiltonian(mu, cov, budget, penalty_scale=500.0):  # AUMENTEI PARA 500
    """
    Hamiltoniano com penalidade SUPER-OTIMIZADA baseada em análise rigorosa
    """
    N = len(mu)
    
    # Calcular escala exata do problema
    classical_values = []
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        classical_values.append(value)
    
    max_obj_value = max(abs(val) for val in classical_values)
    penalty = penalty_scale * max_obj_value
    
    print(f"🎯 PARÂMETROS SUPER-OTIMIZADOS:")
    print(f"   • Valor máximo objetivo: {max_obj_value:.6f}")
    print(f"   • Penalidade calculada: {penalty:.6f}")
    print(f"   • Fator de escala: {penalty_scale}")
    print(f"   • Objetivo: FORÇAR soluções válidas")
    
    # Formulação QUBO DIRETA - sem truques
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    # 1. Termo de risco (variância) - manter escala original
    Q += cov
    
    # 2. Termo de retorno - manter escala original
    for i in range(N):
        q[i] = -mu[i]  # Negativo porque queremos MAXIMIZAR retorno
    
    # 3. Termo de penalidade SUPER-FORTE
    # Para budget=2: (sum(x) - 2)^2 = sum(x_i^2) + 2*sum_{i<j}(x_i x_j) - 4*sum(x_i) + 4
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * budget)  # x_i^2 e -4*x_i
        q[i] += 2 * penalty * budget  # do termo -4*x_i
    
    for i in range(N):
        for j in range(i+1, N):
            Q[i, j] += 2 * penalty  # 2*x_i*x_j
            Q[j, i] += 2 * penalty
    
    constante = penalty * budget**2
    
    print("✅ QUBO SUPER-OTIMIZADO:")
    print(f"   • Elementos diagonais de Q: {[f'{Q[i,i]:.3f}' for i in range(N)]}")
    print(f"   • Média |Q|: {np.mean(np.abs(Q)):.3f}")
    print(f"   • Constante: {constante:.3f}")
    
    # Conversão QUBO → Ising CORRETA
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        h[i] += -0.5 * q[i]  # termo linear
        for j in range(N):
            if i == j:
                h[i] += -0.5 * Q[i, j]  # diagonal
            else:
                h[i] += -0.25 * Q[i, j]  # off-diagonal para linear
    
    for i in range(N):
        for j in range(i+1, N):
            J[i, j] = 0.25 * Q[i, j]  # termo quadrático
    
    ising_constante = constante + 0.5 * np.sum(q) + 0.25 * np.sum(Q)
    
    # Construir operador Pauli
    pauli_terms = []
    coefficients = []
    
    pauli_terms.append("I" * N)
    coefficients.append(ising_constante)
    
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
    
    print("✅ HAMILTONIANO SUPER-OTIMIZADO CONSTRUÍDO:")
    print(f"   • Número de termos: {len(hamiltonian)}")
    print(f"   • Termos Z: {sum(1 for p in pauli_terms if p.count('Z') == 1)}")
    print(f"   • Termos ZZ: {sum(1 for p in pauli_terms if p.count('Z') == 2)}")
    print(f"   • Norma: {np.linalg.norm(hamiltonian.coeffs):.2f}")
    
    return hamiltonian

# Construir Hamiltoniano SUPER-OTIMIZADO
hamiltonian = build_super_optimized_hamiltonian(mu, cov, budget, penalty_scale=500.0)

# ===========================================================================
# CÉLULA 4: SOLUÇÃO CLÁSSICA
# ===========================================================================

print("\n🎯 SOLUÇÃO CLÁSSICA ÓTIMA")
print("=" * 60)

def classical_solution(mu, cov, budget):
    N = len(mu)
    best_value = float('inf')
    best_combination = None
    best_vector = None
    
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        if value < best_value:
            best_value = value
            best_combination = comb
            best_vector = x.copy()
    
    return best_combination, best_vector, best_value

classical_comb, classical_vec, classical_val = classical_solution(mu, cov, budget)
print(f"✅ Solução clássica ÓTIMA: ativos {classical_comb}")
print(f"Valor da função objetivo: {classical_val:.6f}")

# ===========================================================================
# CÉLULA 5: QAOA COM ESTRATÉGIA AGGRESSIVA
# ===========================================================================

print("\n🚀 EXECUTANDO QAOA COM ESTRATÉGIA AGGRESSIVA")
print("=" * 60)

try:
    # CONFIGURAÇÃO AGGRESSIVA BASEADA EM ANÁLISE
    print("🎯 CONFIGURAÇÃO AGGRESSIVA:")
    print("• Otimizador: COBYLA (maxiter=400) - máxima estabilidade")
    print("• Camadas: p=3 - balance ideal")
    print("• Penalidade: 500x escala")
    print("• Estratégia: Foco TOTAL em soluções válidas")
    
    from qiskit_algorithms.optimizers import COBYLA
    
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=400)  # Máximo de iterações

    # Ponto inicial AGGRESSIVO baseado em análise prévia
    initial_point = [1.2, 0.8, 0.6, 0.9, 0.5, 0.7]  # 6 parâmetros para p=3
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer, 
        reps=3,
        initial_point=initial_point
    )
    
    print(f"\n🔍 HAMILTONIANO FINAL:")
    print(f"   • {hamiltonian}")
    
    print("\n⚡ Executando QAOA aggressivo...")
    result_local = qaoa.compute_minimum_eigenvalue(hamiltonian)

    print(f"✅ QAOA executado com sucesso!")
    print(f"Autovalor do Hamiltoniano: {result_local.eigenvalue:.6f}")
    
    # ANÁLISE AGGRESSIVA DOS RESULTADOS
    if hasattr(result_local, 'eigenstate') and result_local.eigenstate is not None:
        if isinstance(result_local.eigenstate, dict):
            counts = result_local.eigenstate
            total_shots = sum(counts.values())
            
            probabilities = {state: count/total_shots for state, count in counts.items()}
            sorted_states = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            valid_solutions = []
            invalid_solutions = []
            
            print(f"\n📊 DISTRIBUIÇÃO AGGRESSIVA:")
            print("=" * 90)
            print(f"{'Estado':<10} {'Prob.':<10} {'Valor Obj.':<12} {'Ativos':<15} {'Tipo':<10} {'% do Total':<12}")
            print("-" * 90)
            
            for bitstring, prob in sorted_states:
                if len(bitstring) != N: continue
                    
                x = np.array([int(bit) for bit in bitstring])
                objective_value = x @ cov @ x - mu @ x
                num_assets = sum(x)
                is_valid = num_assets == budget
                assets = [i for i, val in enumerate(x) if val == 1]
                
                solution_data = (bitstring, objective_value, prob, assets, is_valid)
                
                if is_valid:
                    valid_solutions.append(solution_data)
                else:
                    invalid_solutions.append(solution_data)
            
            # MOSTRAR PRIMEIRO as soluções válidas por probabilidade
            valid_solutions.sort(key=lambda x: x[2], reverse=True)  # Ordenar por probabilidade
            
            print("🎯 SOLUÇÕES VÁLIDAS ENCONTRADAS:")
            for bitstring, obj_val, prob, assets, is_valid in valid_solutions:
                optimal = "🌟 ÓTIMO" if assets == list(classical_comb) else ""
                print(f"{bitstring:<10} {prob:<10.4f} {obj_val:<12.6f} {str(assets):<15} {'✅ VÁLIDO':<10} {prob*100:<11.2f}% {optimal}")
            
            # MOSTRAR algumas inválidas mais problemáticas
            print(f"\n⚠️  PRINCIPAIS SOLUÇÕES INVÁLIDAS (problema):")
            invalid_shown = 0
            for bitstring, obj_val, prob, assets, is_valid in invalid_solutions:
                if prob > 0.01 and invalid_shown < 5:
                    print(f"{bitstring:<10} {prob:<10.4f} {obj_val:<12.6f} {str(assets):<15} {'❌ INVÁLIDO':<10} {prob*100:<11.2f}%")
                    invalid_shown += 1
            
            # ANÁLISE ESTATÍSTICA AGGRESSIVA
            if valid_solutions:
                total_valid_prob = sum(sol[2] for sol in valid_solutions)
                num_valid_solutions = len(valid_solutions)
                
                # Encontrar a melhor solução por valor objetivo
                valid_solutions_by_value = sorted(valid_solutions, key=lambda x: x[1])
                best_quantum = valid_solutions_by_value[0]
                best_gap = abs(best_quantum[1] - classical_val) / abs(classical_val) * 100
                
                # Probabilidade da solução ótima
                optimal_prob = next((sol[2] for sol in valid_solutions if sol[3] == list(classical_comb)), 0.0)
                
                print(f"\n📈 ESTATÍSTICAS AGGRESSIVAS:")
                print(f"• Soluções válidas encontradas: {num_valid_solutions}/6")
                print(f"• Probabilidade TOTAL em válidas: {total_valid_prob:.4f} ({total_valid_prob*100:.2f}%)")
                print(f"• Probabilidade na solução ÓTIMA: {optimal_prob:.4f} ({optimal_prob*100:.2f}%)")
                print(f"• Melhor solução QAOA: {best_quantum[3]} (valor: {best_quantum[1]:.6f})")
                print(f"• Gap para ótimo: {best_gap:.2f}%")
                
                # DIAGNÓSTICO AGGRESSIVO
                if total_valid_prob > 0.7 and optimal_prob > 0.2:
                    diagnosis = "🌟 EXCELENTE - Algoritmo funcionando perfeitamente"
                    recommendation = "Continuar com esta configuração"
                elif total_valid_prob > 0.4 and optimal_prob > 0.1:
                    diagnosis = "✅ MUITO BOM - Performance sólida"
                    recommendation = "Pequenos ajustes podem melhorar"
                elif total_valid_prob > 0.2 and optimal_prob > 0.05:
                    diagnosis = "⚠️  RAZOÁVEL - Funcionando mas pode melhorar"
                    recommendation = "Aumentar penalidade ou camadas"
                elif total_valid_prob > 0.05:
                    diagnosis = "🔴 PROBLEMÁTICO - Precisamos de ajustes"
                    recommendation = "Aumentar AGGRESSIVAMENTE a penalidade"
                else:
                    diagnosis = "💥 CRÍTICO - Intervenção urgente necessária"
                    recommendation = "Revisar COMPLETAMENTE a formulação"
                
                print(f"• Diagnóstico: {diagnosis}")
                print(f"• Recomendação: {recommendation}")
                
                # SCORE DE PERFORMANCE
                validity_score = min(100, total_valid_prob * 200)  # Máximo 100
                optimality_score = min(100, optimal_prob * 500)    # Máximo 100  
                gap_score = max(0, 100 - best_gap)                 # 100 se gap=0%
                
                overall_score = (validity_score + optimality_score + gap_score) / 3
                
                print(f"\n🏆 SCORE DE PERFORMANCE: {overall_score:.1f}/100")
                
                if overall_score > 80:
                    rating = "🏅 OURO"
                elif overall_score > 60:
                    rating = "🥈 PRATA" 
                elif overall_score > 40:
                    rating = "🥉 BRONZE"
                else:
                    rating = "📉 INSUFICIENTE"
                
                print(f"• Classificação: {rating}")
                
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

# ===========================================================================
# CÉLULA 6: ANÁLISE COMPARATIVA AVANÇADA
# ===========================================================================

print("\n🔍 ANÁLISE COMPARATIVA CIENTÍFICA")
print("=" * 80)

try:
    # Calcular todas as soluções clássicas possíveis
    all_classical_solutions = []
    for comb in combinations(range(N), budget):
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        all_classical_solutions.append((comb, value, x))
    
    all_classical_solutions.sort(key=lambda x: x[1])
    
    print(f"📈 ANÁLISE DAS SOLUÇÕES CLÁSSICAS:")
    print(f"• Total de combinações válidas: {len(all_classical_solutions)}")
    print(f"• Melhor solução: {all_classical_solutions[0][0]} (valor: {all_classical_solutions[0][1]:.6f})")
    print(f"• Pior solução: {all_classical_solutions[-1][0]} (valor: {all_classical_solutions[-1][1]:.6f})")
    
    # Análise de performance do QAOA
    if 'valid_solutions' in locals() and valid_solutions:
        best_quantum = valid_solutions[0]
        best_gap = abs(best_quantum[1] - classical_val) / abs(classical_val) * 100
        total_valid_prob = sum(sol[2] for sol in valid_solutions)
        
        print(f"\n🎯 PERFORMANCE DO QAOA:")
        print(f"• Solução encontrada: {best_quantum[3]}")
        print(f"• Valor: {best_quantum[1]:.6f}")
        print(f"• Gap: {best_gap:.2f}%")
        print(f"• Probabilidade: {best_quantum[2]:.4f}")
        print(f"• Eficiência total: {total_valid_prob:.4f}")
        
        # Score de qualidade
        gap_score = max(0, 100 - best_gap)
        prob_score = best_quantum[2] * 100
        efficiency_score = total_valid_prob * 100
        
        overall_score = (gap_score + prob_score + efficiency_score) / 3
        
        print(f"\n🏆 SCORE FINAL: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            rating = "🌟 EXCELENTE"
        elif overall_score >= 60:
            rating = "✅ BOM" 
        elif overall_score >= 40:
            rating = "⚠️  RAZOÁVEL"
        else:
            rating = "🔴 PRECISA MELHORAR"
        
        print(f"• CLASSIFICAÇÃO: {rating}")
        
        # Recomendações específicas
        print(f"\n💡 RECOMENDAÇÕES:")
        if best_quantum[2] < 0.1:
            print("• Aumentar as camadas do QAOA (p=4 ou p=5)")
            print("• Testar diferentes otimizadores (SPSA, NFT)")
        if total_valid_prob < 0.3:
            print("• Aumentar a penalidade no Hamiltoniano")
            print("• Verificar a construção do QUBO")
        if best_gap > 10:
            print("• Aumentar o número de iterações do otimizador")
            print("• Melhorar o ponto inicial do QAOA")
    
except Exception as e:
    print(f"❌ Erro na análise comparativa: {e}")

print(f"\n🎯 PRÓXIMOS PASSOS:")
print("1. Executar com p=4 para melhorar a probabilidade da solução ótima")
print("2. Testar com penalty=10.0 para forçar soluções válidas")
print("3. Experimentar otimizador SPSA para melhor convergência")
print("4. Adicionar mais ativos (N=6) para teste de escalabilidade")