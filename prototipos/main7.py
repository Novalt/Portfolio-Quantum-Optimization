# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO - VERSÃO OTIMIZADA
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

print("🚀 FERRAMENTA DE OTIMIZAÇÃO DE PORTFÓLIO QUÂNTICA - VERSÃO OTIMIZADA")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO OTIMIZADA BASEADA NOS SEUS TESTES
# ===========================================================================

# 🎯 CONFIGURAÇÃO OTIMIZADA PARA MELHOR PERFORMANCE
PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 6,              # ⚡ Mantenha 6 para consistência
    'NUM_SELECIONAR': 3,          # ⚡ 3 de 6
    'TIPO_DADOS': 'sintetico',    # ⚡ 'sintetico' ou 'manual'
    
    # ⚙️ CONFIGURAÇÃO CRÍTICA - OTIMIZADA
    'PENALIDADE_FACTOR': 1200.0,  # ⚡ AUMENTADO: 1200 (seus testes mostram que 800-1605 ainda é baixo)
    'QAOA_CAMADAS': 3,            # ⚡ OTIMIZADO: 3 camadas (balance entre precisão e convergência)
    'QAOA_ITERACOES': 250,        # ⚡ AUMENTADO: 250 iterações
    'OTIMIZADOR': 'COBYLA',       # ⚡ ALTERADO: COBYLA para melhor convergência em problemas menores
    
    # 🎯 MODO DE EXECUÇÃO
    'MODO_EXECUCAO': 'simulacao',  # ⚡ 'simulacao' ou 'ibm_quantum'
    
    # 🔧 CONFIGURAÇÕES AVANÇADAS
    'SEED': 42,                   # ⚡ Seed para reproducibilidade
    'SHOTS': 10000,               # ⚡ Shots para análise estatística
}

print("🎯 CONFIGURAÇÃO OTIMIZADA PARA MAIOR PROBABILIDADE NA SOLUÇÃO ÓTIMA")

# ===========================================================================
# CÉLULA 3: GERADOR DE DADOS COM NORMALIZAÇÃO MELHORADA
# ===========================================================================

def gerar_dados_portfolio_otimizado(config):
    """
    Gera dados de portfólio com normalização melhorada
    """
    np.random.seed(config['SEED'])
    n = config['NUM_ATIVOS']
    
    if config['TIPO_DADOS'] == 'manual':
        returns = np.array(config.get('RETORNOS_MANUAL', [0.12, 0.10, 0.14, 0.07, 0.15, 0.11]))
        covariance = np.array(config.get('COVARIANCIA_MANUAL', [
            [0.1, 0.02, 0.01, 0.03, 0.02, 0.01],
            [0.02, 0.15, 0.05, 0.02, 0.03, 0.02],
            [0.01, 0.05, 0.2, 0.04, 0.02, 0.03],
            [0.03, 0.02, 0.04, 0.1, 0.02, 0.01],
            [0.02, 0.03, 0.02, 0.02, 0.12, 0.02],
            [0.01, 0.02, 0.03, 0.01, 0.02, 0.08]
        ]))
    else:
        # Geração de dados sintéticos MAIS ESTÁVEIS
        returns = np.random.uniform(0.05, 0.20, n)
        
        # Matriz de covariância mais realista e bem condicionada
        covariance = np.random.uniform(0.01, 0.10, (n, n))
        covariance = (covariance + covariance.T) / 2
        np.fill_diagonal(covariance, np.random.uniform(0.08, 0.15, n))
        
        # Garantir que seja positiva definida
        covariance += np.eye(n) * 0.1
    
    # 🔥 NORMALIZAÇÃO CRÍTICA: Escalar para melhor condicionamento numérico
    scale_factor = np.max(np.abs(returns)) * 2
    returns_normalized = returns / scale_factor
    covariance_normalized = covariance / (scale_factor ** 2)
    
    print(f"📊 DADOS GERADOS E NORMALIZADOS:")
    print(f"• Fator de escala: {scale_factor:.4f}")
    print(f"• Retornos originais: {[f'{r:.3f}' for r in returns]}")
    print(f"• Retornos normalizados: {[f'{r:.3f}' for r in returns_normalized]}")
    
    return returns_normalized, covariance_normalized, returns, covariance

def setup_portfolio_problem_otimizado(returns, covariance, num_assets_to_select):
    """
    Configuração otimizada com análise mais detalhada
    """
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"\n📊 CONFIGURAÇÃO DO PORTFÓLIO:")
    print(f"• Ativos disponíveis: {N}")
    print(f"• Ativos a selecionar: {budget}")
    
    # Análise completa das soluções clássicas
    solutions = []
    all_combinations = list(combinations(range(N), budget))
    
    for comb in all_combinations:
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        solutions.append((comb, value, x))
    
    solutions.sort(key=lambda x: x[1])
    best_sol = solutions[0]
    worst_sol = solutions[-1]
    
    print(f"🎯 Melhor portfólio clássico: {best_sol[0]} (valor: {best_sol[1]:.6f})")
    print(f"📉 Pior portfólio clássico: {worst_sol[0]} (valor: {worst_sol[1]:.6f})")
    print(f"📈 Número de combinações possíveis: {len(solutions)}")
    
    # Análise de distribuição de valores
    values = [sol[1] for sol in solutions]
    print(f"📊 Estatísticas dos valores: min={min(values):.4f}, max={max(values):.4f}, avg={np.mean(values):.4f}")
    
    return mu, cov, budget, best_sol, solutions

# Gerar dados otimizados
returns_norm, covariance_norm, returns_orig, covariance_orig = gerar_dados_portfolio_otimizado(PORTFOLIO_CONFIG)
mu, cov, budget, classical_solution, all_solutions = setup_portfolio_problem_otimizado(
    returns_norm, covariance_norm, PORTFOLIO_CONFIG['NUM_SELECIONAR']
)

classical_comb, classical_val, classical_vec = classical_solution

# ===========================================================================
# CÉLULA 4: HAMILTONIANO SUPER OTIMIZADO
# ===========================================================================

print("\n🔧 CONFIGURANDO HAMILTONIANO SUPER OTIMIZADO")
print("=" * 70)

def build_super_optimized_hamiltonian(mu, cov, budget, penalty_factor):
    """
    Hamiltoniano com penalidade adaptativa e construção mais inteligente
    """
    N = len(mu)
    
    # 🔥 CÁLCULO DE PENALIDADE ADAPTATIVA - CORREÇÃO CRÍTICA
    sample_values = []
    sample_combinations = list(combinations(range(N), budget))
    
    # Amostrar algumas combinações para estimar a escala
    for comb in sample_combinations[:min(50, len(sample_combinations))]:
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        sample_values.append(abs(value))
    
    avg_obj_val = np.mean(sample_values) if sample_values else 1.0
    max_obj_val = np.max(sample_values) if sample_values else 1.0
    
    # Penalidade baseada na média e máximo
    base_penalty = (avg_obj_val + max_obj_val) / 2
    penalty = penalty_factor * base_penalty
    
    print(f"⚙️  CONFIGURAÇÃO INTELIGENTE DO HAMILTONIANO:")
    print(f"   • Número de qubits: {N}")
    print(f"   • Valor objetivo médio: {avg_obj_val:.6f}")
    print(f"   • Valor objetivo máximo: {max_obj_val:.6f}")
    print(f"   • Penalidade calculada: {penalty:.2f}")
    print(f"   • Fator de penalidade: {penalty_factor}")
    
    # Construção do QUBO otimizada
    Q = np.zeros((N, N))
    q = np.zeros(N)
    
    # Termos originais
    Q += cov
    q -= mu
    
    # Restrição de budget com penalidade adaptativa
    for i in range(N):
        Q[i, i] += penalty * (1 - 2 * budget)
    
    for i in range(N):
        for j in range(i+1, N):
            Q[i, j] += 2 * penalty
            Q[j, i] += 2 * penalty
    
    # Conversão direta para Ising mais eficiente
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
    
    print(f"✅ Hamiltoniano super otimizado:")
    print(f"   • Termos Pauli: {len(hamiltonian)}")
    print(f"   • Norma do Hamiltoniano: {np.linalg.norm(coeffs_list):.4f}")
    
    return hamiltonian, penalty

hamiltonian, actual_penalty = build_super_optimized_hamiltonian(
    mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR']
)

# ===========================================================================
# CÉLULA 5: QAOA COM PONTOS INICIAIS INTELIGENTES
# ===========================================================================

print("\n🚀 EXECUTANDO QAOA COM OTIMIZAÇÃO AVANÇADA")
print("=" * 70)

def get_advanced_optimizer_config(config, num_assets):
    """
    Configuração avançada do otimizador baseada em análise empírica
    """
    reps = config['QAOA_CAMADAS']
    
    # Configurações baseadas em testes extensivos
    if config['OTIMIZADOR'] == 'COBYLA':
        optimizer = COBYLA(maxiter=config['QAOA_ITERACOES'], tol=1e-6)
    else:
        optimizer = SPSA(maxiter=config['QAOA_ITERACOES'])
    
    # 🔥 PONTOS INICIAIS INTELIGENTES - baseados em pesquisa
    if reps == 2:
        # [γ1, β1, γ2, β2] - γ alto inicial, β baixo
        initial_point = [0.8, 0.3, 0.6, 0.4]
    elif reps == 3:
        # [γ1, β1, γ2, β2, γ3, β3] - decrescendo γ, crescendo β
        initial_point = [0.9, 0.2, 0.7, 0.3, 0.5, 0.4]
    elif reps == 4:
        initial_point = [0.9, 0.1, 0.8, 0.2, 0.6, 0.3, 0.4, 0.4]
    else:
        initial_point = np.random.uniform(0.1, 1.0, 2 * reps)
    
    return {
        'reps': reps,
        'optimizer': optimizer,
        'initial_point': initial_point,
        'max_iter': config['QAOA_ITERACOES']
    }

def run_advanced_qaoa(hamiltonian, config):
    """
    Execução do QAOA com técnicas avançadas
    """
    num_assets = len(mu)
    qaoa_config = get_advanced_optimizer_config(config, num_assets)
    
    print("⚙️  CONFIGURAÇÃO AVANÇADA:")
    print(f"• Algoritmo: QAOA com {qaoa_config['reps']} camadas")
    print(f"• Otimizador: {qaoa_config['optimizer'].__class__.__name__}")
    print(f"• Iterações: {qaoa_config['max_iter']}")
    print(f"• Qubits: {num_assets}")
    print(f"• Ponto inicial: {qaoa_config['initial_point']}")
    
    sampler = StatevectorSampler()
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=qaoa_config['optimizer'],
        reps=qaoa_config['reps'],
        initial_point=qaoa_config['initial_point']
    )
    
    print("⚡ Executando otimização quântica avançada...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    return result

# Executar QAOA otimizado
result = run_advanced_qaoa(hamiltonian, PORTFOLIO_CONFIG)

# ===========================================================================
# CÉLULA 6: ANÁLISE DE RESULTADOS MELHORADA
# ===========================================================================

print("\n📊 RELATÓRIO DE RESULTADOS DETALHADO")
print("=" * 70)

def analyze_improved_results(result, classical_comb, classical_val, mu, cov, budget, all_solutions):
    """
    Análise detalhada com métricas aprimoradas
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        return {"status": "ERRO", "message": "Sem resultados para análise"}
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Coletar e analisar TODAS as soluções
    valid_solutions = []
    invalid_solutions = []
    top_invalid = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != len(mu):
            continue
            
        x = np.array([int(bit) for bit in bitstring])
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
        
        if num_assets == budget:
            valid_solutions.append(solution_data)
        else:
            invalid_solutions.append(solution_data)
    
    # Ordenar soluções
    valid_solutions.sort(key=lambda x: x['value'])
    invalid_solutions.sort(key=lambda x: x['probability'], reverse=True)
    top_invalid = invalid_solutions[:5]
    
    if valid_solutions:
        best_solution = valid_solutions[0]
        
        # Encontrar ranking da solução clássica
        classical_ranking = None
        classical_prob = 0.0
        for i, sol in enumerate(valid_solutions):
            if sol['assets'] == list(classical_comb):
                classical_ranking = i + 1
                classical_prob = sol['probability']
                break
        
        total_valid_prob = sum(sol['probability'] for sol in valid_solutions)
        num_valid = len(valid_solutions)
        
        print("🎯 SOLUÇÃO QUÂNTICA DETALHADA:")
        print(f"• Melhor portfólio: {best_solution['assets']}")
        print(f"• Valor da solução: {best_solution['value']:.6f}")
        print(f"• Probabilidade: {best_solution['probability']:.4f} ({best_solution['probability']*100:.2f}%)")
        
        print(f"\n📈 EFICÁCIA AVANÇADA:")
        print(f"• Solução ótima clássica: {list(classical_comb)}")
        print(f"• Ranking da solução ótima: {classical_ranking}°/{num_valid}")
        print(f"• Probabilidade na ótima: {classical_prob:.4f} ({classical_prob*100:.2f}%)")
        print(f"• Eficiência total válida: {total_valid_prob:.4f} ({total_valid_prob*100:.2f}%)")
        print(f"• Número de soluções válidas: {num_valid}")
        print(f"• Gap de precisão: {abs(best_solution['value'] - classical_val)/abs(classical_val)*100:.2f}%")
        
        # Top 3 soluções válidas
        print(f"\n🏆 TOP 3 SOLUÇÕES VÁLIDAS:")
        for i, sol in enumerate(valid_solutions[:3]):
            print(f"  {i+1}. {sol['assets']} - valor: {sol['value']:.6f} - prob: {sol['probability']:.4f}")
        
        if top_invalid:
            print(f"\n⚠️  PRINCIPAIS SOLUÇÕES INVÁLIDAS:")
            for sol in top_invalid[:3]:
                print(f"   • {sol['bitstring']}: {sol['probability']:.3f} (ativos: {sol['assets']})")
        
        # Avaliação de qualidade aprimorada
        if classical_prob > 0.15 and total_valid_prob > 0.4:
            usability = "✅ EXCELENTE - Pronta para produção"
            score = "A+"
        elif classical_prob > 0.08 and total_valid_prob > 0.25:
            usability = "✅ ALTA - Muito boa"
            score = "A"
        elif classical_prob > 0.04 and total_valid_prob > 0.15:
            usability = "✅ BOA - Funcional"
            score = "B"
        elif classical_prob > 0.02 and total_valid_prob > 0.08:
            usability = "⚠️  MODERADA - Aceitável"
            score = "C"
        elif classical_prob > 0:
            usability = "🔴 BÁSICA - Experimental"
            score = "D"
        else:
            usability = "💥 CRÍTICA - Não funcional"
            score = "F"
        
        print(f"\n🏆 AVALIAÇÃO FINAL: {score}")
        print(f"• Usabilidade: {usability}")
        print(f"• Probabilidade ótima: {classical_prob*100:.2f}%")
        print(f"• Eficiência válida: {total_valid_prob*100:.2f}%")
        
        return {
            'portfolio': best_solution['assets'],
            'value': best_solution['value'],
            'optimal_probability': classical_prob,
            'efficiency': total_valid_prob,
            'usability_score': score,
            'usability': usability,
            'num_valid_solutions': num_valid,
            'classical_ranking': classical_ranking,
            'best_solution_prob': best_solution['probability']
        }
    
    return {"status": "SEM_SOLUCOES", "message": "Nenhuma solução válida encontrada"}

# Análise detalhada
analysis = analyze_improved_results(result, classical_comb, classical_val, mu, cov, budget, all_solutions)

# ===========================================================================
# CÉLULA 7: RELATÓRIO EXECUTIVO COMPLETO
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - PORTFÓLIO QUÂNTICO OTIMIZADO")
print("=" * 70)

if 'portfolio' in analysis:
    print(f"\n✅ RESULTADO OBTIDO:")
    print(f"   • Portfólio recomendado: {analysis['portfolio']}")
    print(f"   • Valor da solução: {analysis['value']:.6f}")
    print(f"   • Probabilidade: {analysis['best_solution_prob']:.4f} ({analysis['best_solution_prob']*100:.2f}%)")
    print(f"   • Nota: {analysis['usability_score']}/A+")
    
    print(f"\n📊 PERFORMANCE AVANÇADA:")
    print(f"   • Eficiência em válidas: {analysis['efficiency']:.3f} ({analysis['efficiency']*100:.1f}%)")
    print(f"   • Precisão na ótima: {analysis['optimal_probability']:.3f} ({analysis['optimal_probability']*100:.1f}%)")
    print(f"   • Ranking da ótima: {analysis['classical_ranking']}° posição")
    print(f"   • Soluções válidas: {analysis['num_valid_solutions']}")
    
    print(f"\n🎯 STATUS:")
    print(f"   {analysis['usability']}")
    
    # Recomendações específicas baseadas no resultado
    print(f"\n🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    
    if analysis['usability_score'] in ['D', 'F']:
        print("1. 🎯 AUMENTAR penalidade para 1500-2000")
        print("2. 🔄 MUDAR para otimizador COBYLA")
        print("3. 📈 AUMENTAR iterações para 300+")
        print("4. 🎛️  TESTAR com 4 camadas QAOA")
    elif analysis['usability_score'] == 'C':
        print("1. ✅ MANTER configuração atual")
        print("2. 🔼 AUMENTAR iterações para 200")
        print("3. 📊 COLETAR mais dados de teste")
    elif analysis['usability_score'] == 'B':
        print("1. 🎉 Configuração BOA para uso")
        print("2. 🔄 TESTAR com dados reais")
        print("3. 🌐 CONSIDERAR execução em IBM Quantum")
    else:  # A ou A+
        print("1. 🏆 Configuração EXCELENTE!")
        print("2. 🌐 PRONTA para IBM Quantum")
        print("3. 📈 INTEGRAR com sistemas reais")

print(f"\n🔧 CONFIGURAÇÃO APLICADA:")
print(f"   • Ativos: {PORTFOLIO_CONFIG['NUM_ATIVOS']} | Selecionar: {PORTFOLIO_CONFIG['NUM_SELECIONAR']}")
print(f"   • Penalidade: {PORTFOLIO_CONFIG['PENALIDADE_FACTOR']} (calculada: {actual_penalty:.1f})")
print(f"   • Camadas QAOA: {PORTFOLIO_CONFIG['QAOA_CAMADAS']}")
print(f"   • Iterações: {PORTFOLIO_CONFIG['QAOA_ITERACOES']}")
print(f"   • Otimizador: {PORTFOLIO_CONFIG['OTIMIZADOR']}")

print(f"\n💡 CONFIGURAÇÕES RECOMENDADAS PARA TESTE:")

print(f"\n🎯 PARA MAXIMIZAR PROBABILIDADE NA ÓTIMA:")
print("   • Penalidade: 1200-1500")
print("   • Camadas: 3")
print("   • Iterações: 200-300") 
print("   • Otimizador: COBYLA")

print(f"\n⚡ PARA EXECUÇÃO RÁPIDA:")
print("   • Penalidade: 800-1000")
print("   • Camadas: 2")
print("   • Iterações: 100-150")
print("   • Otimizador: SPSA")

print(f"\n📊 MÉTRICAS DE SUCESSO ALVO:")
print("   • Probabilidade na ótima: > 5% (IDEAL: > 10%)")
print("   • Eficiência em válidas: > 20% (IDEAL: > 30%)")
print("   • Ranking da ótima: Top 3")

print(f"\n🌟 FERRAMENTA OTIMIZADA - PRONTA PARA TESTES AVANÇADOS!")