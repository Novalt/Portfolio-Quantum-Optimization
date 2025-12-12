# ===========================================================================
# CÉLULA 1: IMPORTAÇÕES E CONFIGURAÇÃO - MODO LOCAL
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

print("🚀 PORTFÓLIO QUÂNTICO - EXECUÇÃO LOCAL OTIMIZADA")
print("=" * 70)

# ===========================================================================
# CÉLULA 2: CONFIGURAÇÃO OTIMIZADA PARA EXECUÇÃO LOCAL
# ===========================================================================

PORTFOLIO_CONFIG = {
    'NUM_ATIVOS': 8,              # ⚡ Problema bem dimensionado para teste
    'NUM_SELECIONAR': 2,          # ⚡ 2 de 8 = 28 combinações (bom para teste)
    'TIPO_DADOS': 'realista',     # ⚡ Dados baseados em mercado real
    
    # ⚙️ CONFIGURAÇÃO OTIMIZADA PARA LOCAL
    'PENALIDADE_FACTOR': 1.5,     # ⚡ Penalidade reduzida para melhor convergência
    'QAOA_CAMADAS': 2,            # ⚡ 2 camadas para execução rápida
    'QAOA_ITERACOES': 100,        # ⚡ Mais iterações para melhor qualidade
    'OTIMIZADOR': 'COBYLA',       # ⚡ COBYLA melhor para simulação local
    
    # 🎯 CONFIGURAÇÃO DE EXECUÇÃO - 100% LOCAL
    'MODO_EXECUCAO': 'simulacao',  # ⚡ Força execução local
    
    'SEED': 42,
}

print("🎯 CONFIGURADO PARA EXECUÇÃO LOCAL")
print("💡 8 ativos, 2 selecionar - Problema ideal para testes")
print("💰 SEM consumo de créditos da IBM")

# ===========================================================================
# CÉLULA 3: DADOS REALISTAS OTIMIZADOS - CORRIGIDO
# ===========================================================================

def gerar_dados_reais_otimizados():
    """
    Gera dados realistas otimizados para execução local
    """
    np.random.seed(42)
    n = PORTFOLIO_CONFIG['NUM_ATIVOS']
    
    print("📊 GERANDO DADOS DE MERCADO REALISTAS...")
    
    # Retornos anuais realistas para 8 ativos
    returns = np.array([0.08, 0.12, 0.15, 0.06, 0.18, 0.09, 0.11, 0.14])
    
    # Matriz de covariância realista para 8 ativos
    base_cov = np.array([
        [0.040, 0.008, 0.012, 0.004, 0.015, 0.006, 0.009, 0.011],
        [0.008, 0.090, 0.010, 0.007, 0.020, 0.008, 0.012, 0.015],
        [0.012, 0.010, 0.120, 0.009, 0.025, 0.010, 0.015, 0.018],
        [0.004, 0.007, 0.009, 0.030, 0.006, 0.012, 0.008, 0.007],
        [0.015, 0.020, 0.025, 0.006, 0.150, 0.012, 0.020, 0.022],
        [0.006, 0.008, 0.010, 0.012, 0.012, 0.060, 0.010, 0.009],
        [0.009, 0.012, 0.015, 0.008, 0.020, 0.010, 0.080, 0.014],
        [0.011, 0.015, 0.018, 0.007, 0.022, 0.009, 0.014, 0.100]
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
    print(f"📊 Variâncias: {[f'{covariance[i,i]:.3f}' for i in range(min(3, n))]}...")
    
    return returns, covariance

# ===========================================================================
# CÉLULA 4: CONFIGURAÇÃO DO PROBLEMA - CORRIGIDA
# ===========================================================================

def setup_problema_local_otimizado(returns, covariance, num_assets_to_select):
    """
    Configura problema otimizado para execução local
    """
    N = len(returns)
    mu = np.array(returns)
    cov = np.array(covariance)
    budget = num_assets_to_select
    
    print(f"\n📊 CONFIGURAÇÃO DO PORTFÓLIO:")
    print(f"• Ativos: {N}, Selecionar: {budget}")
    
    # Calcular TODAS as combinações (é viável para 8 ativos)
    all_combinations = list(combinations(range(N), budget))
    solutions = []
    
    for comb in all_combinations:
        x = np.zeros(N)
        x[list(comb)] = 1
        value = x @ cov @ x - mu @ x
        solutions.append((comb, value, x))
    
    solutions.sort(key=lambda x: x[1])
    best_solution = solutions[0]
    worst_solution = solutions[-1]
    
    values = [sol[1] for sol in solutions]
    
    print(f"🎯 Melhor portfólio: {best_solution[0]} (valor: {best_solution[1]:.6f})")
    print(f"📉 Pior portfólio: {worst_solution[0]} (valor: {worst_solution[1]:.6f})")
    print(f"📈 Estatísticas completas:")
    print(f"  • Média: {np.mean(values):.6f}")
    print(f"  • Desvio padrão: {np.std(values):.6f}")
    print(f"  • Número de combinações: {len(solutions)}")
    
    return mu, cov, budget, best_solution, solutions

# Gerar dados
returns, covariance = gerar_dados_reais_otimizados()
mu, cov, budget, reference_solution, all_solutions = setup_problema_local_otimizado(
    returns, covariance, PORTFOLIO_CONFIG['NUM_SELECIONAR']
)

reference_comb, reference_val, reference_vec = reference_solution

# ===========================================================================
# CÉLULA 5: HAMILTONIANO OTIMIZADO PARA EXECUÇÃO LOCAL
# ===========================================================================

print("\n🔧 CONSTRUINDO HAMILTONIANO OTIMIZADO")
print("=" * 70)

def build_hamiltonian_local_otimizado(mu, cov, budget, penalty_factor, all_solutions):
    """
    Hamiltoniano otimizado para execução local
    """
    N = len(mu)
    
    # Calcular escala de forma precisa (temos todas as soluções)
    values = [sol[1] for sol in all_solutions]
    avg_val = np.mean(values)
    std_val = np.std(values)
    
    # Penalidade adaptativa
    penalty = penalty_factor * std_val
    
    print(f"⚙️  CONFIGURAÇÃO LOCAL OTIMIZADA:")
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
    
    # Análise da escala
    coeffs_array = np.array(coeffs_list)
    norm_hamiltonian = np.linalg.norm(coeffs_array)
    
    print(f"✅ Hamiltoniano local otimizado:")
    print(f"   • Termos Pauli: {len(hamiltonian)}")
    print(f"   • Norma: {norm_hamiltonian:.2f}")
    
    # Verificação da escala
    if norm_hamiltonian > 10:
        print("⚠️  ALERTA: Hamiltoniano pode estar com escala alta")
    else:
        print("✅ Escala adequada para execução local")
    
    return hamiltonian, penalty

hamiltonian, actual_penalty = build_hamiltonian_local_otimizado(
    mu, cov, budget, PORTFOLIO_CONFIG['PENALIDADE_FACTOR'], all_solutions
)

# ===========================================================================
# CÉLULA 6: EXECUÇÃO LOCAL OTIMIZADA
# ===========================================================================

print("\n🚀 INICIANDO EXECUÇÃO LOCAL OTIMIZADA")
print("=" * 70)

def run_qaoa_local_otimizado(hamiltonian, config):
    """
    Execução do QAOA otimizada para ambiente local
    """
    reps = config['QAOA_CAMADAS']
    
    if config['OTIMIZADOR'] == 'SPSA':
        optimizer = SPSA(maxiter=config['QAOA_ITERACOES'])
    else:
        optimizer = COBYLA(maxiter=config['QAOA_ITERACOES'])
    
    # Pontos iniciais otimizados para local
    if reps == 2:
        initial_point = [0.8, 0.4, 0.6, 0.3]
    elif reps == 3:
        initial_point = [0.8, 0.3, 0.7, 0.4, 0.5, 0.5]
    else:
        initial_point = np.random.uniform(0.1, 0.9, 2 * reps)
    
    print("⚙️  CONFIGURAÇÃO DE EXECUÇÃO LOCAL:")
    print(f"• Camadas QAOA: p={reps}")
    print(f"• Iterações: {config['QAOA_ITERACOES']}")
    print(f"• Otimizador: {optimizer.__class__.__name__}")
    print(f"• Ponto inicial: {initial_point}")
    
    sampler = StatevectorSampler()
    
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point
    )
    
    print("⚡ Executando otimização QAOA local...")
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    print("✅ EXECUÇÃO LOCAL CONCLUÍDA!")
    return result

# Execução principal - SEM tentar conectar na IBM
result = run_qaoa_local_otimizado(hamiltonian, PORTFOLIO_CONFIG)

# ===========================================================================
# CÉLULA 7: ANÁLISE DE RESULTADOS PARA EXECUÇÃO LOCAL
# ===========================================================================

print("\n📊 ANALISANDO RESULTADOS DA EXECUÇÃO LOCAL")
print("=" * 70)

def analise_resultados_local(result, reference_comb, reference_val, mu, cov, budget, all_solutions):
    """
    Análise otimizada para resultados locais
    """
    if not hasattr(result, 'eigenstate') or result.eigenstate is None:
        print("❌ Nenhum resultado para analisar")
        return {"status": "ERRO"}
    
    counts = result.eigenstate
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    N = len(mu)
    total_combinacoes = len(all_solutions)
    prob_aleatoria = 1 / total_combinacoes
    
    print(f"📈 CONTEXTO DO PROBLEMA:")
    print(f"• Combinações totais: {total_combinacoes}")
    print(f"• Probabilidade aleatória: {prob_aleatoria:.4f} ({prob_aleatoria*100:.2f}%)")
    
    # Analisar soluções
    valid_solutions = []
    
    for bitstring, prob in probabilities.items():
        if len(bitstring) != N:
            continue
            
        x = np.array([int(bit) for bit in bitstring])
        objective_value = x @ cov @ x - mu @ x
        num_assets = sum(x)
        assets = [i for i, val in enumerate(x) if val == 1]
        
        if num_assets == budget:
            valid_solutions.append({
                'bitstring': bitstring,
                'value': objective_value,
                'probability': prob,
                'assets': assets
            })
    
    valid_solutions.sort(key=lambda x: x['value'])
    
    if valid_solutions:
        best_solution = valid_solutions[0]
        
        # Encontrar todas as informações da solução de referência
        referencia_encontrada = False
        prob_referencia = 0.0
        ranking_referencia = None
        
        for i, sol in enumerate(valid_solutions):
            if sol['assets'] == list(reference_comb):
                referencia_encontrada = True
                prob_referencia = sol['probability']
                ranking_referencia = i + 1
                break
        
        total_valid_prob = sum(sol['probability'] for sol in valid_solutions)
        
        print(f"\n🎯 RESULTADOS OBTIDOS:")
        print(f"• Melhor portfólio: {best_solution['assets']}")
        print(f"• Valor objetivo: {best_solution['value']:.6f}")
        print(f"• Probabilidade: {best_solution['probability']:.4f} ({best_solution['probability']*100:.2f}%)")
        
        print(f"\n📊 EFICÁCIA DO ALGORITMO:")
        print(f"• Solução ótima: {list(reference_comb)}")
        print(f"• Referência encontrada: {'✅ SIM' if referencia_encontrada else '❌ NÃO'}")
        if referencia_encontrada:
            print(f"• Probabilidade na referência: {prob_referencia:.4f} ({prob_referencia*100:.2f}%)")
            print(f"• Ranking da referência: {ranking_referencia}°")
        print(f"• Eficiência em válidas: {total_valid_prob:.3f} ({total_valid_prob*100:.1f}%)")
        print(f"• Soluções válidas encontradas: {len(valid_solutions)}")
        
        # Top 5 soluções
        print(f"\n🏆 TOP 5 SOLUÇÕES VÁLIDAS:")
        for i, sol in enumerate(valid_solutions[:5]):
            marcador = "🎯" if sol['assets'] == list(reference_comb) else "  "
            print(f"  {marcador} {i+1}. {sol['assets']} - valor: {sol['value']:.4f} - prob: {sol['probability']:.4f}")
        
        # Avaliação para execução local
        if best_solution['probability'] > 0.1:
            avaliacao = "🎉 EXCELENTE"
            nota = "A+"
        elif best_solution['probability'] > 0.05:
            avaliacao = "✅ MUITO BOM"
            nota = "A"
        elif best_solution['probability'] > 0.02:
            avaliacao = "👍 BOM" 
            nota = "B"
        elif best_solution['probability'] > 0.01:
            avaliacao = "⚠️  RAZOÁVEL"
            nota = "C"
        elif best_solution['probability'] > prob_aleatoria:
            avaliacao = "🔴 BÁSICO"
            nota = "D"
        else:
            avaliacao = "💥 INSUFICIENTE"
            nota = "F"
        
        melhoria = best_solution['probability'] / prob_aleatoria
        
        print(f"\n🏅 AVALIAÇÃO LOCAL: {nota}")
        print(f"• {avaliacao}")
        print(f"• Melhoria sobre aleatório: {melhoria:.1f}x")
        
        return {
            'portfolio': best_solution['assets'],
            'value': best_solution['value'],
            'probability': best_solution['probability'],
            'found_reference': referencia_encontrada,
            'reference_probability': prob_referencia,
            'reference_ranking': ranking_referencia,
            'efficiency': total_valid_prob,
            'usability_score': nota,
            'num_valid_solutions': len(valid_solutions),
            'improvement_over_random': melhoria
        }
    
    return {"status": "SEM_SOLUCOES"}

analysis = analise_resultados_local(result, reference_comb, reference_val, mu, cov, budget, all_solutions)

# ===========================================================================
# CÉLULA 8: RELATÓRIO EXECUTIVO - EXECUÇÃO LOCAL
# ===========================================================================

print("\n" + "=" * 70)
print("🏁 RELATÓRIO EXECUTIVO - EXECUÇÃO LOCAL")
print("=" * 70)

if 'portfolio' in analysis:
    print(f"\n✅ RESULTADO OBTIDO:")
    print(f"   • Portfólio recomendado: {analysis['portfolio']}")
    print(f"   • Valor objetivo: {analysis['value']:.6f}")
    print(f"   • Probabilidade: {analysis['probability']:.4f} ({analysis['probability']*100:.2f}%)")
    print(f"   • Nota: {analysis['usability_score']}")
    
    print(f"\n📈 DESEMPENHO DO ALGORITMO:")
    print(f"   • Melhoria sobre aleatório: {analysis['improvement_over_random']:.1f}x")
    print(f"   • Eficiência em válidas: {analysis['efficiency']*100:.1f}%")
    print(f"   • Soluções válidas encontradas: {analysis['num_valid_solutions']}")
    if analysis['found_reference']:
        print(f"   • Ranking da solução ótima: {analysis['reference_ranking']}°")
        print(f"   • Probabilidade na ótima: {analysis['reference_probability']*100:.2f}%")

print(f"\n🔧 CONFIGURAÇÃO APLICADA:")
print(f"   • Ativos: {PORTFOLIO_CONFIG['NUM_ATIVOS']} | Selecionar: {PORTFOLIO_CONFIG['NUM_SELECIONAR']}")
print(f"   • Penalidade: {PORTFOLIO_CONFIG['PENALIDADE_FACTOR']} (calculada: {actual_penalty:.6f})")
print(f"   • Camadas QAOA: {PORTFOLIO_CONFIG['QAOA_CAMADAS']}")
print(f"   • Iterações: {PORTFOLIO_CONFIG['QAOA_ITERACOES']}")
print(f"   • Otimizador: {PORTFOLIO_CONFIG['OTIMIZADOR']}")

print(f"\n💡 PRÓXIMOS PASSOS BASEADOS NO RESULTADO:")

if analysis.get('usability_score', 'F') in ['A', 'A+', 'B']:
    print("1. 🎉 Sucesso! Resultados promissores obtidos")
    print("2. 🌐 Considerar teste em hardware real (IBM Quantum)")
    print("3. 📈 Aumentar complexidade do problema (mais ativos)")
    print("4. 🔄 Testar com mais camadas QAOA (p=3)")
else:
    print("1. 🔧 Ajustar penalidade (experimente 1.0 a 2.0)")
    print("2. 🔄 Aumentar número de iterações (150-200)")
    print("3. 📊 Testar com diferentes pontos iniciais")
    print("4. 🎯 Verificar construção do Hamiltoniano")

print(f"\n🎯 CRITÉRIOS PARA HARDWARE REAL:")
print("   • Nota A ou B neste teste local")
print("   • Probabilidade > 2% na solução ótima") 
print("   • Melhoria > 50x sobre aleatório")
print("   • Eficiência > 15% em válidas")

print(f"\n💰 STATUS CRÉDITOS: 100% PRESERVADOS - Execução local concluída!")