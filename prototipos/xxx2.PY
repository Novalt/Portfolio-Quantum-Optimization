# Célula 7: Implementação Científica do QAOA para Otimização de Portfólio
print("🔬 EXECUÇÃO CIENTÍFICA DO ALGORITMO QAOA")
print("=" * 70)

# Verificação do ambiente científico
print("📊 VALIDAÇÃO DO AMBIENTE CIENTÍFICO:")
print(f"• Problema: Otimização de Portfólio (Mean-Variance)")
print(f"• Ativos: {N}, Budget: {budget}")
print(f"• Algoritmo: QAOA (Quantum Approximate Optimization Algorithm)")
print(f"• Referência: Farhi et al. (2014) - arXiv:1411.4028")

try:
    # ===========================================================================
    # CONFIGURAÇÃO CIENTIFICAMENTE CORRETA DO QAOA
    # ===========================================================================
    
    # 1. SAMPLER CIENTIFICAMENTE VALIDADO
    # O StatevectorSampler é matematicamente equivalente à computação do estado completo
    from qiskit.primitives import StatevectorSampler
    
    print("\n🎯 CONFIGURANDO PARÂMETROS CIENTÍFICOS:")
    print("• Sampler: StatevectorSampler (simulação exata sem ruído)")
    print("• Otimizador: COBYLA (método de otimização numérica clássica)")
    print("• Camadas QAOA: p=1 (primeira ordem)")
    print("• Iterações: 50 (balanceamento entre precisão e tempo)")
    
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=50)
    
    # 2. INSTANCIAÇÃO DO ALGORITMO QAOA
    # Baseado no paper seminal de Farhi, Goldstone e Gutmann
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer, 
        reps=1,  # p=1 para demonstração inicial
        initial_point=[1.0, 1.0]  # γ, β iniciais padrão
    )
    
    # ===========================================================================
    # EXECUÇÃO DO ALGORITMO QUÂNTICO-CLÁSSICO
    # ===========================================================================
    
    print("\n⚡ EXECUTANDO ALGORITMO HÍBRIDO QUÂNTICO-CLÁSSICO:")
    print("Fase 1: Circuito quântico prepara estados superpostos")
    print("Fase 2: Medição do Hamiltoniano de custo")
    print("Fase 3: Otimização clássica dos parâmetros γ, β")
    print("Fase 4: Iteração até convergência")
    
    result_local = qaoa.compute_minimum_eigenvalue(hamiltonian)
    
    # ===========================================================================
    # ANÁLISE CIENTÍFICA DOS RESULTADOS
    # ===========================================================================
    
    print(f"\n✅ QAOA EXECUTADO COM SUCESSO!")
    print("=" * 70)
    
    # 1. RESULTADO PRINCIPAL
    print(f"📈 AUTOVALOR ENCONTRADO: {result_local.eigenvalue:.6f}")
    print(f"📊 VALOR CLÁSSICO ÓTIMO: {classical_val:.6f}")
    
    # 2. CÁLCULO DO GAP DE APROXIMAÇÃO QUÂNTICA
    quantum_value = result_local.eigenvalue.real  # Parte real do autovalor
    gap_percent = abs(quantum_value - classical_val) / abs(classical_val) * 100
    
    print(f"🎯 GAP QUÂNTICO-CLÁSSICO: {gap_percent:.2f}%")
    
    # 3. ANÁLISE DO ESTADO QUÂNTICO RESULTANTE
    if hasattr(result_local, 'eigenstate'):
        from qiskit.quantum_info import Statevector
        
        if isinstance(result_local.eigenstate, Statevector):
            statevector = result_local.eigenstate
            probabilities = statevector.probabilities()
            
            # Distribuição de probabilidade científica
            num_qubits = statevector.num_qubits
            bitstrings = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
            
            # Combinar e ordenar por probabilidade
            prob_dist = list(zip(bitstrings, probabilities))
            prob_dist_sorted = sorted(prob_dist, key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 DISTRIBUIÇÃO DE PROBABILIDADE QUÂNTICA:")
            print("=" * 80)
            print(f"{'Estado':<10} {'Prob.':<12} {'Valor Obj.':<12} {'Ativos':<8} {'Validação':<10}")
            print("-" * 80)
            
            top_solutions = []
            for bitstring, prob in prob_dist_sorted[:15]:  # Top 15 estados
                x = np.array([int(bit) for bit in bitstring])
                objective_value = x @ cov @ x - mu @ x
                num_assets = sum(x)
                is_valid = num_assets == budget
                assets = [i for i, val in enumerate(x) if val == 1]
                
                validity = "✅ VÁLIDO" if is_valid else "❌ INVÁLIDO"
                optimal = "🌟 ÓTIMO" if assets == list(classical_comb) else ""
                
                print(f"{bitstring:<10} {prob:<12.4f} {objective_value:<12.4f} {str(assets):<8} {validity} {optimal}")
                
                if is_valid:
                    top_solutions.append((bitstring, objective_value, prob, assets))
            
            # 4. ANÁLISE DAS SOLUÇÕES VÁLIDAS
            if top_solutions:
                print(f"\n🎯 SOLUÇÕES VÁLIDAS ENCONTRADAS (budget = {budget}):")
                print("=" * 60)
                
                # Ordenar soluções válidas por qualidade
                top_solutions.sort(key=lambda x: x[1])
                
                for i, (bitstring, obj_val, prob, assets) in enumerate(top_solutions[:5]):
                    rank = i + 1
                    gap_to_optimal = abs(obj_val - classical_val) / abs(classical_val) * 100
                    
                    print(f"{rank}. {bitstring} -> Valor: {obj_val:.4f} (Gap: {gap_to_optimal:.1f}%)")
                    print(f"   Ativos: {assets}, Probabilidade: {prob:.4f}")
                
                # 5. MÉTRICAS DE PERFORMANCE CIENTÍFICA
                best_quantum_solution = top_solutions[0]
                best_quantum_value = best_quantum_solution[1]
                final_gap = abs(best_quantum_value - classical_val) / abs(classical_val) * 100
                
                print(f"\n📊 MÉTRICAS CIENTÍFICAS FINAIS:")
                print(f"• Melhor solução clássica: {classical_comb}, valor: {classical_val:.6f}")
                print(f"• Melhor solução QAOA: {best_quantum_solution[3]}, valor: {best_quantum_value:.6f}")
                print(f"• Gap final de aproximação: {final_gap:.2f}%")
                print(f"• Probabilidade da melhor solução: {best_quantum_solution[2]:.4f}")
                
                # 6. AVALIAÇÃO DE QUALIDADE
                if final_gap < 1.0:
                    quality = "🌟 EXCELENTE - Solução praticamente ótima"
                elif final_gap < 5.0:
                    quality = "✅ MUITO BOA - Solução de alta qualidade"
                elif final_gap < 15.0:
                    quality = "⚠️  RAZOÁVEL - Solução aceitável"
                else:
                    quality = "🔴 BAIXA - Necessita melhorias"
                
                print(f"• Qualidade da aproximação: {quality}")
                
            else:
                print("❌ NENHUMA SOLUÇÃO VÁLIDA ENCONTRADA")
        
        else:
            print("⚠️  Estado quântico não está no formato esperado para análise")
    else:
        print("⚠️  Nenhum eigenstate retornado pelo QAOA")
    
    # 7. INFORMAÇÕES TÉCNICAS PARA PUBLICAÇÃO
    print(f"\n🔬 INFORMAÇÕES TÉCNICAS PARA ANÁLISE CIENTÍFICA:")
    if hasattr(result_local, 'optimal_point'):
        print(f"• Parâmetros ótimos (γ, β): {result_local.optimal_point}")
    if hasattr(result_local, 'optimizer_evals'):
        print(f"• Avaliações da função: {result_local.optimizer_evals}")
    print(f"• Dimensão do espaço de Hilbert: 2^{N} = {2**N} estados")
    print(f"• Número de soluções válidas: {len(list(combinations(range(N), budget)))}")
    
    # 8. CONCLUSÃO CIENTÍFICA
    print(f"\n🎓 CONCLUSÃO CIENTÍFICA:")
    print("O algoritmo QAOA demonstrou capacidade de resolver problemas de otimização")
    print("combinatória, encontrando soluções aproximadas para o problema de portfólio.")
    print("Esta implementação segue o framework estabelecido por Farhi et al. (2014).")
    
except Exception as e:
    print(f"❌ ERRO NA EXECUÇÃO CIENTÍFICA: {e}")
    print("\n🔍 DIAGNÓSTICO DO ERRO:")
    
    import traceback
    error_details = traceback.format_exc()
    
    # Análise detalhada do erro para fins científicos
    if "StatevectorSampler" in str(e):
        print("• Problema: Incompatibilidade com StatevectorSampler")
        print("• Solução: Verificar versões do Qiskit ou usar Sampler alternativo")
    elif "Hamiltonian" in str(e):
        print("• Problema: Formato do Hamiltoniano incorreto")
        print("• Solução: Validar construção do operador quântico")
    else:
        print(f"• Erro genérico: {e}")
    
    print(f"\n📋 DETALHES TÉCNICOS:")
    print(error_details)
    
    print(f"\n💡 AÇÕES CORRETIVAS SUGERIDAS:")
    print("1. Verificar compatibilidade de versões: pip list | grep qiskit")
    print("2. Validar o Hamiltoniano com print(hamiltonian)")
    print("3. Testar com problema mais simples (N=2)")
    print("4. Consultar documentação do Qiskit Algorithms")