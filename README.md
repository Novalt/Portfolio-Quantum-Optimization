# Portfolio Quantum Optimization

Este projeto implementa um algoritmo QAOA (Quantum Approximate Optimization Algorithm) para otimização de portfólio, executado no hardware quântico da IBM.

## 📋 Descrição

O problema consiste em selecionar um subconjunto de ativos (por exemplo, 3 de 6) para minimizar o risco (covariância) e maximizar o retorno. O algoritmo QAOA é usado para resolver este problema de otimização combinatória.

Foram desenvolvidas duas versões principais:
- `mainX13IBM.py`: Versão simplificada e rápida.
- `mainX14IBM.py`: Versão com análise avançada, comparação com solução clássica e ordenação por qualidade.

## 🚀 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/Portfolio-Quantum-Optimization.git
   cd Portfolio-Quantum-Optimization


crie um ambiente virtual (opcional, mas recomendado):

bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instale as dependências:

bash
pip install -r requirements.txt


🔧 Configuração do IBM Quantum
Crie uma conta no IBM Quantum.

Obtenha seu token de API.

Execute o script de configuração:

bash
python env-IBM-Cloud-pyAuthentication.py
Ou configure manualmente:

python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="SEU_TOKEN")

▶️ Execução
Versão 14 (Recomendada)
bash
python src/mainX14IBM.py
Versão 13 (Mais rápida)
bash
python src/mainX13IBM.py
📊 Resultados
Os resultados serão exibidos no terminal, incluindo:

Distribuição por número de ativos selecionados

Top 5 soluções válidas

Comparação com a solução clássica ótima (na versão 14)

Métricas de desempenho

Configuração Ótima Encontrada
Penalidade: 35.0

Parâmetros QAOA: [0.7, 0.3, 0.5, 0.5]

Eficiência: ~26% de soluções válidas

Probabilidade da solução ótima: ~1.2%

🗂️ Estrutura de Arquivos
src/mainX13IBM.py: Código da versão 13.

src/mainX14IBM.py: Código da versão 14.

src/utils.py: Funções auxiliares (se houver).

requirements.txt: Dependências do projeto.

setup_ibm_quantum.py: Script para configurar o IBM Quantum.

README.md: Este arquivo.

⚙️ Personalização
Edite o dicionário CONFIG no início do arquivo mainX14IBM.py para ajustar:

Número de ativos (NUM_ATIVOS)

Número de ativos a selecionar (NUM_SELECIONAR)

Fator de penalidade (PENALIDADE_FACTOR)

Parâmetros do QAOA (PARAMETROS_FIXOS)

Número de shots (NUM_SHOTS)

📈 Exemplo de Saída
text
🚀 PROBLEMA DE PORTFÓLIO - 6 ATIVOS NO HARDWARE IBM (VERSÃO OTIMIZADA)
======================================================================
✅ CONFIGURAÇÃO OTIMIZADA COM BASE NOS TESTES ANTERIORES
   • Penalidade: 35.0
   • Parâmetros QAOA: [0.7, 0.3, 0.5, 0.5]
📊 PORTFÓLIO: 6 ativos, selecionar 3

... (outras saídas)

📊 MÉTRICAS FINAIS:
• Soluções válidas (3 ativos): 25.4%
• Shots válidos: 520/2048
• Solução clássica ótima encontrada pelo QAOA: ✅ SIM
• Probabilidade da solução ótima: 1.27%
• Desempenho: ✅ BOM
🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.