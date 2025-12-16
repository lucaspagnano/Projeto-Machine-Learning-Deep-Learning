import re
import pandas as pd

# --- CONFIGURAÇÃO ---
INPUT_FILE = 'output_otimizacao.txt'  # Nome do seu ficheiro .txt com o log
OUTPUT_FILE = 'Resultados_Recuperados.xlsx'

def parse_log_to_excel(input_path, output_path):
    data = []
    
    # Regex flexível para capturar os formatos que usámos anteriormente:
    # Captura: Neurons (int), Drop (float), LR (float), Acc (float ou %)
    # Exemplo alvo: "Neurons=128, Drop=0.25, LR=0.001 => Acc=0.85"
    regex_pattern = re.compile(r"Neurons=(\d+).*?Drop(?:out)?=([\d\.]+).*?LR=([\d\.]+).*?Acc(?:uracy)?=([\d\.]+%?)", re.IGNORECASE)

    print(f"--- A ler ficheiro: {input_path} ---")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        count = 0
        for line in lines:
            match = regex_pattern.search(line)
            if match:
                neurons = int(match.group(1))
                dropout = float(match.group(2))
                lr = float(match.group(3))
                
                # Tratar Accuracy (remover % se existir e converter)
                acc_str = match.group(4).replace('%', '')
                acc = float(acc_str)
                
                # Se o valor estiver em percentagem (ex: 85.0), converter para 0.85
                # Se já estiver em 0.85, mantém. (Assumindo threshold de 1.0)
                if acc > 1.0:
                    acc = acc / 100.0

                # Tentar identificar o algoritmo pelo texto da linha (Opcional)
                algo = "GWO" if "GWO" in line else ("Random" if "Random" in line else "Optimization")

                data.append({
                    'Algorithm': algo,
                    'Neurons': neurons,
                    'Dropout': dropout,
                    'Learning_Rate': lr,
                    'Val_Accuracy': acc
                })
                count += 1
        
        if count == 0:
            print("AVISO: Nenhuma linha correspondente encontrada. Verifique o conteúdo do txt.")
        else:
            # Criar DataFrame e Exportar
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
            print(f"Sucesso! {count} registos recuperados.")
            print(f"Ficheiro guardado como: {output_path}")
            print("\nPrimeiras 5 linhas recuperadas:")
            print(df.head())

    except FileNotFoundError:
        print(f"ERRO: O ficheiro '{input_path}' não foi encontrado.")

# Executar a função
if __name__ == "__main__":
    parse_log_to_excel(INPUT_FILE, OUTPUT_FILE)