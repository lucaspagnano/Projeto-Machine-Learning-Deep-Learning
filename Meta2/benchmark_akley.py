import numpy as np
from SwarmPackagePy import pso, gwo 
import math

np.set_printoptions(precision=2, suppress=True)

# Define a função de Ackley
def ackley(solution):
    n = len(solution)
    sum1 = np.sum(solution**2)
    sum2 = np.sum(np.cos(2 * np.pi * solution))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + 20 + np.exp(1)

# --- Configurações da Simulação ---
D = 3
n_agents = 30     
n_iterations = 100  

lb = [-32] * D  
ub = [32] * D   
# ------------------------------------

print(f"--- Iniciando Benchmark (Dimensão={D}) ---")
print(f"Agentes: {n_agents} | Iterações: {n_iterations}")

# --- Executar o PSO (Particle Swarm Optimization) ---
print("\nA executar PSO...")
pso_optimizer = pso(n_agents, ackley, lb, ub, D, n_iterations)

# Apanhar a melhor SOLUÇÃO do atributo que descobrimos
best_solution_pso = pso_optimizer._sw__Gbest
# Calcular o fitness dessa solução
best_fitness_pso = ackley(best_solution_pso)

print(f"PSO - Melhor Fitness (Valor da função): {best_fitness_pso:.2f}")
print(f"PSO - Melhor Solução (Posição): {best_solution_pso}")

# --- Executar o GWO (Gray Wolf Optimization) ---
print("\nA executar GWO...")
gwo_optimizer = gwo(n_agents, ackley, lb, ub, D, n_iterations)

# Apanhar a melhor SOLUÇÃO do atributo que descobrimos
best_solution_gwo = gwo_optimizer._sw__Gbest
# Calcular o fitness dessa solução
best_fitness_gwo = ackley(best_solution_gwo)

print(f"GWO - Melhor Fitness (Valor da função): {best_fitness_gwo:.2f}")
print(f"GWO - Melhor Solução (Posição): {best_solution_gwo}")

print("\n--- Benchmark Concluído ---")