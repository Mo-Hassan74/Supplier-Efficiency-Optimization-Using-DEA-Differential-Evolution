
import numpy as np
import pandas as pd

NP = 100  # population size
F = 0.5  # scale factor
Cr = 0.9  # crossover rate
max_iter = 100 # maximum iterations


# --- Data of suppliers ---
def prepare_data():
    data = {
        'Supplier': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'PR': [290, 240, 300, 255, 295, 250, 245, 285, 270, 270, 285, 275],
        'LD': [7, 3, 4, 5, 10, 3, 7, 6, 6, 12, 3, 5],
        'RR': [3, 5, 6, 3, 8, 3, 4, 4, 6, 4, 5, 8],
        'SQ': [95, 98, 12, 100, 65, 110, 92, 73, 75, 81, 112, 85]
    }
    return pd.DataFrame(data)



# --- Fuzzification Function ---

def fuzzify_and_defuzzify_prices(df):
    prices = df['PR'].values
    sigma = np.std(prices)

    def defuzzify_triangular(a, b, c, max_attempts=1000):
        for _ in range(max_attempts):
            # fuzzification
            x = np.random.uniform(a, c)
            if x <= a:
                mu = 0.0
            elif a < x < b:
                mu = (x - a) / (b - a)
            elif x == b:
                mu = 1.0
            elif b < x < c:
                mu = (c - x) / (c - b)
            elif x >= c:
                mu = 0.0
            # defuzzification
            alpha = np.random.uniform(0, 1)
            # print(f"x = {x:.4f}, mu = {mu:.4f}, alpha = {alpha:.4f}") printing alpha-cut values and random x , calculated membership value for each x
            if alpha <= mu:
                return x
        return b  # if there is no accepted value after the 1000 attempts return the original crisp price

    defuzzified_prices = []
    for price in prices:
        a = price - sigma
        c = price + sigma
        b = price
        new_price = defuzzify_triangular(a, b, c)
        defuzzified_prices.append(new_price)

    df['PR'] = defuzzified_prices
    return df

# --- Population Initialization ---
def initialize_population(supplier_idx, NP, df):
    """Generate and normalize random weights for a specific supplier (12)"""
    population = []
    supplier_data = df.loc[supplier_idx, ['PR', 'LD', 'RR']]

    for _ in range(NP):
        while True:
            input_weights = np.random.uniform(0, 1, 3)
            output_weight = np.random.uniform(0, 1)

            denominator = np.dot(input_weights, supplier_data)
            if denominator > 0:
                normalized_weights = input_weights / denominator # for normalization constarint
                population.append(list(normalized_weights) + [output_weight])
                # population.append([*normalized_weights, output_weight])# (*-> for unbacking to flat the input weights in same list with output weight
                break

    return np.array(population)


# --- Fitness Evaluation ---
def evaluate_fitness(weights, supplier_idx, df):
    """Calculate DEA efficiency and constraint violations"""
    Z1, Z2, Z3, W = np.clip(weights, 0, 1) # make sure positive weights
    supplier_data = df.loc[supplier_idx, ['PR', 'LD', 'RR', 'SQ']]

    # Calculate current supplier's efficiency
    weighted_input = Z1 * supplier_data['PR'] + Z2 * supplier_data['LD'] + Z3 * supplier_data['RR']
    if weighted_input <= 0:
        return -np.inf, np.inf
    efficiency = (W * supplier_data['SQ']) / weighted_input

    # Check constraints for all suppliers
    violations = 0
    for _, row in df.iterrows():
        input_n = Z1 * row['PR'] + Z2 * row['LD'] + Z3 * row['RR']
        if input_n <= 0:
            return -np.inf, np.inf

        constraint = (W * row['SQ']) / input_n - 1
        if constraint > 0:
            violations += constraint  # how much this solution violate the constraints

    return efficiency, violations


# --- Mutation Operator ---
def mutate(population, F, i):
    """Generate mutant vector"""
    candidates = [idx for idx in range(len(population)) if idx != i]
    a, b, c = np.random.choice(candidates, 3, replace=False)
    return np.clip(population[a] + F * (population[b] - population[c]), 0, 1)


# --- Crossover Operator ---
def crossover(target, mutant, Cr):
    trial = np.copy(target)
    """
    (np.random.randint(0, 4) == np.arange(4)
    to make sure at least one value will be taken from mutant
    arrange return list of [0,1,2,3]
    """
    cross_points = (np.random.rand(4) < Cr) | (np.random.randint(0, 4) == np.arange(4))
    for i in range(4):  # for each of the 4 weights (Z1,Z2,Z3,W)
        if cross_points[i]:  # if the position is  true
            trial[i] = mutant[i]  # take value from mutant

    return trial


# --- Differential Evolution Core ---
def optimize_supplier(supplier_idx, df):
    """Run DE optimization for a single supplier"""
    population = initialize_population(supplier_idx, NP, df)

    for _ in range(max_iter):
        for i in range(NP):
            mutant = mutate(population, F, i) # when we choose 3 individauls we must't choose i vector.
            trial = crossover(population[i], mutant, Cr) # we will compare it with target vector i to choose the best of them
            trial_eff, trial_viol = evaluate_fitness(trial, supplier_idx, df) # efficiency and violation of target vector
            target_eff, target_viol = evaluate_fitness(population[i], supplier_idx, df)## efficiency and violation of trial vector
            # selection
            if trial_viol == 0:
                if target_viol > 0 or trial_eff > target_eff:
                    population[i] = trial
            elif trial_viol < target_viol:
                population[i] = trial

    # Find best solution
    best_weights, best_efficiency = None, -np.inf
    best_violations = np.inf

    for weights in population:
        efficiency, violations = evaluate_fitness(weights, supplier_idx, df)
        if violations == 0 and efficiency > best_efficiency:
            best_efficiency, best_weights = efficiency, weights
        elif violations < best_violations:
            best_violations, best_weights = violations, weights

    return best_weights, best_efficiency

def main():
    df_full = prepare_data()
    print("Choose dataset size to run the DEA-DE algorithm:")
    print("1. Small (5 suppliers)")
    print("2. Medium (8 suppliers)")
    print("3. Full (12 suppliers)")

    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == '1':# first 5 supplier
        df = df_full.iloc[:5].reset_index(drop=True)
        print("\nRunning DEA-DE on Small Dataset (5 suppliers)\n")
    elif choice == '2':# first 8 supplier
        df = df_full.iloc[:8].reset_index(drop=True)
        print("\nRunning DEA-DE on Medium Dataset (8 suppliers)\n")
    else:
        df = df_full  # full dataset(12 supplier)
        print("\nRunning DEA-DE on Full Dataset (12 suppliers)\n")

    # Apply fuzzification on the prices
    df = fuzzify_and_defuzzify_prices(df)

    # Print Fuzzified PR values
    #print("\nFuzzified PR values:")
    #print(df[['Supplier', 'PR']])

    results = []
    for supplier_idx in range(len(df)):
        weights, efficiency = optimize_supplier(supplier_idx, df)
        results.append({
            'Supplier': df.loc[supplier_idx, 'Supplier'],
            'Z1': weights[0],
            'Z2': weights[1],
            'Z3': weights[2],
            'W': weights[3],
            'Efficiency': efficiency
        })

    # Print results
    results_df = pd.DataFrame(results).sort_values('Efficiency', ascending=False).reset_index(drop=True)
    print("\nOptimized weights and efficiency for all suppliers:")
    print(results_df)

    print("\nOptimal Weights for Top Supplier:")
    top_supplier = results_df.iloc[0]
    print(top_supplier[['Supplier', 'Z1', 'Z2', 'Z3', 'W', 'Efficiency']].to_string())


if __name__ == "__main__":
    main()