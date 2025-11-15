# Project 01 - Search Algorithms

A project implementing and comparing **8 optimization algorithms** on **2 problems**: Rastrigin (continuous) and Knapsack (discrete).

## 📋 Project Description

This project is an implementation for the **Introduction to AI** course, including:
- **8 search/optimization algorithms**: PSO, ABC, FA, CS, GA, HC, ACO, SA
- **2 benchmark problems**: 
  - Rastrigin Function (continuous optimization)
  - 0/1 Knapsack Problem (discrete optimization)
- **Performance analysis**: Convergence analysis, robustness testing, scalability evaluation

## 🗂️ Project Structure

```
algorithm/
├── algorithms/          # Source code of 8 algorithms
│   ├── pso.py          # Particle Swarm Optimization
│   ├── abc.py          # Artificial Bee Colony
│   ├── fa.py           # Firefly Algorithm
│   ├── cs.py           # Cuckoo Search
│   ├── ga.py           # Genetic Algorithm
│   ├── hc.py           # Hill Climbing
│   ├── aco.py          # Ant Colony Optimization
│   └── sa.py           # Simulated Annealing
├── problems/            # Problem definitions
│   ├── rastrigin.py    # Rastrigin function
│   └── knapsack.py     # Knapsack problem
├── experiments/         # Experiment scripts
│   ├── run_rastrigin.py
│   └── run_knapsack.py
├── visualizations/      # Plotting scripts
│   ├── plot_convergence.py
│   ├── plot_robustness.py
│   ├── plot_rastrigin_3d.py
│   ├── plot_complexity.py
│   ├── plot_tradeoff.py
│   └── plot_single_algo_convergence.py
├── results/            # Experiment results
│   ├── *.csv          # Summary and raw data files
│   ├── convergence/    # Convergence curve plots
│   ├── robustness/    # Robustness (box plot) visualizations
│   ├── rastrigin_3d/  # 3D surface and contour plots for Rastrigin
│   ├── complexity/    # Complexity analysis plots
│   ├── tradeoff/      # Time-quality tradeoff plots
│   └── single_algo_convergence/  # Individual algorithm convergence plots
├── notebooks/          # Jupyter notebooks (if any)
├── README.md
├── FITNESS_EXPLANATION.md
└── TODO.md
```

## 📊 Implemented Algorithms

### Multi-purpose Algorithms (Continuous & Discrete)
The following 6 algorithms can solve **BOTH** Rastrigin (continuous) and Knapsack (discrete) problems:
1. **PSO** - Particle Swarm Optimization
2. **ABC** - Artificial Bee Colony
3. **FA** - Firefly Algorithm
4. **CS** - Cuckoo Search
5. **GA** - Genetic Algorithm
6. **HC** - Hill Climbing

**Note**: Each algorithm has 2 versions:
- `_continuous` version: Used for Rastrigin problem (continuous optimization)
- `_discrete` version: Used for Knapsack problem (discrete optimization)
  - PSO, ABC, FA, CS: Use sigmoid method to convert continuous solutions to binary
  - GA: Use appropriate crossover/mutation for each problem type
  - HC: Use appropriate neighbor search strategy for each problem type

### Specialized Algorithms for Discrete Optimization (Knapsack)
The following 2 algorithms are only implemented for Knapsack problem:
1. **ACO** - Ant Colony Optimization
2. **SA** - Simulated Annealing

## 🔧 Requirements

- **Python 3.7+**
- **NumPy** (computation)
- **Matplotlib** (visualization)

### Library Installation

```bash
pip install numpy matplotlib
```

## 🚀 How to Run

### Step 1: Run Experiments

**Note**: This process will take a few minutes (approximately 5-15 minutes depending on machine configuration).

```bash
# Run experiments for Rastrigin Function
python experiments/run_rastrigin.py

# Run experiments for Knapsack Problem
python experiments/run_knapsack.py
```

### Step 2: Generate Plots

```bash
# Plot convergence curves
python visualizations/plot_convergence.py

# Plot robustness (box plots)
python visualizations/plot_robustness.py

# Plot heatmap & contour of Rastrigin function
python visualizations/plot_rastrigin_3d.py

# Plot complexity analysis
python visualizations/plot_complexity.py

# Plot time-quality tradeoff
python visualizations/plot_tradeoff.py

# Plot individual algorithm convergence
python visualizations/plot_single_algo_convergence.py
```

### Step 3: View Results

- **All results**: In the `results/` directory
  - CSV data (raw data and summary)
  - PNG plots (visualizations)

## 📈 Experiments Performed

### Rastrigin Function
- **Algorithms tested**: 6 algorithms (PSO, ABC, FA, CS, GA, HC)
- **Dimensions**: 10, 30
- **Number of runs**: 20 (for each algorithm)
- **Population size**: 50
- **Max iterations**: 1000

### Knapsack Problem
- **Algorithms tested**: 8 algorithms (PSO, ABC, FA, CS, GA, HC, ACO, SA)
- **Problem sizes**: 20 items, 50 items
- **Number of runs**: 20 (for each algorithm)
- **Population size**: 50
- **Max iterations**: 1000

## 📊 Output Results

### In the `results/` directory:

**CSV Files (Root directory):**
- `rastrigin_summary.csv` - Summary statistics (mean, std, time)
- `rastrigin_convergence.csv` - Convergence data by iteration
- `rastrigin_raw_fitness.csv` - Raw data from multiple runs
- `knapsack_summary.csv` - Summary statistics
- `knapsack_convergence.csv` - Convergence data
- `knapsack_raw_fitness.csv` - Raw data

**Subdirectories (PNG Visualizations):**

- **`convergence/`** - Convergence curve plots showing algorithm performance over iterations for different problem sizes and dimensions

- **`robustness/`** - Box plot visualizations demonstrating algorithm stability and variance across multiple runs

- **`rastrigin_3d/`** - 3D surface plots, heatmaps, and contour plots visualizing the Rastrigin function landscape

- **`complexity/`** - Complexity analysis plots comparing algorithm performance with respect to problem size

- **`tradeoff/`** - Time-quality tradeoff plots showing the relationship between computation time and solution quality

- **`single_algo_convergence/`** - Individual algorithm convergence plots for detailed per-algorithm analysis

## 🎯 Rastrigin Function

The Rastrigin function is a popular benchmark function in optimization, with the form:

```
f(x) = 10n + Σ[x_i² - 10cos(2πx_i)]
```

- **Domain**: x_i ∈ [-5.12, 5.12]
- **Global minimum**: f(0, 0, ..., 0) = 0
- **Characteristics**: Highly multimodal (many local optima)

## 🎒 Knapsack Problem

The 0/1 Knapsack problem:
- **Input**: n items with weight and value
- **Constraint**: Total weight ≤ capacity
- **Objective**: Maximize total value

## 🔬 Analysis

The project performs the following analyses:

1. **Convergence Analysis**: Evaluate convergence speed of algorithms
2. **Robustness Testing**: Test stability across 20 runs
3. **Scalability Evaluation**: Compare performance with different problem sizes
4. **Statistical Comparison**: Mean, Standard Deviation, Min/Max fitness

## 📚 References

### Algorithms
- Kennedy & Eberhart (1995) - Particle Swarm Optimization
- Karaboga (2005) - Artificial Bee Colony
- Yang (2008) - Firefly Algorithm
- Yang & Deb (2009) - Cuckoo Search
- Goldberg (1989) - Genetic Algorithms
- Dorigo (1992) - Ant Colony Optimization
- Kirkpatrick et al. (1983) - Simulated Annealing

### Problems
- Rastrigin (1974) - Systems of Extremal Control
- Knapsack Problem - Classic NP-Complete problem

## 👤 Author

Project developed for **Introduction to AI - HCMUS** course

## 📝 Notes

- All code is written from scratch using only **NumPy** (no optimization libraries like scipy, scikit-learn, deap, etc.)
- Source code follows the function signature standards specified in requirements
- Algorithm parameters have been adjusted to suit each problem

## 🐛 Troubleshooting

**If encountering import errors:**
```bash
# Run from project_01/ directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**If matplotlib cannot display:**
```bash
# Check backend
python -c "import matplotlib; print(matplotlib.get_backend())"
```

## 📧 Contact

If you have questions or issues, please open an issue or contact via email.

---

**Good luck with your experiments! 🚀**

