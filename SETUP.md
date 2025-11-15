# Setup Guide

This guide will help you set up the environment and run the Search Algorithms project, including both Python scripts and Jupyter notebooks.

## 📋 Prerequisites

- **Python 3.7 or higher** (Python 3.8+ recommended)
- **pip** (Python package installer)
- **Git** (optional, for cloning the repository)

## 🔧 Installation Steps

### Step 1: Clone or Download the Project

If you have the project in a Git repository:
```bash
git clone <repository-url>
cd algorithm
```

Or simply navigate to the project directory if you already have it.

### Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment is recommended to avoid conflicts with other Python projects.

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - For numerical computations
- `matplotlib` - For creating visualizations
- `jupyter` - For running Jupyter notebooks
- `ipython` - Enhanced Python shell for notebooks
- `notebook` - Jupyter notebook interface

### Step 4: Verify Installation

Verify that all packages are installed correctly:

```bash
python -c "import numpy; import matplotlib; import jupyter; print('All packages installed successfully!')"
```

## 🚀 Running Python Scripts

### Method 1: Run from Project Root Directory

Navigate to the project root directory (`algorithm/`) and run scripts:

```bash
# Run Rastrigin experiments
python experiments/run_rastrigin.py

# Run Knapsack experiments
python experiments/run_knapsack.py

# Generate visualizations
python visualizations/plot_convergence.py
python visualizations/plot_robustness.py
python visualizations/plot_complexity.py
# ... and other visualization scripts
```

### Method 2: Set PYTHONPATH (If Import Errors Occur)

If you encounter import errors, set the PYTHONPATH:

**On Linux/Mac:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python experiments/run_rastrigin.py
```

**On Windows:**
```cmd
set PYTHONPATH=%PYTHONPATH%;%CD%
python experiments/run_rastrigin.py
```

**On Windows PowerShell:**
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
python experiments/run_rastrigin.py
```

## 📓 Running Jupyter Notebooks

### Step 1: Start Jupyter Notebook Server

From the project root directory (`algorithm/`), start Jupyter:

```bash
jupyter notebook
```

This will:
- Open Jupyter in your default web browser
- Show the file browser interface
- Allow you to navigate to the `notebooks/` folder

### Step 2: Open and Run Notebooks

1. Navigate to the `notebooks/` folder in the Jupyter interface
2. Open one of the following notebooks:
   - `01_Experiment_Rastrigin.ipynb` - Run Rastrigin experiments
   - `02_Experiment_Knapsack.ipynb` - Run Knapsack experiments
   - `03_Sandbox_Parameter_Tuning.ipynb` - Quick parameter testing
   - `04_Final_Report_Dashboard.ipynb` - View saved results

3. **Run cells sequentially:**
   - Click on a cell and press `Shift + Enter` to run it
   - Or use the "Run" button in the toolbar
   - Run cells from top to bottom in order

### Step 3: Notebook Execution Order

For experiment notebooks (`01_` and `02_`), follow this order:

1. **Cell 1 (Markdown)**: Read the explanation
2. **Cell 2 (Code)**: Setup & Imports - Run this first
3. **Cell 3 (Markdown)**: Configuration explanation
4. **Cell 4 (Code)**: Configuration - Modify parameters if needed
5. **Cell 5 (Markdown)**: Experiment explanation
6. **Cell 6 (Code)**: Run Experiments - This may take several minutes
7. **Cell 7 (Markdown)**: Save results explanation
8. **Cell 8 (Code)**: Save Results to CSV
9. **Cell 9+ (Code)**: Visualization cells - Run to generate plots

### Alternative: JupyterLab (Optional)

If you prefer JupyterLab interface:

```bash
pip install jupyterlab
jupyter lab
```

## 📁 Project Structure

Make sure your project structure looks like this:

```
algorithm/
├── algorithms/          # Algorithm implementations
├── problems/            # Problem definitions
├── experiments/          # Python experiment scripts
├── visualizations/      # Python visualization scripts
├── notebooks/           # Jupyter notebooks
├── results/             # Output directory (created automatically)
├── requirements.txt     # Dependencies file
├── SETUP.md            # This file
└── README.md           # Project documentation
```

## 🐛 Troubleshooting

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'algorithms'`

**Solution:**
1. Make sure you're running from the project root directory (`algorithm/`)
2. Set PYTHONPATH as shown in "Method 2" above
3. In notebooks, the first cell should handle path setup automatically

### Issue: Matplotlib Display Problems

**Problem:** Plots not showing in notebooks or scripts

**Solution:**
1. For notebooks: Make sure `%matplotlib inline` is in the first code cell
2. For scripts: Check matplotlib backend:
   ```bash
   python -c "import matplotlib; print(matplotlib.get_backend())"
   ```
3. On Linux without display: Use `matplotlib.use('Agg')` before importing pyplot

### Issue: Jupyter Not Found

**Problem:** `jupyter: command not found`

**Solution:**
1. Make sure virtual environment is activated
2. Reinstall: `pip install jupyter`
3. Try: `python -m jupyter notebook`

### Issue: Permission Errors

**Problem:** Permission denied when installing packages

**Solution:**
1. Don't use `sudo` with virtual environments
2. Make sure virtual environment is activated
3. On Windows, run terminal as Administrator if needed

### Issue: Notebook Kernel Not Starting

**Problem:** Kernel keeps restarting or won't start

**Solution:**
1. Check Python version: `python --version` (should be 3.7+)
2. Reinstall ipykernel: `pip install --upgrade ipykernel`
3. Restart Jupyter server

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Python 3.7+ is installed (`python --version`)
- [ ] Virtual environment is created and activated
- [ ] All packages installed (`pip list` shows numpy, matplotlib, jupyter)
- [ ] Can import modules: `python -c "from algorithms import pso; print('OK')"`
- [ ] Jupyter starts: `jupyter notebook` opens browser
- [ ] Notebooks can be opened and cells can be executed

## 📝 Notes

- **Execution Time**: Running full experiments may take 5-15 minutes depending on your machine
- **Results Directory**: The `results/` directory will be created automatically when you run experiments
- **Notebook Output**: Notebooks save both CSV files and PNG plots to the `results/` directory
- **Parameter Modification**: You can modify experiment parameters in the Configuration cells of notebooks

## 🆘 Getting Help

If you encounter issues not covered here:

1. Check the `README.md` for project overview
2. Review error messages carefully
3. Ensure all dependencies are installed correctly
4. Verify you're in the correct directory
5. Check Python and package versions

## 🎯 Quick Start Summary

```bash
# 1. Navigate to project directory
cd algorithm

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Python scripts
python experiments/run_rastrigin.py

# 5. OR run Jupyter notebooks
jupyter notebook
# Then open notebooks/01_Experiment_Rastrigin.ipynb
```

---

**Happy coding! 🚀**

