# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoFilter is a genetic algorithm framework for optimizing optical thin-film filter designs for spectral imaging systems. It uses evolutionary optimization to design 4x4 arrays of multi-layer filters (16 filters, each with 20 alternating SiO2/TiO2 layers) that maximize spectral discrimination for hyperspectral image reconstruction.

## Directory Structure

```
EvoFilter/
├── src/                       # Core Python source code
│   ├── main.py               # Genetic algorithm main entry
│   ├── fitness.py            # Base fitness evaluation (metrics computation)
│   ├── fitness_functions.py  # Modular fitness function framework
│   ├── initial.py            # Filter initialization
│   ├── population_ops.py     # Genetic operators
│   ├── tmm_core.py           # Transfer Matrix Method
│   ├── cal_response.py       # Spectral response calculation
│   ├── cal_mask.py           # Mask generation
│   ├── evaluate.py           # Reconstruction evaluation
│   └── ...
├── prework_src/
│   ├── nk_collection.py      # calculate refractor index
│   └── select_spe.py         # select spectral curve
├── scripts/                  # Shell scripts
├── data/                     # Data files (.pkl, .mat)
├── results/                  # Experiment results
├── mask/                     # Generated masks
├── recon_results/            # Reconstruction outputs
├── plots/                    # Generated plots
├── lib/                      # Third-party libraries
├── exp_details/              # Experiment records
└── config.yaml               # Configuration file
```

## Common Commands

**Run from project root directory:**

**List available fitness functions:**
```bash
python src/main.py --list_fitness
```

**Run genetic algorithm with default settings:**
```bash
python src/main.py
```

**Run with specific fitness function and correlation mode:**
```bash
python src/main.py --fitness_type information --corr_mode row
python src/main.py --fitness_type uncorrelation --corr_mode column
python src/main.py --fitness_type combined --w_info 1.0 --w_uncorr 0.5 --corr_mode both
```

**Generate mask from optimized filter design:**
```bash
python src/cal_mask.py  # Edit thick_dir and mask_dir in the script first
```

**Evaluate reconstruction quality:**
```bash
python src/evaluate.py -t ../dataset/TSA_simu_data/Truth/ -m mask/mask.mat -o recon_results
```

## Fitness Function Framework

### Available Fitness Functions (`--fitness_type`)

| Type | Description | Use Case |
|:-----|:------------|:---------|
| `information` | Maximize D-Optimality (log det) | Focus on reconstruction quality |
| `uncorrelation` | Minimize filter correlation | Focus on spectral diversity |
| `stability` | Minimize condition number | Focus on numerical robustness |
| `combined` | Weighted combination (default) | Balance multiple objectives |
| `weighted_sum` | Normalized weighted sum | Interpretable contributions |
| `pareto` | Pareto front fitness | Multi-objective optimization |

### Correlation Mode (`--corr_mode`)

| Mode | Matrix Size | Physical Meaning |
|:-----|:------------|:-----------------|
| `row` | 16×16 | Filter-to-filter correlation (spectral independence) |
| `column` | W×W | Wavelength-to-wavelength correlation (spectral resolution) |
| `both` | Both | Takes the worse score (bucket effect) |

### Adding Custom Fitness Functions

1. Create a new class in `src/fitness_functions.py`
2. Inherit from `BaseFitnessFunction`
3. Implement `calculate()` method
4. Register with `@register_fitness("name")` decorator

```python
@register_fitness("custom")
class CustomFitness(BaseFitnessFunction):
    def calculate(self, eval_result):
        # Your fitness logic here
        return fitness_value, details_dict
```

## Architecture

### Core Optimization Pipeline

1. **src/main.py** - `GeneticAlgorithm` class orchestrates the evolutionary process:
   - Tournament selection, elitism (top 2), crossover, and adaptive mutation
   - Modular fitness function selection via `--fitness_type`
   - Saves best individuals per generation to `results/<timestamp>/`

2. **src/fitness_functions.py** - Modular fitness function framework:
   - `BaseFitnessFunction`: Abstract base class with common penalty methods
   - `InformationFitness`, `UncorrelationFitness`, `StabilityFitness`: Single-objective
   - `CombinedFitness`: Multi-objective weighted combination
   - Registry pattern for easy extension

3. **src/fitness.py** - `FitnessEvaluator` computes raw metrics:
   - **Uncorrelation**: 1 - max(|off-diagonal correlation|), supports row/column/both modes
   - **Information**: log(det(F @ Cov_s @ F.T)) - D-optimality criterion
   - **Stability**: 1 - normalized log10(condition number)
   - **Transmittance**: Mean and uniformity of filter responses

4. **src/initial.py** - `FilterInitializer` generates filter designs:
   - Each filter: 20 layers, thickness [20, 200] nm per layer, total [400, 2000] nm
   - Strategies: random, QWOT, Fabry-Perot cavity

5. **src/population_ops.py** - `EvolutionaryOperator` provides genetic operations

### Physics Simulation

6. **src/tmm_core.py** - Transfer Matrix Method for thin-film optics
7. **src/cal_response.py** - Computes spectral responses using TMM

### Reconstruction & Evaluation

8. **src/evaluate.py** - FISTA solver, PSNR/SSIM metrics
9. **src/cal_mask.py** - Converts thickness arrays to mask format

## Configuration

### config.yaml Structure

```yaml
ga:
  pop_size: 100
  generations: 200
  mutation_rate: 0.2
  crossover_rate: 0.8
  num_workers: 32

fitness:
  fitness_type: combined    # information, uncorrelation, stability, combined, etc.
  corr_mode: row            # row, column, both
  w_info: 1.0               # Weights for combined type
  w_uncorr: 0.0
  w_stability: 0.0

penalties:
  trans_threshold: 0.5
  lambda_trans: 3000.0
  corr_threshold: 0.3
  lambda_corr: 20.0
  lambda_uniformity: 20.0
  cond_threshold: 100.0
  lambda_stability: 2.0
```

## Key Data Files

- `data/nk_map.pkl` - Wavelength-dependent refractive indices for SiO2/TiO2 (400-700nm)
- `data/database_spe.mat` - Spectral library for computing information score covariance
- Results saved in `results/<timestamp>/` with evolution logs, best individuals (.npy), and plots
