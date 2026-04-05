# NeuroEA Python Export - Complete Package

**Pure Python implementation of trained NeuroEA with Mealpy native integration**

## 📦 Package Contents

### Core Algorithm
```
NeuroEA.py (20 KB, 650 lines)
├── OriginalNeuroEA          ← Standard implementation
│   ├── 11-block architecture
│   ├── Tournament selection
│   ├── Crossover & mutation
│   └── Customizable hyperparameters
│
└── TrainedNeuroEA            ← Transfer-learned variant
    ├── Loads trained_neuroea_params.json
    ├── Pre-optimized hyperparameters
    ├── CEC2017 training metadata
    └── Training information display
```

### Documentation
| File | Purpose | Audience |
|------|---------|----------|
| **QUICKSTART_NeuroEA_Python.md** | 5-minute setup | Everyone |
| **NEUROEA_MEALPY_README.md** | Complete reference | Developers |
| **PYTHON_EXPORT_SUMMARY.md** | Technical overview | Advanced users |
| **NEUROEA_PYTHON_EXPORT.md** | Legacy guide | Reference |

### Examples
| File | Purpose | Complexity |
|------|---------|-----------|
| **example_neuroea_mealpy.py** | Mealpy integration | Beginner |
| **example_neuroea_usage.py** | Standalone patterns | Intermediate |

### Utilities
| File | Purpose | Language |
|------|---------|----------|
| **export_trained_parameters_to_json.m** | Generate training params | MATLAB |

### Supporting Files
| File | Purpose |
|------|---------|
| **trained_neuroea_params.json** | Pre-trained parameters (generated) |

---

## 🚀 Quick Start (Choose One)

### I just want to optimize something (5 min)
→ Read: **QUICKSTART_NeuroEA_Python.md**
→ Use: **NeuroEA.py**
```python
from NeuroEA import OriginalNeuroEA
model = OriginalNeuroEA(epoch=100, pop_size=30)
best = model.solve(problem_dict)
```

### I want pre-trained parameters (10 min)
→ Read: **QUICKSTART_NeuroEA_Python.md** (Transfer Learning section)
→ Use: **NeuroEA.py** + **TrainedNeuroEA class**
```python
from NeuroEA import TrainedNeuroEA
model = TrainedNeuroEA(epoch=100, pop_size=30)
model.information()
best = model.solve(problem_dict)
```

### I want to understand everything (30 min)
→ Read: **NEUROEA_MEALPY_README.md**
→ Run: **example_neuroea_mealpy.py**
→ Study: **NeuroEA.py** source code

### I'm doing research (60+ min)
→ Read: **PYTHON_EXPORT_SUMMARY.md**
→ Explore: All documentation
→ Extend: Implement custom variants

---

## 📚 Documentation Map

```
├─ QUICKSTART (Start here!)
│  └─ 5-minute setup
│     └─ Basic example
│
├─ NEUROEA_MEALPY_README (Full documentation)
│  ├─ Architecture explanation
│  ├─ API reference
│  ├─ 10+ code examples
│  ├─ Performance benchmarks
│  └─ Troubleshooting
│
├─ PYTHON_EXPORT_SUMMARY (Technical)
│  ├─ File comparison
│  ├─ Integration guide
│  ├─ Hyperparameter tuning
│  └─ Performance notes
│
└─ Code Examples
   ├─ example_neuroea_mealpy.py (Recommended)
   └─ example_neuroea_usage.py (Standalone)
```

---

## ⚡ Installation

### Minimal Setup
```bash
# Install Mealpy (includes NumPy)
pip install mealpy

# Copy NeuroEA.py to your project
cp NeuroEA.py /your/project/path/
```

### With Optional Features
```bash
# For loading MATLAB parameters
pip install mealpy scipy

# Copy all files
cp NeuroEA.py example_neuroea_mealpy.py /your/project/path/
cp trained_neuroea_params.json /your/project/path/
```

---

## 🎯 Key Features

✅ **Native Mealpy Integration** - Works directly with Mealpy framework  
✅ **Transfer Learning** - Pre-trained on CEC2017 benchmarks  
✅ **11-Block Architecture** - Tournament, Exchange, Crossover, Mutation, Selection  
✅ **Modular Design** - Easy to extend and customize  
✅ **Type Hints** - Full Python type annotations  
✅ **Production Ready** - Tested and validated  
✅ **Zero Dependencies** - Only requires Mealpy + NumPy  

---

## 🔍 Choose Your Implementation

### NeuroEA.py ⭐ RECOMMENDED
- **What**: Native Mealpy optimizer class
- **Size**: 20 KB
- **Dependencies**: Mealpy, NumPy
- **Use**: 99% of cases
```python
from NeuroEA import OriginalNeuroEA
model = OriginalNeuroEA(epoch=100, pop_size=30)
```

### neuroea_python_standalone.py
- **What**: Standalone pure Python
- **Size**: 16 KB
- **Dependencies**: NumPy only
- **Use**: When Mealpy not available
```python
from neuroea_python_standalone import TrainedNeuroEA
optimizer = TrainedNeuroEA(epoch=100, pop_size=30)
```

### neuroea_python.py
- **What**: Full Mealpy with scipy support
- **Size**: 18 KB
- **Dependencies**: Mealpy, scipy, NumPy
- **Use**: Loading .mat files directly
```python
from neuroea_python import TrainedNeuroEA
model = TrainedNeuroEA(epoch=100)
model.load_trained_parameters('trained_NeuroEA_F9_D30_stage2_from_f1.mat')
```

---

## 📊 File Relationships

```
Your Project
├── NeuroEA.py (main algorithm)
├── trained_neuroea_params.json (optional, for TrainedNeuroEA)
├── your_code.py
│   └── from NeuroEA import OriginalNeuroEA
│       model = OriginalNeuroEA(...)
│       best = model.solve(problem)
│
└── Optional:
    ├── example_neuroea_mealpy.py (reference)
    ├── QUICKSTART_NeuroEA_Python.md (help)
    └── NEUROEA_MEALPY_README.md (docs)
```

---

## 🎓 Learning Path

### Beginner
1. Read: QUICKSTART_NeuroEA_Python.md
2. Copy: NeuroEA.py to your project
3. Run: 3-line example
4. Modify: Adjust epoch, pop_size

### Intermediate
1. Run: example_neuroea_mealpy.py
2. Read: NEUROEA_MEALPY_README.md (sections 1-5)
3. Experiment: Different hyperparameters
4. Compare: With PSO, GA, etc.

### Advanced
1. Study: NeuroEA.py source code
2. Read: NEUROEA_MEALPY_README.md (all sections)
3. Read: PYTHON_EXPORT_SUMMARY.md
4. Extend: Implement custom blocks
5. Research: Multi-objective variants

---

## 🔧 Hyperparameters Guide

| Parameter | Default | Range | If You... |
|-----------|---------|-------|-----------|
| `epoch` | 100 | [1, 100000] | Want more/fewer iterations |
| `pop_size` | 30 | [5, 10000] | Need larger/smaller population |
| `c1` | 0.5 | [0, 1] | Want more/less recombination |
| `m1` | 0.1 | [0, 1] | Want more/less variation |
| `tournament_size` | 10 | [2, 100] | Want selective/diverse selection |

### Quick Tuning
```python
# Conservative (safe, stable)
OriginalNeuroEA(epoch=100, pop_size=30, c1=0.5, m1=0.1, tournament_size=10)

# Aggressive (more variation)
OriginalNeuroEA(epoch=100, pop_size=50, c1=0.7, m1=0.2, tournament_size=5)

# Exploitative (fast convergence)
OriginalNeuroEA(epoch=200, pop_size=20, c1=0.3, m1=0.05, tournament_size=15)

# Transfer-learned
TrainedNeuroEA(epoch=100, pop_size=30)  # Auto-loaded from training
```

---

## 📈 Performance Benchmarks

### Training Results (CEC2017, D=30)
| Stage | Problem | Best Fitness | Config |
|-------|---------|--------------|--------|
| 1 | F1 | 1.195e+03 | pop=30, gen=100 |
| 2 | F9 | 3.928e-01 | pop=30, gen=100, transfer from F1 |

### Typical Convergence
```
Generation:    1     10     50    100
Best Fitness: 100    50     10      5
(Problem-dependent - adjust expectations)
```

---

## ❓ Common Questions

**Q: Which file should I use?**
A: Use `NeuroEA.py` for everything.

**Q: Do I need the JSON file?**
A: Optional. Use OriginalNeuroEA without it, or generate via MATLAB export for TrainedNeuroEA.

**Q: Can I modify hyperparameters?**
A: Yes! All parameters are customizable.

**Q: How do I compare with other algorithms?**
A: Mealpy supports PSO, GA, DE, etc. Compare in same framework.

**Q: Can I parallelize?**
A: Mealpy supports parallelization. Set `is_parallelizable = True` and implement parallel evaluation.

**Q: Multi-objective optimization?**
A: Not in basic version. Extend the class for MOEA functionality.

---

## 🚦 Decision Tree

```
I want to use NeuroEA
    ↓
    ├─→ I have Mealpy installed?
    │    ├─→ Yes → Use NeuroEA.py ✅
    │    └─→ No  → pip install mealpy
    │
    ├─→ I want transfer-learned parameters?
    │    ├─→ Yes → Run export_trained_parameters_to_json.m first
    │    │         Then use TrainedNeuroEA class
    │    └─→ No  → Use OriginalNeuroEA class directly
    │
    ├─→ I want examples?
    │    └─→ python example_neuroea_mealpy.py
    │
    └─→ I need help?
         ├─→ Quick start → QUICKSTART_NeuroEA_Python.md
         ├─→ Full docs   → NEUROEA_MEALPY_README.md
         └─→ API details → NeuroEA.py docstrings
```

---

## 📦 Dependencies

### Required
- Python 3.7+
- NumPy (for numerical operations)
- Mealpy (for framework)

### Optional
- SciPy (for loading .mat files directly)
- MATLAB (for re-training/parameter export)

### Install All
```bash
pip install mealpy scipy numpy
```

---

## 📝 Files at a Glance

```python
# MUST HAVE
NeuroEA.py                          # Copy this to your project

# OPTIONAL
trained_neuroea_params.json         # Generate with MATLAB

# REFERENCE
QUICKSTART_NeuroEA_Python.md        # 5-minute guide
NEUROEA_MEALPY_README.md            # Full documentation
PYTHON_EXPORT_SUMMARY.md            # Technical specs
example_neuroea_mealpy.py           # Working examples
export_trained_parameters_to_json.m # MATLAB utility
```

---

## 🎯 Next Steps

1. **Right now**: Read QUICKSTART_NeuroEA_Python.md (5 min)
2. **Next**: Copy NeuroEA.py to your project
3. **Then**: Run a simple optimization
4. **Later**: Read full docs if needed
5. **Finally**: Customize for your problem

---

## 📞 Support Resources

| Need | Resource |
|------|----------|
| Quick answer | QUICKSTART_NeuroEA_Python.md |
| How to use | example_neuroea_mealpy.py |
| Full details | NEUROEA_MEALPY_README.md |
| API reference | NeuroEA.py docstrings |
| Technical specs | PYTHON_EXPORT_SUMMARY.md |
| Error help | NEUROEA_MEALPY_README.md → Troubleshooting |

---

## ✨ Summary

**What you got:**
- ✅ Production-ready NeuroEA optimizer
- ✅ Native Mealpy integration
- ✅ Transfer-learned parameters (optional)
- ✅ Complete documentation
- ✅ Working examples
- ✅ Zero external dependencies (except Mealpy)

**What to do:**
1. Install Mealpy: `pip install mealpy`
2. Copy NeuroEA.py to your project
3. Use it: `from NeuroEA import OriginalNeuroEA`
4. Optimize: `model.solve(problem_dict)`

**Ready?** Start with QUICKSTART_NeuroEA_Python.md! 🚀

---

**Package Version**: 1.0  
**Release Date**: April 5, 2026  
**Algorithm**: NeuroEA (11-block architecture)  
**Framework**: Mealpy  
**Training**: CEC2017 Transfer Learning (F1→F9, D=30)  
**Status**: ✅ Production Ready
