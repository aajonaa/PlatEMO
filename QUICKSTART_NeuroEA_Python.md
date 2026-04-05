# Quick Start: NeuroEA Python Export

## TL;DR - 5 Minute Setup

### Step 1: Copy File
Copy `NeuroEA.py` to your project

### Step 2: Install Mealpy
```bash
pip install mealpy numpy
```

### Step 3: Use It
```python
from NeuroEA import OriginalNeuroEA
from mealpy import FloatVar
import numpy as np

# Define your optimization problem
problem_dict = {
    "bounds": FloatVar(n_vars=30, lb=(-10.,)*30, ub=(10.,)*30, name="x"),
    "obj_func": lambda x: np.sum(x**2),  # Your objective function
    "minmax": "min",
}

# Create optimizer
model = OriginalNeuroEA(epoch=100, pop_size=30)

# Solve
best = model.solve(problem_dict)
print(f"Best solution: {best.solution}")
print(f"Best fitness: {best.target.fitness}")
```

Done! 🎉

---

## For Transfer-Learned Parameters

### Extra Step 1: Export from MATLAB
```matlab
% In MATLAB, PlatEMO directory:
export_trained_parameters_to_json
```

This generates `trained_neuroea_params.json`

### Extra Step 2: Use TrainedNeuroEA
```python
from NeuroEA import TrainedNeuroEA

# Uses pre-trained parameters from CEC2017
model = TrainedNeuroEA(epoch=100, pop_size=30)

# See what was trained
model.information()

# Solve
best = model.solve(problem_dict)
```

---

## File Purposes

| File | Use | When |
|------|-----|------|
| **NeuroEA.py** | Main algorithm | Always |
| TrainedNeuroEA (in NeuroEA.py) | Pre-trained params | Have trained_neuroea_params.json |
| example_neuroea_mealpy.py | See examples | Learning the API |
| NEUROEA_MEALPY_README.md | Full docs | Need detailed info |

---

## Hyperparameter Meanings

| Parameter | What it does | Try |
|-----------|--------------|-----|
| `epoch` | Iterations | 50-1000 |
| `pop_size` | Population size | 20-100 |
| `c1` | Crossover rate | 0.3-0.8 |
| `m1` | Mutation rate | 0.05-0.2 |
| `tournament_size` | Tournament size | 5-20 |

---

## Common Questions

**Q: What's the difference between OriginalNeuroEA and TrainedNeuroEA?**
A: TrainedNeuroEA loads pre-optimized parameters from CEC2017 training. Start with OriginalNeuroEA and customize, or use TrainedNeuroEA for pre-trained defaults.

**Q: Do I need MATLAB?**
A: No, unless you want to re-train or generate new trained parameters. The NeuroEA.py file works standalone.

**Q: Can I use it without Mealpy?**
A: Yes, use `neuroea_python_standalone.py` instead (no dependencies).

**Q: How do I compare with PSO/GA?**
```python
from mealpy import PSO, GA, NeuroEA

models = [
    NeuroEA.OriginalNeuroEA(epoch=100, pop_size=30),
    PSO.OriginalPSO(epoch=100, pop_size=30),
    GA.BaseGA(epoch=100, pop_size=30),
]

for model in models:
    best = model.solve(problem_dict)
    print(f"{model.__class__.__name__}: {best.target.fitness:.6e}")
```

**Q: Can I use it for multi-objective?**
A: Not in the basic version. You'd need to modify the `evolve()` method or use MOEA variants.

---

## Performance Expectations

### Training Results (D=30)
- **Stage 1 (F1)**: ~1200 (fitness)
- **Stage 2 (F9)**: ~0.39 (offset from optimum)

### Typical Convergence
```
Epoch 1:   fitness ≈ 100
Epoch 10:  fitness ≈ 50
Epoch 50:  fitness ≈ 10
Epoch 100: fitness ≈ 5
```

(Depends heavily on your problem and parameters)

---

## Common Mistakes to Avoid

❌ **Wrong**: Forgetting to import Mealpy
```python
from NeuroEA import OriginalNeuroEA
# ERROR: Mealpy not imported!
```

✅ **Correct**:
```python
from NeuroEA import OriginalNeuroEA
from mealpy import FloatVar  # IMPORTANT!
```

---

❌ **Wrong**: Not using FloatVar
```python
problem = {
    "bounds": [(-10, 10)] * 30,  # This won't work
    "obj_func": sphere,
}
```

✅ **Correct**:
```python
from mealpy import FloatVar
problem = {
    "bounds": FloatVar(n_vars=30, lb=-10, ub=10),
    "obj_func": sphere,
}
```

---

❌ **Wrong**: Forgetting required parameters
```python
model = OriginalNeuroEA()  # Uses defaults - might not fit your problem
```

✅ **Correct**:
```python
model = OriginalNeuroEA(epoch=100, pop_size=50)  # Explicit configuration
```

---

## Next: Run Example

```bash
python example_neuroea_mealpy.py
```

This will show:
1. Basic usage
2. Transfer learning usage
3. Hyperparameter comparison

---

## Read More

- **NEUROEA_MEALPY_README.md** - Complete documentation
- **example_neuroea_mealpy.py** - Runnable examples
- **NeuroEA.py** - Source code with docstrings

---

## Support

**Algorithm not converging?**
- Try increasing `epoch` (more generations)
- Increase `pop_size` (larger population)
- Adjust `c1`, `m1` (crossover/mutation rates)

**Bad results?**
- Check your objective function is correct
- Verify bounds are appropriate
- Try different hyperparameters
- Run multiple times (stochastic)

**Need help?**
- Check docstrings in NeuroEA.py
- Read example_neuroea_mealpy.py
- Review NEUROEA_MEALPY_README.md

---

## Modern Python Style

NeuroEA.py follows modern Python best practices:

✅ Type hints
✅ Comprehensive docstrings  
✅ PEP 8 compliant  
✅ Mealpy standard interface  
✅ Proper error handling  
✅ Parameter validation  

---

**Ready to optimize? Start with Step 1 above!** 🚀
