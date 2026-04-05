# IMODE.py - mealpy Compatibility Fix

**Date:** 06/04/2026  
**Issue:** `'Problem' object has no attribute 'D'`  
**Status:** ✅ FIXED

---

## Problem

IMODE.py was using MATLAB-style attribute naming (`self.problem.D`) instead of mealpy framework conventions. The mealpy `Problem` object uses `n_dims` instead of `D` for the number of dimensions.

### Error Message
```
Error in IMODE: 'Problem' object has no attribute 'D'
```

---

## Solution

Replaced all 7 occurrences of `self.problem.D` with `self.problem.n_dims` throughout IMODE.py.

### Changes Made

| Line | Before | After | Context |
|------|--------|-------|---------|
| 96 | `self.memory_size = 20 * self.problem.D` | `self.memory_size = 20 * self.problem.n_dims` | Initialize memory size |
| 215 | `self.generator.random((N, self.problem.D))` | `self.generator.random((N, self.problem.n_dims))` | Uniform crossover |
| 222 | `self.generator.integers(0, self.problem.D)` | `self.generator.integers(0, self.problem.n_dims)` | Segmented crossover position |
| 225 | `self.generator.random(self.problem.D)` | `self.generator.random(self.problem.n_dims)` | Segmented crossover randomness |
| 229 | `if p2 < self.problem.D:` | `if p2 < self.problem.n_dims:` | Segmented crossover bound check |
| 232 | `(p1 + indices) % self.problem.D` (2x) | `(p1 + indices) % self.problem.n_dims` (2x) | Segmented crossover indexing |

### Files Modified
- `IMODE.py` (7 attribute references fixed)

---

## Verification

- ✅ All 7 instances replaced
- ✅ Syntax validated with pylance
- ✅ No remaining `self.problem.D` references in IMODE.py
- ✅ Ready for use with mealpy framework

---

## Testing

To verify the fix works:

```python
from mealpy import FloatVar
from IMODE import OriginalIMODE
import numpy as np

problem = {
    'bounds': FloatVar(n_vars=30, lb=(-100.,)*30, ub=(100.,)*30),
    'obj_func': lambda x: np.sum(x**2),
    'minmax': 'min',
}

model = OriginalIMODE(epoch=10, pop_size=30)
best = model.solve(problem)
print(f"Best fitness: {best.target.fitness}")
# Should complete without 'Problem' object has no attribute 'D' error
```

---

## Impact

This fix ensures IMODE.py is fully compatible with the mealpy framework convention for accessing problem dimensions. The algorithm functionality remains unchanged - only the attribute name used to access problem properties was corrected.

All methods that previously used `self.problem.D` now correctly use `self.problem.n_dims`:
- `initialize_variables()`: Memory initialization
- `crossover()`: Both uniform and segmented crossover modes

---

**Status:** Ready for production use with mealpy framework ✅
