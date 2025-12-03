# coord-transform-package

A ready-to-use Python package for coordinate transformations between Cartesian, Cylindrical, and Spherical systems.

---

## Project structure

```
coord-transform-package/
├── README.md
├── pyproject.toml
├── LICENSE
├── src/
│   └── coord_transform/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── examples/
│   └── plot_demo.py
├── tests/
│   └── test_core.py
└── .github/workflows/ci.yml
```

---

## README.md (content)

```markdown
# coord-transform

Lightweight, well-tested Python package for fast, vectorized coordinate transforms between
Cartesian, Cylindrical and Spherical coordinates.

Features
- Cartesian ↔ Cylindrical
- Cartesian ↔ Spherical
- Cylindrical ↔ Spherical (via conversions)
- Fully vectorized using NumPy (works with scalars, lists and arrays)
- Thorough docstrings and unit tests
- Example 3D plotting script

## Install

Use in editable mode for development:

```bash
pip install -e .
```

Or install from source (after packaging):

```bash
pip install .
```

## Quick usage

```python
from coord_transform.core import CoordinateTransform
import numpy as np

ct = CoordinateTransform()
# single point
x, y, z = 3.0, 4.0, 5.0
r, theta, zc = ct.cart_to_cyl(x, y, z)
rho, th, phi = ct.cart_to_sph(x, y, z)

# vectorized (N x 3) array
pts = np.array([[3.0,4.0,5.0], [1.0,2.0,2.0]])
cart = pts.T  # shape (3, N)
rs, thetas, zs = ct.cart_to_cyl(cart)
```

## Plot example

Run the demo script:

```bash
python examples/plot_demo.py
```

## Testing

Run tests with pytest:

```bash
pytest -q
```
```
```

---

## pyproject.toml (minimal)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "coord-transform"
version = "0.1.0"
description = "Vectorized Cartesian/Cylindrical/Spherical coordinate transforms"
authors = [ { name = "Your Name", email = "you@example.com" } ]
license = { text = "MIT" }

[project.urls]
"Repository" = "https://github.com/yourusername/coord-transform"
```

---

## src/coord_transform/__init__.py

```python
"""coord_transform package
"""

from .core import CoordinateTransform

__all__ = ["CoordinateTransform"]
```

---

## src/coord_transform/utils.py

```python
import numpy as np


def _as_numpy_array(x):
    """Convert input to numpy array with dtype=float.

    Accepts scalars, lists, tuples, or numpy arrays. Returns a numpy array.
    """
    arr = np.asarray(x, dtype=float)
    return arr
```

---

## src/coord_transform/core.py

```python
"""Core coordinate transformation utilities.

This file implements a class `CoordinateTransform` that supports scalar and
vectorized transforms between Cartesian, Cylindrical and Spherical systems.

Conventions used:
- Cartesian: (x, y, z)
- Cylindrical: (r, theta, z) where theta = arctan2(y, x), r >= 0
- Spherical: (rho, theta, phi) where rho >= 0, theta = arctan2(y, x), phi = angle from +z (colatitude), so phi in [0, pi]

All methods accept either:
- three scalars (x, y, z) and return three scalars, or
- a single array-like of shape (3, N) or (N, 3) to operate on many points at once

The outputs follow the same shape rules.
"""

from typing import Tuple
import numpy as np
from .utils import _as_numpy_array


class CoordinateTransform:
    """Coordinate transformation helper.

    Methods are provided in both directions. Inputs can be scalars or
    arrays; outputs keep the vectorized shape.
    """

    def _normalize_input(self, a, b=None, c=None):
        """Handle different input shapes and return array shaped (3, N).

        If `b` and `c` are None and `a` is array-like, we accept shapes
        (3, N) or (N, 3) or (3,) or (N,) (inpaired)
        """
        if b is None and c is None:
            arr = _as_numpy_array(a)
            if arr.ndim == 1 and arr.shape[0] == 3:
                return arr.reshape(3, 1)
            if arr.ndim == 2:
                # either (3, N) or (N, 3)
                if arr.shape[0] == 3:
                    return arr
                if arr.shape[1] == 3:
                    return arr.T
            raise ValueError("Single array input must have shape (3,) or (3,N) or (N,3)")
        else:
            x = _as_numpy_array(a)
            y = _as_numpy_array(b)
            z = _as_numpy_array(c)
            # broadcast to same shape
            x, y, z = np.broadcast_arrays(x, y, z)
            # flatten to (N,) then stack
            return np.vstack((x.ravel(), y.ravel(), z.ravel()))

    def _maybe_unwrap(self, arr3):
        """If input was scalar (N==1) return scalars, else return arrays.
        arr3 is shape (3, N)
        """
        if arr3.shape[1] == 1:
            return arr3[0, 0], arr3[1, 0], arr3[2, 0]
        return arr3

    # ---------- Cartesian -> Cylindrical ----------
    def cart_to_cyl(self, x, y=None, z=None) -> Tuple:
        """Convert Cartesian (x,y,z) to Cylindrical (r, theta, z).

        theta is in radians, from -pi..pi via arctan2(y, x).
        Works with scalars or vectorized arrays.
        """
        A = self._normalize_input(x, y, z)
        xs, ys, zs = A[0, :], A[1, :], A[2, :]
        r = np.hypot(xs, ys)
        theta = np.arctan2(ys, xs)
        out = np.vstack((r, theta, zs))
        return self._maybe_unwrap(out)

    # ---------- Cartesian -> Spherical ----------
    def cart_to_sph(self, x, y=None, z=None) -> Tuple:
        """Convert Cartesian (x,y,z) to Spherical (rho, theta, phi).

        phi is colatitude: angle from +z axis. If rho == 0 then phi = 0.
        """
        A = self._normalize_input(x, y, z)
        xs, ys, zs = A[0, :], A[1, :], A[2, :]
        rho = np.sqrt(xs**2 + ys**2 + zs**2)
        theta = np.arctan2(ys, xs)
        # avoid division by zero
        phi = np.zeros_like(rho)
        nonzero = rho > 0
        phi[nonzero] = np.arccos(zs[nonzero] / rho[nonzero])
        out = np.vstack((rho, theta, phi))
        return self._maybe_unwrap(out)

    # ---------- Cylindrical -> Cartesian ----------
    def cyl_to_cart(self, r, theta=None, z=None) -> Tuple:
        """Convert Cylindrical (r, theta, z) to Cartesian (x, y, z).

        Accepts vectorized inputs.
        """
        A = self._normalize_input(r, theta, z)
        rs, thetas, zs = A[0, :], A[1, :], A[2, :]
        x = rs * np.cos(thetas)
        y = rs * np.sin(thetas)
        out = np.vstack((x, y, zs))
        return self._maybe_unwrap(out)

    # ---------- Spherical -> Cartesian ----------
    def sph_to_cart(self, rho, theta=None, phi=None) -> Tuple:
        """Convert Spherical (rho, theta, phi) to Cartesian (x, y, z).

        phi is colatitude (angle from +z). Works vectorized.
        """
        A = self._normalize_input(rho, theta, phi)
        rhos, thetas, phis = A[0, :], A[1, :], A[2, :]
        sin_phi = np.sin(phis)
        x = rhos * sin_phi * np.cos(thetas)
        y = rhos * sin_phi * np.sin(thetas)
        z = rhos * np.cos(phis)
        out = np.vstack((x, y, z))
        return self._maybe_unwrap(out)

    # ---------- Cylindrical -> Spherical ----------
    def cyl_to_sph(self, r, theta=None, z=None) -> Tuple:
        # Convert Cyl -> Cart -> Sph
        x, y, zc = self.cyl_to_cart(r, theta, z)
        return self.cart_to_sph(x, y, zc)

    # ---------- Spherical -> Cylindrical ----------
    def sph_to_cyl(self, rho, theta=None, phi=None) -> Tuple:
        x, y, z = self.sph_to_cart(rho, theta, phi)
        return self.cart_to_cyl(x, y, z)
```

---

## examples/plot_demo.py

```python
"""Simple demo showing 3D scatter and transformations.

Run: python examples/plot_demo.py
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from coord_transform.core import CoordinateTransform

ct = CoordinateTransform()

# Some sample Cartesian points
points = np.array([
    [3.0, 4.0, 5.0],
    [1.0, 2.0, 2.0],
    [0.0, 1.0, 0.0],
    [-2.0, -2.0, 1.0]
])

# shape (3, N)
cart = points.T
r, thetas, zs = ct.cart_to_cyl(cart)
rho, thetas_s, phis = ct.cart_to_sph(cart)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Original Cartesian
ax1.scatter(cart[0, :], cart[1, :], cart[2, :])
ax1.set_title('Cartesian')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

# Reconstructed from spherical -> cart (sanity check)
recon = np.vstack(ct.sph_to_cart(rho, thetas_s, phis)) if isinstance(rho, np.ndarray) else np.array(ct.sph_to_cart(rho, thetas_s, phis))

ax2.scatter(recon[0, :], recon[1, :], recon[2, :])
ax2.set_title('Reconstructed from Spherical')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

plt.tight_layout()
plt.show()
```

---

## tests/test_core.py

```python
import numpy as np
from coord_transform.core import CoordinateTransform

ct = CoordinateTransform()


def test_cart_cyl_roundtrip_scalar():
    x, y, z = 3.0, 4.0, 5.0
    r, th, zc = ct.cart_to_cyl(x, y, z)
    x2, y2, z2 = ct.cyl_to_cart(r, th, zc)
    assert np.allclose([x, y, z], [x2, y2, z2])


def test_cart_sph_roundtrip_vector():
    pts = np.array([[3.0, 4.0, 5.0], [1.0, 2.0, 2.0]])
    cart = pts.T
    rho, th, ph = ct.cart_to_sph(cart)
    recon = ct.sph_to_cart(rho, th, ph)
    # recon is (3,N)
    recon_pts = np.array(recon).T
    assert np.allclose(pts, recon_pts)
```

---

## .github/workflows/ci.yml

```yaml
name: Python package

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          pip install pytest
      - name: Run tests
        run: |
          pytest -q
```

---

## License
MIT recommended. Put your name and year in the LICENSE file if you want.

---

# Next steps

If you'd like, I can:
- generate a zip file of the full project here,
- push it to a GitHub repo (I can provide the exact files you need to paste/upload), or
- run through any part and customize (e.g. change spherical convention to use elevation instead of colatitude, or add velocity transforms for 6D).

Tell me which of these you want and I’ll prepare it.