"""Minimal test to trace the sign error in the representation formula.

Half-space: rigid wall at y=0, source at (0, 0.5).
Expected: p_total = G_0(direct) + G_0(image)
Representation formula: p_total(x) = p_inc(x) + integral
integral should = G_0(x, image)

We'll print complex values at every step.
"""
import numpy as np
from scipy.special import hankel1

k = 36.64
SOURCE = np.array([0.0, 0.5])
IMAGE = np.array([0.0, -0.5])
FIELD = np.array([0.3, 0.3])

# Exact values
def G0(p1, p2):
    r = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return -0.25j * hankel1(0, k * max(r, 1e-15))

p_inc = G0(FIELD, SOURCE)
p_image = G0(FIELD, IMAGE)
p_exact = p_inc + p_image

print("EXACT SOLUTION:")
print(f"  p_inc   = {p_inc:.8f},  |p_inc|   = {abs(p_inc):.6f}")
print(f"  p_image = {p_image:.8f},  |p_image| = {abs(p_image):.6f}")
print(f"  p_exact = {p_exact:.8f},  |p_exact| = {abs(p_exact):.6f}")
print(f"  p_scat  = p_image = {p_image:.8f}")

# Representation formula: p_total = p_inc + integral
# integral = int_(-inf)^(+inf) dG0/dn_y(FIELD, (s,0)) * p_surf(s) ds
# where n_body = (0, 1), p_surf = 2*G0((s,0), SOURCE)

# First, verify the KERNEL at one specific boundary point
s_test = 1.0
y_test = np.array([s_test, 0.0])
n_body = np.array([0.0, 1.0])

# Kernel by formula: dG0/dn_y = nabla_y G0 . n_body
# nabla_y G0(x,y) = (ik/4) H_1(kr) (y-x)/r
dx = FIELD - y_test  # x - y
r = np.sqrt(dx @ dx)
kr = k * r

# Method 1: Direct formula  dG0/dn = nabla_y G0 . n
# nabla_y G_0 = (ik/4) H_1(kr) (y-x)/r
grad_y = 0.25j * k * hankel1(1, kr) * (y_test - FIELD) / r
kernel_method1 = grad_y @ n_body

# Method 2: Code formula  -(ik/4) H_1(kr) (x-y).n / r
kernel_method2 = -0.25j * k * hankel1(1, kr) * (dx @ n_body) / r

# Method 3: Finite difference  [G0(x, y+eps*n) - G0(x, y-eps*n)] / (2*eps)
eps = 1e-6
kernel_method3 = (G0(FIELD, y_test + eps * n_body) - G0(FIELD, y_test - eps * n_body)) / (2 * eps)

print(f"\nKERNEL at y=({s_test}, 0), normal=(0,1):")
print(f"  Method 1 (nabla_y . n): {kernel_method1:.10f}")
print(f"  Method 2 (code formula): {kernel_method2:.10f}")
print(f"  Method 3 (finite diff):  {kernel_method3:.10f}")
print(f"  Match 1-2: {abs(kernel_method1 - kernel_method2) < 1e-10}")
print(f"  Match 1-3: {abs(kernel_method1 - kernel_method3) / abs(kernel_method1):.2e}")

# Now compute the integral using method 1 (which we've verified)
print("\nCOMPUTING INTEGRAL over full x-axis [-L, L]:")
L = 10.0
N = 50000
s_arr = np.linspace(-L, L, N + 1)
s_mid = 0.5 * (s_arr[:-1] + s_arr[1:])
ds = s_arr[1] - s_arr[0]

integral = 0.0 + 0.0j
for i in range(N):
    y = np.array([s_mid[i], 0.0])
    p_surf = 2.0 * G0(y, SOURCE)

    dx_i = FIELD - y
    r_i = np.sqrt(dx_i @ dx_i)
    if r_i < 1e-12:
        continue
    grad_y_i = 0.25j * k * hankel1(1, k * r_i) * (y - FIELD) / r_i
    kernel_i = grad_y_i @ n_body

    integral += kernel_i * p_surf * ds

print(f"  integral     = {integral:.8f},  |integral| = {abs(integral):.6f}")
print(f"  expected     = {p_image:.8f},  |expected| = {abs(p_image):.6f}")
print(f"  -expected    = {-p_image:.8f}")
print(f"  |int - expected| / |expected| = {abs(integral - p_image)/abs(p_image):.4f}")
print(f"  |int + expected| / |expected| = {abs(integral + p_image)/abs(p_image):.4f}")
print()
print(f"  p_repr = p_inc + integral = {p_inc + integral:.8f}")
print(f"  p_exact                   = {p_exact:.8f}")
print(f"  error = {abs(p_inc + integral - p_exact)/abs(p_exact):.4f}")
