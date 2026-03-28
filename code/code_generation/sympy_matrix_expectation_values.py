import sympy

from sympy import *

init_printing()

n = 3
M = MatrixSymbol("A", n, n)  # Create an n times n matrix
Msym = Matrix(n, n, lambda i, j: M[min(i, j), max(i, j)])  # Symmetrize said n times n matrix

r = Matrix(symarray("r", (n)))  # position vector
mu = Matrix(symarray("mu", (n)))  # shift vector
# r=symarray(r,(n))
# mu=symarray(r,(n))
expression = exp(-(r - mu).T @ Msym @ (r - mu))  # The general shape of a single Gaussian
inv_expression = exp(((r - mu).T @ Msym @ ((r - mu))))
nablaSquareds = []
for i in range(n):
    secondDeriv_i = expression.diff(r[i]).diff(r[i])
    nablaSquareds.append(secondDeriv_i)
expression_0 = nablaSquareds[0]
kinetic_energy = -1 / 2 * expression_0
for i in range(1, n):
    kinetic_energy += -1 / 2 * nablaSquareds[i]
print("Kinetic energy")
print(kinetic_energy[0])
print("Kinetic energy (simplified)")
expectation_values_needed = ((kinetic_energy * inv_expression)[0]).simplify()
pprint(expectation_values_needed)
collected = expectation_values_needed.series(r[0], 0, 5).removeO()
for i in range(1, n):
    collected = collected.series(r[i], 0, 5).removeO()
# for i in range(n):
#    collected=collected.series(r[i],0,5).collect(r[i])
# terms=(r[0],r[0]*r[1],r[1],r[0]*r[2],r[2],r[1]*r[2])
# collected=collect(collected,terms)
print("Collected")
pprint(collected)
MsymSquared = Msym * Msym

expr2 = 0
for k in range(n):
    expr2 += Msym[k, k]  # Trace term
expr2 += (4 * mu.T @ MsymSquared @ r - 2 * r.T @ MsymSquared @ r - 2 * mu.T @ MsymSquared @ mu)[0]
print("Removing terms term by term")
pprint((collected - expr2).simplify())
