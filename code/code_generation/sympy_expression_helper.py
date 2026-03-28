from sympy import *

cl = Symbol("ci_c")
cl_c = cl  # conjugate(cl)
cj = Symbol("cj")
rho = IndexedBase("rho_T_conj")
B = IndexedBase("B")
omega = IndexedBase("omega")
r = IndexedBase("r")
A = IndexedBase("A")
K = IndexedBase("IabLj")
mu = IndexedBase("\mu")
Kmu = IndexedBase("IabLjmuj")
N = Symbol("N")
mukMu = Symbol("(mKm)")
i, j, k, l, m, n, o, p, q = symbols("i,j,k,l,m,n,o,p,q", integer=True)

Pi = cl_c + 4 * Sum((rho[i]) * r[i], (i, 1, N)) - 2 * Sum(Sum(r[i] * r[j] * (B[i, j]) ** 2, (i, 1, N)), (j, 1, N))
Pj = cj + 4 * Sum(omega[k] * r[k], (k, 1, N)) - 2 * Sum(Sum(r[k] * r[l] * A[k, l] ** 2, (k, 1, N)), (l, 1, N))
deriv_term_LJ = -Sum(Sum(K[n, m] * r[n] * r[m], (n, 1, N)), (m, 1, N)) + 2 * Sum(r[m] * Kmu[m], (m, 1, N)) - mukMu
# print(latex(expand(Pj*Pi*deriv_term_LJ)))

# term=-ck*cl_c*muTKmu + 2*ck*cl_c*Sum(Kmu[m]*r[m]) + ck*cl_c*Sum(K[n, m]*r[m]*r[n]) - 4*ck*muTKmu*Sum(r[i]*rho[i]) + 2*ck*muTKmu*Sum(B[i, j]*r[i]*r[j]) + 8*ck*Sum(Kmu[m]*r[m])*Sum(r[i]*rho[i]) - 4*ck*Sum(Kmu[m]*r[m])*Sum(B[i, j]*r[i]*r[j]) + 4*ck*Sum(r[i]*rho[i])*Sum(K[n, m]*r[m]*r[n]) - 2*ck*Sum(B[i, j]*r[i]*r[j])*Sum(K[n, m]*r[m]*r[n]) - 4*cl_c*muTKmu*Sum(omega[k]*r[k]) + 2*cl_c*muTKmu*Sum(A[k, l]*r[k]*r[l]) + 8*cl_c*Sum(Kmu[m]*r[m])*Sum(omega[k]*r[k]) - 4*cl_c*Sum(Kmu[m]*r[m])*Sum(A[k, l]*r[k]*r[l]) + 4*cl_c*Sum(omega[k]*r[k])*Sum(K[n, m]*r[m]*r[n]) - 2*cl_c*Sum(A[k, l]*r[k]*r[l])*Sum(K[n, m]*r[m]*r[n]) - 16*muTKmu*Sum(omega[k]*r[k])*Sum(r[i]*rho[i]) + 8*muTKmu*Sum(omega[k]*r[k])*Sum(B[i, j]*r[i]*r[j]) + 8*muTKmu*Sum(r[i]*rho[i])*Sum(A[k, l]*r[k]*r[l]) - 4*muTKmu*Sum(A[k, l]*r[k]*r[l])*Sum(B[i, j]*r[i]*r[j]) + 32*Sum(Kmu[m]*r[m])*Sum(omega[k]*r[k])*Sum(r[i]*rho[i]) - 16*Sum(Kmu[m]*r[m])*Sum(omega[k]*r[k])*Sum(B[i, j]*r[i]*r[j]) - 16*Sum(Kmu[m]*r[m])*Sum(r[i]*rho[i])*Sum(A[k, l]*r[k]*r[l]) + 8*Sum(Kmu[m]*r[m])*Sum(A[k, l]*r[k]*r[l])*Sum(B[i, j]*r[i]*r[j]) + 16*Sum(omega[k]*r[k])*Sum(r[i]*rho[i])*Sum(K[n, m]*r[m]*r[n]) - 8*Sum(omega[k]*r[k])*Sum(B[i, j]*r[i]*r[j])*Sum(K[n, m]*r[m]*r[n]) - 8*Sum(r[i]*rho[i])*Sum(A[k, l]*r[k]*r[l])*Sum(K[n, m]*r[m]*r[n]) + 4*Sum(A[k, l]*r[k]*r[l])*Sum(B[i, j]*r[i]*r[j])*Sum(K[n, m]*r[m]*r[n])
cj_deriv = Symbol("cj_deriv")
omega_deriv = IndexedBase("omega_deriv")
Asquared_deriv = IndexedBase("Asquared_deriv")
Pj_deriv = (
    cj_deriv
    + 4 * Sum(omega_deriv[k] * r[k], (k, 1, N))
    - 2 * Sum(Asquared_deriv[k, l] * r[k] * r[l], (k, 1, N), (l, 1, N))
)
pprint(expand(Pi * Pj_deriv))


z = Symbol("z")
y = IndexedBase("y")
deriv_term_mu = 2 * Sum(y[n] * r[n], (n, 1, N)) - 2 * z

lambda_ = Symbol("self.lambda_")

HenonHeilesPotential = (
    1 / 2 * Sum(r[o] * r[o], (o, 1, N))
    - lambda_ / 3 * Sum(r[o + 1] * r[o + 1] * r[o + 1], (o, 1, N - 1))
    + lambda_ * Sum(r[o] * r[o] * r[o + 1], (o, 1, N - 1))
)
zeroth_term = expand(Pi * HenonHeilesPotential * deriv_term_LJ)
first_term = expand(Pj * HenonHeilesPotential * deriv_term_LJ)
second_term = expand(HenonHeilesPotential * Pj_deriv)

# print(second_term)
Pj_deriv = cj_deriv + 4 * Sum(omega_deriv[k] * r[k], (k, 1, N))
zeroth_term = expand(Pi * HenonHeilesPotential * deriv_term_mu)
first_term = expand(Pj * HenonHeilesPotential * deriv_term_mu)
second_term = expand(HenonHeilesPotential * Pj_deriv)
print(second_term)
