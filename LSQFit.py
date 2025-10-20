import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from math import log
from random import gauss

xmin = 1.0
xmax = 20.0
npoints = 12
sigma = 0.2
lx = np.zeros(npoints)
ly = np.zeros(npoints)
ley = np.zeros(npoints)
pars = [0.5, 1.3, 0.5]

def f(x, par):
    return par[0] + par[1] * log(x) + par[2] * log(x) * log(x)

def getX(x):  # x = array-like
    step = (xmax - xmin) / npoints
    for i in range(npoints):
        x[i] = xmin + i * step

def getY(x, y, ey):  # x,y,ey = array-like
    for i in range(npoints):
        y[i] = f(x[i], pars) + gauss(0, sigma)
        ey[i] = sigma

# get a random sampling of the (x,y) data points, rerun to generate different data sets for the plot below
getX(lx)
getY(lx, ly, ley)

fig, ax = plt.subplots()
ax.errorbar(lx, ly, yerr=ley, fmt='o')
ax.set_title("Pseudoexperiment")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()
fig.savefig("pseudoexperiment_py.png")  # *** save figure instead of showing it ***
#fig.show()  # *** disabled for headless environment ***

# *** modify and add your code here ***
nexperiments = 1000  # for example

# initialize arrays to store fit results and chi^2
par_a = np.zeros(nexperiments)
par_b = np.zeros(nexperiments)
par_c = np.zeros(nexperiments)
chi2_reduced = np.zeros(nexperiments)

# perform many least squares fits on different pseudo experiments here
# fill histograms w/ required data
for iexp in range(nexperiments):
    getY(lx, ly, ley)

    # build design matrix A (npoints x 3)
    A = np.zeros((npoints, 3))
    for i in range(npoints):
        lxi = log(lx[i])
        A[i, 0] = 1.0
        A[i, 1] = lxi
        A[i, 2] = lxi**2

    W = np.diag(1.0 / ley**2)

    # least squares solution: (A^T W A)^-1 A^T W y
    AtW = A.T @ W
    AtWA = AtW @ A
    AtWy = AtW @ ly
    coeffs = inv(AtWA) @ AtWy

    par_a[iexp] = coeffs[0]
    par_b[iexp] = coeffs[1]
    par_c[iexp] = coeffs[2]

    # compute chi2 and reduced chi2
    y_fit = A @ coeffs
    residuals = ly - y_fit
    chi2 = np.sum((residuals / ley) ** 2)
    dof = npoints - 3
    chi2_reduced[iexp] = chi2 / dof

# plot results of the pseudoexperiments
fig2, axs = plt.subplots(2, 2, figsize=(10, 8))
plt.tight_layout(pad=3.0)

# careful, the automated binning may not be optimal for displaying your results!
axs[0, 0].hist2d(par_a, par_b, bins=50)
axs[0, 0].set_title('Parameter b vs a')
axs[0, 0].set_xlabel('a')
axs[0, 0].set_ylabel('b')

axs[0, 1].hist2d(par_a, par_c, bins=50)
axs[0, 1].set_title('Parameter c vs a')
axs[0, 1].set_xlabel('a')
axs[0, 1].set_ylabel('c')

axs[1, 0].hist2d(par_b, par_c, bins=50)
axs[1, 0].set_title('Parameter c vs b')
axs[1, 0].set_xlabel('b')
axs[1, 0].set_ylabel('c')

axs[1, 1].hist(chi2_reduced, bins=50)
axs[1, 1].set_title('Reduce chi^2 distribution')
axs[1, 1].set_xlabel('chi^2 / dof')

fig2.savefig("fit_results_py.png")  # *** save histogram plot instead of showing it ***
print("Plots saved: pseudoexperiment_py.png, fit_results_py.png")  # *** notify user of output ***

# **************************************

input("hit Enter to exit")

