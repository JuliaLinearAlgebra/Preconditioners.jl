#=
# [Preconditioners overview](@id 1-overview)

This page illustrates some of the method(s) in the Julia package
[`Preconditioners.jl`](https://github.com/JuliaLinearAlgebra/Preconditioners.jl).

This page was generated from a single Julia file:
[1-overview.jl](@__REPO_ROOT_URL__/1-overview.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`1-overview.ipynb`](@__NBVIEWER_ROOT_URL__/1-overview.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`1-overview.ipynb`](@__BINDER_ROOT_URL__/1-overview.ipynb).


# ### Setup

# Packages needed here.

using Preconditioners: DiagonalPreconditioner, CholeskyPreconditioner
using InteractiveUtils: versioninfo
using SparseArrays: sprand
using LinearAlgebra: I, cond


#=
## Overview

Preconditioning is useful
for accelerating solutions
to systems of equations
and optimization problems.


## Examples
=#

n = 1000
A = sprand(n, n, 0.01)
A = A + A' + 10I

# Examine conditioning:
cA = cond(Matrix(A), 2)

# Diagonal preconditioner
Dp = DiagonalPreconditioner(A)

# Apply preconditioner
DA = Dp \ A

# Examine effect on condition number
cd = cond(Matrix(DA), 2)

# Incomplete Cholesky preconditioner with cut-off level 2
Pc = CholeskyPreconditioner(A, 2)

# Here is a quick way to handle a matrix argument.
# This could be done more efficiently eventually; see Issue #33.
import Base.\
\(C::CholeskyPreconditioner, A::AbstractMatrix) = reduce(hcat, [C \ c for c in eachcol(A)])

# apply preconditioner
CA = Pc \ A

# examine effect on condition number
cc = cond(Matrix(CA), 2)


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
