# Preconditioners

[![Build Status](https://travis-ci.org/mohamed82008/Preconditioners.jl.svg?branch=master)](https://travis-ci.org/mohamed82008/Preconditioners.jl) [![codecov.io](http://codecov.io/github/mohamed82008/Preconditioners.jl/coverage.svg?branch=master)](http://codecov.io/github/mohamed82008/Preconditioners.jl?branch=master)

## Examples

```julia

A = sprand(1000, 1000, 0.01)
A = A + A' + 30I

# Diagonal preconditioner
p = DiagonalPreconditioner(A)

# Incomplete Cholesky preconditioner with cut-off level 2
p = CholeskyPreconditioner(A, 2)

# Algebraic multigrid preconditioner (AMG)
# Ruge-Stuben variant
p = AMGPreconditioner{RugeStuben}(A)
# Smoothed aggregation
p = AMGPreconditioner{SmoothedAggregation}(A)

# Solve the system of equations
b = A*ones(1000)
x = cg(A, b, Pl=p)

A = sprand(1000, 1000, 0.01)
A = A + A' + 30I
# Updates the preconditioner with the new matrix A
UpdatePreconditioner!(p, A)

```


## Advanced AMG preconditioners

More advanced AMG preconditioners are also possible by building the `MultiLevel` struct that `AMGPreconditioner` wraps yourself using the package [AMG.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl).
