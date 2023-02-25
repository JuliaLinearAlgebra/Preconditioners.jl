# Preconditioners

[![Build Status](https://travis-ci.org/JuliaLinearAlgebra/Preconditioners.jl.svg?branch=master)](https://travis-ci.org/JuliaLinearAlgebra/Preconditioners.jl) [![codecov.io](http://codecov.io/github/JuliaLinearAlgebra/Preconditioners.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaLinearAlgebra/Preconditioners.jl?branch=master)

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

## Citing Preconditioners.jl

If you use Preconditioners for your own research, please consider citing the following publication: Mohamed Tarek. Preconditioners.jl: A Flexible and Extensible Framework for Preconditioning in Iterative Solvers. 2023. doi: 10.13140/RG.2.2.26655.02721.
```
@article{MohamedTarekPreconditionersjl,
  doi = {10.13140/RG.2.2.26655.02721},
  url = {https://rgdoi.net/10.13140/RG.2.2.26655.02721},
  author = {Tarek,  Mohamed},
  language = {en},
  title = {Preconditioners.jl: A Flexible and Extensible Framework for Preconditioning in Iterative Solvers},
  year = {2023}
}
```
