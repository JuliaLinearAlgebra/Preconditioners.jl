module Preconditioners

using AlgebraicMultigrid, Compat, LimitedLDLFactorizations
const AMG = AlgebraicMultigrid
const LLDL = LimitedLDLFactorizations

using LinearAlgebra, SparseArrays
import LinearAlgebra: ldiv!, \, *, mul!

abstract type AbstractPreconditioner end

include("incompletecholesky.jl")
include("diagonal.jl")
include("amg.jl")

export  CholeskyPreconditioner, 
        EmptyCholeskyPreconditioner, 
        DiagonalPreconditioner, 
        EmptyDiagonalPreconditioner,
        AMGPreconditioner,
        SmoothedAggregation,
        RugeStuben,
        UpdatePreconditioner!

end # module
