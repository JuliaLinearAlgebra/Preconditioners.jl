module Preconditioners

using AlgebraicMultigrid, Compat
const AMG = AlgebraicMultigrid

using LinearAlgebra, SparseArrays
import LinearAlgebra: ldiv!, \, *, mul!

abstract type AbstractPreconditioner end

include("incomplete_selected_inversion.jl")
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
