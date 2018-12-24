module Preconditioners

using AlgebraicMultigrid, Compat, LimitedLDLFactorizations
const AMG = AlgebraicMultigrid
const LLDL = LimitedLDLFactorizations

using LinearAlgebra, SparseArrays
import LinearAlgebra: ldiv!, \, *, mul!

abstract type AbstractPreconditioner end

function get_data(A)
    if A isa Symmetric || A isa Hermitian
        @warn("Using the data field of the symmetric/Hermitian matrix input.")
        return A.data
    else
        return A
    end
end

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
