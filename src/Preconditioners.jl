module Preconditioners

using IncompleteSelectedInversion

import Base.LinAlg: A_ldiv_B!, \

abstract type AbstractPreconditioner end

include("incompletecholesky.jl")
include("diagonal.jl")

export  CholeskyPreconditioner, 
        EmptyCholeskyPreconditioner, 
        DiagonalPreconditioner, 
        EmptyDiagonalPreconditioner,
        UpdatePreconditioner!

end # module
