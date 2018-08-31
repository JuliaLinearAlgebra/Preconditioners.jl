mutable struct DiagonalPreconditioner{T, S<:AbstractVector{T}} <: AbstractPreconditioner
    D::S
end
function EmptyDiagonalPreconditioner(A::AbstractMatrix)
    D = zeros(eltype(A), size(A,1))
    return DiagonalPreconditioner{eltype(D), typeof(D)}(D)
end
function DiagonalPreconditioner(A::AbstractMatrix)
    D = diag(A, 0)
    return DiagonalPreconditioner{eltype(D), typeof(D)}(D)
end
function diag!(D, A::AbstractMatrix)
    length(D) == size(A,1) == size(A,2) || throw("D and A sizes are not compatible.")
    @inbounds @simd for i in 1:length(D)
        D[i] = A[i,i]
    end
    return
end
function UpdatePreconditioner!(D::DiagonalPreconditioner, K::AbstractMatrix)
    diag!(D.D, K)
    return D
end
function ldiv!(y, C::DiagonalPreconditioner, b)
    @inbounds @simd for i in 1:length(C.D)
        y[i,:] .= view(b, i, :) ./ C.D[i]
    end
    return y
end
function (\)(C::DiagonalPreconditioner, b)
    y = zeros(b)
    @inbounds @simd for i in 1:length(C.D)
        y[i,:] .= view(b, i, :) / C.D[i]
    end
    return y
end

