mutable struct DiagonalPreconditioner{T, S<:AbstractVector{T}} <: AbstractPreconditioner
    D::S
end
function EmptyDiagonalPreconditioner(A::AbstractMatrix)
    D = zeros(eltype(A), size(A,1))
    return DiagonalPreconditioner{eltype(D), typeof(D)}(D)
end
function DiagonalPreconditioner(A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(ArgumentError("matrix must be square"))
    D = diag(A, 0)
    # Since diag(::SparseMatrixCSC) return a SparseVector we convert it to a Vector since
    # indexing is much faster for that case, and the diagonal should be dense anyway.
    # The type check is added here instead of having dispatches on SparseMatrixCSC since
    # that would not catch e.g. Hermitian{SparseMatrixCSC} and other variants.
    if D isa SparseVector
        if nnz(D) == size(A, 1)
            # Fast path in case all values of the diagonal are stored (this should always be
            # true since otherwise we will divide by 0 later)
            D = D.nzval
        else
            # Fallback (TODO: Better to just error?)
            D = convert(Vector, D)
        end
    end
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
@inline ldiv!(C::DiagonalPreconditioner, b) = ldiv!(b, C, b)
@inline function ldiv!(y, C::DiagonalPreconditioner, b)
    @inbounds @simd for j ∈ 1:size(y, 2)
        for i ∈ 1:length(C.D)
            y[i,j] = b[i,j] / C.D[i]
        end
    end
    return y
end
@inline function (\)(C::DiagonalPreconditioner, b)
    y = zero(b)
    @inbounds @simd for j ∈ 1:size(y, 2)
        for i ∈ 1:length(C.D)
            y[i,j] = b[i,j] / C.D[i]
        end
    end
    return y
end
