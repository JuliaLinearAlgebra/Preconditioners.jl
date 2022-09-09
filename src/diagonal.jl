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
    if !(axes(D, 1) == axes(A, 1) == axes(A, 2))
        throw(ArgumentError("incompatible indices for input arguments"))
    end
    @inbounds for (i, j, k) in zip(eachindex(D), axes(A, 1), axes(A, 2))
        D[i] = A[j, k]
    end
    return
end
function UpdatePreconditioner!(D::DiagonalPreconditioner, K::AbstractMatrix)
    diag!(D.D, K)
    return D
end
ldiv!(C::DiagonalPreconditioner, b) = ldiv!(b, C, b)
function ldiv!(y, C::DiagonalPreconditioner, b)
    if !(axes(y, 1) == axes(b, 1) == axes(C.D, 1) && axes(y, 2) == axes(b, 2))
        throw(ArgumentError("incompatible indices for input arguments"))
    end
    @inbounds for (yj, bj) in zip(axes(y, 2), axes(b, 2))
        for (yi, bi, Di) in zip(axes(y, 1), axes(b, 1), eachindex(C.D))
            y[yi, yj] = b[bi, bj] / C.D[Di]
        end
    end
    return y
end
function (\)(C::DiagonalPreconditioner, b)
    return ldiv!(similar(b), C, b)
end
