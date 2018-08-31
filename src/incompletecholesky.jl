mutable struct CholeskyPreconditioner{T, S <: AbstractSparseMatrix{T}} <: AbstractPreconditioner
    L::LowerTriangular{T, S}
    c::Int
end
function EmptyCholeskyPreconditioner(A, c=1)
    if A isa Symmetric
        return CholeskyPreconditioner{eltype(A), SparseMatrixCSC{eltype(A),Int}}(LowerTriangular(speye(A.data)), c)
    else
        return CholeskyPreconditioner{eltype(A), SparseMatrixCSC{eltype(A),Int}}(LowerTriangular(speye(A)), c)    
    end
end
function CholeskyPreconditioner(A, c=2)
    if A isa Symmetric
        L = cldlt(A.data,c)
    else
        L = cldlt(A,c)
    end
    @inbounds for j in 1:size(L,2)
        d = (L[j,j])^(eltype(A)(1)/2)
        L[j,j] = d
        for i in Base.Iterators.drop(nzrange(L,j), 1)
            L.nzval[i] *= d
        end
    end
    return CholeskyPreconditioner{eltype(L), typeof(L)}(LowerTriangular(L),c)
end
function UpdatePreconditioner!(C::CholeskyPreconditioner, A, c=C.c)
    L = cldlt(A,c)
    @inbounds for j in 1:size(L, 2)
        d = sqrt(L[j,j])
        L[j,j] = d
        for i in Base.Iterators.drop(nzrange(L,j), 1)
            L.nzval[i] *= d
        end
    end
    C.L = LowerTriangular(L)
    C
end
function ldiv!(y::AbstractVector{T}, C::CholeskyPreconditioner{T, S}, b::AbstractVector{T}) where {T, S}
    y .= b
    ldiv!(C.L', y)
    ldiv!(C.L, y)
    return y
end
function (\)(C::CholeskyPreconditioner{T, S}, b::AbstractVector{T}) where {T, S <: AbstractSparseMatrix{T}}
    y = copy(b)
    ldiv!(C.L', y)
    ldiv!(C.L, y)
    return y
end
