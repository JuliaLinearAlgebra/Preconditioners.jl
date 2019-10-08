function assert_pd(d, α)
    @assert α == 0 && all(x -> x > 0, d) "The input matrix is not positive definite."
end

function update_L!(L, d)
    @inbounds for j in 1:size(L, 2)
        _d = sqrt(d[j])
        L[j,j] = _d
        for i in Base.Iterators.drop(nzrange(L, j), 1)
            L.nzval[i] *= _d
        end
    end
    return L
end

mutable struct CholeskyPreconditioner{T, S <: AbstractSparseMatrix{T}} <: AbstractPreconditioner
    L::LowerTriangular{T, S}
    memory::Int
end
function EmptyCholeskyPreconditioner(A, memory=1)
    T = eltype(A)
    _A = A isa Symmetric || A isa Hermitian ? A.data : A
    return CholeskyPreconditioner(LowerTriangular(sparse(one(T)*I, size(_A)...)), memory)
end

function CholeskyPreconditioner(A, memory=2)
    _A = get_data(A)
    L, d, α = lldl(_A, memory=memory)
    assert_pd(d, α)
    update_L!(L, d)
    return CholeskyPreconditioner(LowerTriangular(L), memory)
end

function UpdatePreconditioner!(C::CholeskyPreconditioner, A, memory=C.memory)
    _A = get_data(A)
    L, d, α = lldl(_A, memory=memory)
    assert_pd(d, α)
    update_L!(L, d)
    C.L = LowerTriangular(L)
    return C
end

@inline function ldiv!(C::CholeskyPreconditioner{T}, y::AbstractVector{T}) where {T}
    ldiv!(C.L, y)
    return ldiv!(C.L', y)
end
@inline function ldiv!(y::AbstractVector{T}, C::CholeskyPreconditioner{T}, b::AbstractVector{T}) where {T}
    y .= b
    return ldiv!(C, y)
end
@inline function (\)(C::CholeskyPreconditioner{T, <:AbstractSparseMatrix{T}}, b::AbstractVector{T}) where {T}
    y = copy(b)
    return ldiv!(C, y)
end
