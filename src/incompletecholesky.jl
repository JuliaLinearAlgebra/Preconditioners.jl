function assert_pd(d, α)
    @assert α == 0 && all(x -> x > 0, d) "The input matrix is not positive definite."
end

mutable struct CholeskyPreconditioner{L} <: AbstractPreconditioner
    ldlt::L
    memory::Int
    function CholeskyPreconditioner(A, memory=2)
        _A = get_data(A)
        LLDL = lldl(_A, memory=memory)
        assert_pd(LLDL.D, LLDL.α)
        return new{typeof(LLDL)}(LLDL, memory)
    end
end

function UpdatePreconditioner!(C::CholeskyPreconditioner, A, memory=C.memory)
    _A = get_data(A)
    LLDL = lldl(_A, memory=memory)
    assert_pd(LLDL.D, LLDL.α)
    C.ldlt = LLDL
    return C
end

@inline function ldiv!(C::CholeskyPreconditioner, y::AbstractVector)
    ldiv!(y, C.ldlt, copy(y))
    return y
end
@inline function ldiv!(y::AbstractVector, C::CholeskyPreconditioner, b::AbstractVector)
    return ldiv!(y, C.ldlt, b)
end
@inline function (\)(C::CholeskyPreconditioner, b::AbstractVector)
    return ldiv!(copy(b), C.ldlt, b)
end
