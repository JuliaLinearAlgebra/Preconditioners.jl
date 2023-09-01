mutable struct AMGPreconditioner{T, TML<:AMG.MultiLevel, C<:AMG.Cycle}
    ml::TML
    cycle::C
end
struct RugeStuben end
struct SmoothedAggregation end

for (t, f) in [(:RugeStuben, :ruge_stuben), (:SmoothedAggregation, :smoothed_aggregation)]
    @eval begin
        function AMGPreconditioner(::Type{$t}, A::AbstractMatrix; cycle = AMG.V())
            _A = get_data(A)
            ml = $f(_A)
            return AMGPreconditioner{$t, typeof(ml), typeof(cycle)}(ml, cycle)
        end

        function UpdatePreconditioner!(C::AMGPreconditioner{$t}, A::AbstractMatrix)
            _A = get_data(A)
            C.ml = $f(_A)
            return C
        end
    end
end

AMGPreconditioner(A::AbstractMatrix; kwargs...) = AMGPreconditioner(RugeStuben, A; kwargs...)
AMGPreconditioner{T}(A::AbstractMatrix; kwargs...) where T = AMGPreconditioner(T, A; kwargs...)

@inline function \(p::AMGPreconditioner, b)
    x = copy(b)
    return ldiv!(x, AMG.Preconditioner(p.ml, p.cycle), b)
end
@inline *(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml, p.cycle) * b
@inline ldiv!(p::AMGPreconditioner, b) = b .= p \ b
@inline function ldiv!(x::AbstractVector, p::AMGPreconditioner, b::AbstractVector)
    x .= b
    return ldiv!(x, AMG.Preconditioner(p.ml, p.cycle), b)
end
@inline function ldiv!(x::AbstractMatrix, p::AMGPreconditioner, b::AbstractMatrix)
    foreach(zip(eachcol(x), eachcol(b))) do (_x, _b)
        ldiv!(_x, p, _b)
    end
    return x
end
@inline mul!(b, p::AMGPreconditioner, x) = mul!(b, AMG.Preconditioner(p.ml, p.cycle), x)
