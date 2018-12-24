mutable struct AMGPreconditioner{T, TML<:AMG.MultiLevel}
    ml::TML
end
struct RugeStuben end
struct SmoothedAggregation end

for (t, f) in [(:RugeStuben, :ruge_stuben), (:SmoothedAggregation, :smoothed_aggregation)]
    @eval begin
        function AMGPreconditioner(::Type{$t}, A::AbstractMatrix)
            _A = get_data(A)
            ml = $f(_A)
            return AMGPreconditioner{$t, typeof(ml)}(ml)
        end

        function UpdatePreconditioner!(C::AMGPreconditioner{$t}, A::AbstractMatrix)
            _A = get_data(A)
            C.ml = $f(_A)
            return C
        end
    end
end

AMGPreconditioner(A::AbstractMatrix) = AMGPreconditioner(RugeStuben, A)
AMGPreconditioner{T}(A::AbstractMatrix) where T = AMGPreconditioner(T, A)

@inline function \(p::AMGPreconditioner, b)
    x = copy(b); 
    return ldiv!(x, AMG.Preconditioner(p.ml), b)
end
@inline *(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml) * b
@inline function ldiv!(x, p::AMGPreconditioner, b)
    x .= b
    return ldiv!(x, AMG.Preconditioner(p.ml), b)
end
@inline mul!(b, p::AMGPreconditioner, x) = mul!(b, AMG.Preconditioner(p.ml), x)
