mutable struct AMGPreconditioner{S, TML<:AMG.MultiLevel}
    ml::TML
end
struct RugeStuben end
struct SmoothedAggregation end

for (t, f) in [(:RugeStuben, :ruge_stuben), (:SmoothedAggregation, :smoothed_aggregation)]
    @eval begin
        function AMGPreconditioner(::Type{$t}, A::AbstractMatrix)
            ml = $f(A)
            return AMGPreconditioner{$t, typeof(ml)}(ml)
        end

        function UpdatePreconditioner!(C::AMGPreconditioner{$t}, A)
            C.ml = $f(A)
            return C
        end
    end
end
AMGPreconditioner(A::AbstractMatrix) = AMGPreconditioner(RugeStuben, A)
AMGPreconditioner{T}(A::AbstractMatrix) where T = AMGPreconditioner(T, A)

\(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml) \ b
*(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml) * b
A_ldiv_B!(x, p::AMGPreconditioner, b) = A_ldiv_B!(x, AMG.Preconditioner(p.ml), b)
A_mul_B!(b, p::AMGPreconditioner, x) = A_mul_B!(b, AMG.Preconditioner(p.ml), x)
