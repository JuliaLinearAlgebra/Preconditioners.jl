mutable struct AMGPreconditioner{S, TML<:AMG.MultiLevel}
    ml::TML
end
struct RugeStuben end
struct SmoothedAggregation end

for (t, f) in [(:RugeStuben, :ruge_stuben), (:SmoothedAggregation, :smoothed_aggregation)]
    @eval begin
        function AMGPreconditioner(::Type{$t}, A::AbstractMatrix)
            if A isa Symmetric
                warn("Using the data field of the symmetric matrix input.")
                ml = $f(A.data)
            else
                ml = $f(A)
            end
            return AMGPreconditioner{$t, typeof(ml)}(ml)
        end

        function UpdatePreconditioner!(C::AMGPreconditioner{$t}, A::AbstractMatrix)
            if A isa Symmetric
                warn("Using the data field of the symmetric matrix input.")
                C.ml = $f(A.data)
            else
                C.ml = $f(A)
            end
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
