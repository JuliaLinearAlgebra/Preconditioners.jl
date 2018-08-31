mutable struct AMGPreconditioner{T, TML<:AMG.MultiLevel}
    ml::TML
end
struct RugeStuben end
struct SmoothedAggregation end

for (t, f) in [(:RugeStuben, :ruge_stuben), (:SmoothedAggregation, :smoothed_aggregation)]
    @eval begin
        function AMGPreconditioner(::Type{$t}, A::AbstractMatrix)
            if A isa Symmetric || A isa Hermitian
                @warn("Using the data field of the symmetric/Hermitian matrix input.")
                ml = $f(A.data)
            else
                ml = $f(A)
            end
            return AMGPreconditioner{$t, typeof(ml)}(ml)
        end

        function UpdatePreconditioner!(C::AMGPreconditioner{$t}, A::AbstractMatrix)
            if A isa Symmetric || A isa Hermitian
                @warn("Using the data field of the symmetric/Hermitian matrix input.")
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
