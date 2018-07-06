mutable struct AMGPreconditioner{TML<:AMG.MultiLevel}
    ml::TML
end
AMGPreconditioner(A::AbstractMatrix) = AMGPreconditioner(ruge_stuben(A))

function UpdatePreconditioner!(C::AMGPreconditioner, A)
    C.ml = ruge_stuben(A)
    return C
end

\(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml) \ b
*(p::AMGPreconditioner, b) = AMG.Preconditioner(p.ml) * b
A_ldiv_B!(x, p::AMGPreconditioner, b) = A_ldiv_B!(x, AMG.Preconditioner(p.ml), b)
A_mul_B!(b, p::AMGPreconditioner, x) = A_mul_B!(b, AMG.Preconditioner(p.ml), x)
