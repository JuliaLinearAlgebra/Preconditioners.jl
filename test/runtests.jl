using LinearAlgebra, SparseArrays, Preconditioners, IterativeSolvers, Random

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

Random.seed!(1)

@testset "$T preconditioner" for (T, F) in (("Diagonal", DiagonalPreconditioner), ("Incomplete Cholesky", CholeskyPreconditioner), ("AMG Ruge-Stuben", RugeStuben), ("AMG Smoothed Aggregation", SmoothedAggregation))
    n = 100
    A = sprand(n, n, 10/n)
    A = A + A' + 50I
    atol = sqrt(eps(Float64))*n

    if F === CholeskyPreconditioner
        C1 = EmptyCholeskyPreconditioner(A)
        UpdatePreconditioner!(C1, A, 2)

        C2 = CholeskyPreconditioner(A, 2)
        @test isapprox(C1.L, C2.L, atol=atol)
    
        C3 = CholeskyPreconditioner(A, n)
        @test isapprox(norm(C3 \ ones(n) - A \ ones(n), Inf), 0.0, atol=0.001)
    end

    if F === RugeStuben || F === SmoothedAggregation
        p = AMGPreconditioner(F, A)
    else
        p = F(A)
    end
    @test isapprox(cg(A, A*ones(n), Pl=p), ones(n), atol=atol)
    if F === RugeStuben
        p = AMGPreconditioner(F, A)
        @test isapprox(cg(A, A*ones(n), Pl=p), ones(n), atol=atol)
    end
    A = sprand(n, n, 10/n)
    A = A + A' + 30I
    UpdatePreconditioner!(p, A)
    @test isapprox(cg(A, A*ones(n), Pl=p), ones(n), atol=atol)

    if F === RugeStuben || F === SmoothedAggregation
        p = AMGPreconditioner(F, Symmetric(A))
    else
        p = F(Symmetric(A))
    end

    @test isapprox(cg(A, A*ones(n), Pl=p), ones(n), atol=atol)
end
