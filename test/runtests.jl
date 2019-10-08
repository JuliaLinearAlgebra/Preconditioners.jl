using LinearAlgebra, SparseArrays, Preconditioners, IterativeSolvers, Random, Test

Random.seed!(1)

function test_matrix(A, F, atol)
    n = size(A, 1)
    b = ones(n)
    if F === CholeskyPreconditioner
        C1 = EmptyCholeskyPreconditioner(A)
        UpdatePreconditioner!(C1, A, 2)

        C2 = CholeskyPreconditioner(A, 2)
        @test isapprox(C1.L, C2.L, atol=atol)

        C3 = CholeskyPreconditioner(Symmetric(A), 2)
        @test isapprox(C2.L, C3.L, atol=atol)

        C4 = CholeskyPreconditioner(A, n)
        @test isapprox(norm(C4 \ b - Symmetric(A) \ b, Inf), 0.0, atol=atol)
    end
    if F === RugeStuben || F === SmoothedAggregation
        p = AMGPreconditioner(F, A)
    else
        p = F(A)
    end
    @test isapprox(p \ b, ldiv!(p, b), atol=atol)
    @test isapprox(ldiv!(copy(b), p, b), ldiv!(p, b), atol=atol)
    @test isapprox(cg(A, A*b, Pl=p), b, atol=atol)
    if F === RugeStuben
        p = AMGPreconditioner(F, A)
        @test isapprox(cg(A, A*b, Pl=p), b, atol=atol)
    end
    A = sprand(n, n, 10/n)
    A = A + A' + 30I
    UpdatePreconditioner!(p, A)
    @test isapprox(cg(A, A*b, Pl=p), b, atol=atol)
end

@testset "$T preconditioner" for (T, F) in (("Diagonal", DiagonalPreconditioner), ("Incomplete Cholesky", CholeskyPreconditioner), ("AMG Ruge-Stuben", RugeStuben), ("AMG Smoothed Aggregation", SmoothedAggregation))
    n = 100
    atol = 0.01
    A = sprand(n, n, 10/n)
    A = A + A' + 50I
    test_matrix(A, F, atol)
    test_matrix(Symmetric(A), F, atol)
    test_matrix(Hermitian(A), F, atol)
end
