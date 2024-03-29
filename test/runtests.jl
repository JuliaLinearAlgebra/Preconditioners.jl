using LinearAlgebra: I, diag, ldiv!, norm, Symmetric, Hermitian, eigen
using SparseArrays: sprand
using Preconditioners: CholeskyPreconditioner, DiagonalPreconditioner
using Preconditioners: RugeStuben, SmoothedAggregation
using Preconditioners: UpdatePreconditioner!, AMGPreconditioner
using IterativeSolvers: cg, lobpcg
using Random: seed!
using Test: @test, @testset, @test_throws
using OffsetArrays: OffsetMatrix, OffsetVector

seed!(1)

function test_matrix(A, F, atol)
    n = size(A, 1)
    b = ones(n)
    if F === CholeskyPreconditioner
        C = CholeskyPreconditioner(A, n)
        @test isapprox(norm(C \ b - Symmetric(A) \ b, Inf), 0.0, atol=atol)
        UpdatePreconditioner!(C, A, 2)

        C2 = CholeskyPreconditioner(A, n; check_tril = true) # check kwargs
    end
    if F === RugeStuben || F === SmoothedAggregation
        p = AMGPreconditioner(F, A)
    else
        p = F(A)
    end
    if F === DiagonalPreconditioner
        @test p.D isa Vector
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

    # Test with OffsetArrays
    if F === DiagonalPreconditioner
        OA = OffsetMatrix(A, (-1, -1))
        OD = OffsetVector(diag(A, 0), -1)
        OP = DiagonalPreconditioner(OD)
        b = ones(n)
        ob = OffsetVector(b, -1)
        err = ArgumentError("incompatible indices for input arguments")
        @test_throws err OP \ b
        @test_throws err ldiv!(OP, b)
        @test_throws err ldiv!(b, OP, b)
        @test OP \ ob ≈ ldiv!(OP, copy(ob)) ≈ ldiv!(similar(ob), OP, ob)
        @test_throws err UpdatePreconditioner!(OP, A)
        x = first(OP.D)
        UpdatePreconditioner!(OP, OA + 2I)
        @test first(OP.D) == x + 2
    end
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

@testset "AMG + LOBPCG" begin
    A = sprand(1000, 1000, 0.01)
    A = A + A' + 30I
    X0 = randn(1000,1)
    tol = 1e-3
    F = lobpcg(A, false, X0, tol=tol, maxiter=200, P=AMGPreconditioner{SmoothedAggregation}(A))
    @test eigen(Matrix(A)).values[1] ≈ F.λ[1] atol = 1e-3

    X0 = randn(1000,2)
    tol = 1e-3
    F = lobpcg(A, false, X0, tol=tol, maxiter=200, P=AMGPreconditioner{SmoothedAggregation}(A))
    @test eigen(Matrix(A)).values[1:2] ≈ F.λ atol = 1e-3
end
