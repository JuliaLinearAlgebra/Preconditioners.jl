using Preconditioners
using IterativeSolvers

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

@testset "$T preconditioner" for (T, F) in (("Diagonal", DiagonalPreconditioner), ("Incomplete Cholesky", CholeskyPreconditioner), ("AMG Ruge-Stuben", AMGPreconditioner{RugeStuben}), ("AMG Smoothed Aggregation", AMGPreconditioner{SmoothedAggregation}))
    A = sprand(1000, 1000, 0.01)
    A = A + A' + 30I
    p = F(A)
    @test isapprox(cg(A, A*ones(1000), Pl=p), ones(1000))

    A = sprand(1000, 1000, 0.01)
    A = A + A' + 30I
    UpdatePreconditioner!(p, A)
    @test isapprox(cg(A, A*ones(1000), Pl=p), ones(1000))
end
