#=
Copied and modified from the MIT licensed package IncompleteSelectedInversion (https://github.com/ettersi/IncompleteSelectedInversion.jl) by Simon Etter.
=#

#=
 Sorted subset of {1,...,n}.
 Used to represent the row indices of a single column in F.
=#
struct SortedIntSet
    next::Vector{Int} 
    #=
     next[n+1] = first entry in the set
     If i is an element of the set, then next[i] is the 
     smallest j > i in the set, or n+1 if no such j exists.
    =#
    SortedIntSet(n) = new(Vector{Int}(undef, n+1))
end

start(s::SortedIntSet) = length(s.next)
Base.@propagate_inbounds done(s::SortedIntSet, p) = s.next[p] == length(s.next)
Base.@propagate_inbounds Base.iterate(s::SortedIntSet, p=start(s)) = done(s, p) ? nothing : (s.next[p], s.next[p])

Base.@propagate_inbounds function init!(s::SortedIntSet,i)
    next = s.next
    n = length(next)-1

    @boundscheck begin
        @assert 1 <= i <= n
    end

    @inbounds begin
        next[n+1] = i
        next[i] = n+1
        return s
    end
end

Base.@propagate_inbounds function Base.insert!(s::SortedIntSet,i,p)
    next = s.next
    n = length(next)-1

    @boundscheck begin
        @assert 1 <= i <= n
        @assert p == n+1 || 1 <= p <= i
    end
    @inbounds begin
        while next[p] < i
            p = next[p]
        end
        if next[p] == i
            return false
        end
        next[p],next[i] = i,next[p]
        return true
    end
end



#=
 Iterate through a sparse matrix in the order required
 by the LDL^T factorisation. 
=#

iterate_jkp(Ap,Ai) = Iteration_jkp(Ap,Ai)

struct Iteration_jkp{Ti}
    Ap::Vector{Ti}
    Ai::Vector{Ti}
    nextk::Vector{Int}
    nextp::Vector{Int}
end
struct Iteration_kp{Ti}
    Ap::Vector{Ti}
    Ai::Vector{Ti}
    nextk::Vector{Int}
    nextp::Vector{Int}
    j::Int
end

function Iteration_jkp(Ap,Ai)
    n = length(Ap)-1
    nextk = Vector{Int}(undef, n)
    fill!(nextk,n+1)
    nextp = Vector{Int}(undef, n)
    return Iteration_jkp(Ap,Ai,nextk,nextp)
end
start(jkp::Iteration_jkp) = 0
done(jkp::Iteration_jkp, j) = j == length(jkp.Ap) - 1
Base.@propagate_inbounds function Base.iterate(jkp::Iteration_jkp, j=start(jkp))
    done(jkp, j) && return nothing
    Ap = jkp.Ap
    Ai = jkp.Ai
    nextp = jkp.nextp
    nextk = jkp.nextk

    if j > 0
        for p in Ap[j]:Ap[j+1]-1
            i = Ai[p]
            if i > j
                nextp[j] = p
                nextk[i],nextk[j] = j,nextk[i]
                break
            end
        end
    end
    j += 1
    return (j, Iteration_kp(Ap,Ai,nextk,nextp,j)), j
end

Base.@propagate_inbounds start(kp::Iteration_kp) = kp.nextk[kp.j]
done(kp::Iteration_kp, k) = k > kp.j
Base.@propagate_inbounds function Base.iterate(kp::Iteration_kp, k=start(kp))
    done(kp, k) && return nothing
    Ap = kp.Ap
    Ai = kp.Ai
    nextp = kp.nextp
    nextk = kp.nextk

    pp = nextp[k]
    kk = nextk[k]
    nextp[k] += 1
    if nextp[k] < Ap[k+1]
        i = Ai[nextp[k]]
        nextk[i],nextk[k] = k,nextk[i]
    end
    return (k,pp:Ap[k+1]-1),kk
end

function checkmat(Ap,Ai)
    n = length(Ap)-1
    for j = 1:n
        @assert 1 <= Ai[Ap[j]] <= n
        for p in Ap[j]+1:Ap[j+1]-1
            @assert Ai[p-1] < Ai[p] <= n
        end
    end
end
function checkmat(Ap,Ai,Ax,Ay...) 
    @assert length(Ax) >= Ap[end]-1
    checkmat(Ap,Ai,Ay...)
end


# Note: There are better algorithms for computing the full symbolic LDL^T factorisation!
export symbolic_ldlt
function symbolic_ldlt(Ap,Ai)
    checkmat(Ap,Ai)

    @inbounds begin
        Ti = eltype(Ap)
        n = length(Ap)-1

        # Return variables
        Fp = Vector{Ti}(n+1); Fp[1] = 1
        Fi = Vector{Ti}(0)

        # Workspace for a single column
        Fji = SortedIntSet(n)

        # Main algorithm
        for (j,kvals) in iterate_jkp(Fp,Fi)
            # Initialise column
            init!(Fji,j)
            lasti = j
            for p in Ap[j]:Ap[j+1]-1
                i = Ai[p]
                if i <= j; continue; end
                insert!(Fji,i,lasti)
                lasti = i
            end

            # Pull updates into L[j:n,j]
            for (k,pvals) in kvals
                lasti = n+1
                for p in pvals
                    i = Fi[p]
                    insert!(Fji,i,lasti)
                    lasti = i
                end
            end

            # Copy temporary column into F
            for i in Fji
                push!(Fi,i)
            end
            Fp[j+1] = length(Fi)+1
        end
        return Fp,Fi
    end
end


export symbolic_cldlt
function symbolic_cldlt(Ap,Ai,c)
    checkmat(Ap,Ai)

    @inbounds begin
        Ti = eltype(Ap)
        n = length(Ap)-1

        # Return variables
        Fp = Vector{Ti}(undef, n+1); Fp[1] = 1
        Fi = Vector{Ti}(undef, 0)
        Fl = Vector{Ti}(undef, 0)

        # Workspace for a single column
        Fji = SortedIntSet(n)
        Fjl = Vector{Int}(undef, n)

        # Main algorithm
        for (j,kvals) in iterate_jkp(Fp,Fi)
            # Initialise column
            init!(Fji,j)
            Fjl[j] = 0 
            lasti = j
            for p in Ap[j]:Ap[j+1]-1
                i = Ai[p]
                if i <= j; continue; end
                insert!(Fji,i,lasti)
                Fjl[i] = 0 
                lasti = i
            end

            # Pull updates into L[j:n,j]
            for (k,pvals) in kvals
                lkj = Fl[first(pvals)]
                if lkj >= c; continue; end
                lasti = n+1
                for p in pvals
                    i = Fi[p]
                    lik = Fl[p]
                    Flij = lik + lkj + 1
                    if Flij <= c
                        if insert!(Fji,i,lasti)
                            Fjl[i] = Flij
                        else
                            Fjl[i] = min(Fjl[i],Flij)
                        end
                        lasti = i
                    end
                end
            end

            # Copy temporary column into F
            for i in Fji
                push!(Fi,i)
                push!(Fl,Fjl[i])
            end
            Fp[j+1] = length(Fi)+1
        end
        return Fp,Fi,Fl
    end
end


export numeric_ldlt
function numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj = Base.conj)
    checkmat(Ap,Ai,Av)
    checkmat(Fp,Fi)

    @inbounds begin
        Ti = eltype(Ap)
        Tv = eltype(Av)
        n = length(Ap)-1

        # Return variables
        Fv = Vector{Tv}(undef, length(Fi))

        # Workspace for a single column
        Fjv = Vector{Tv}(undef, n)

        # Main algorithm
        for (j,kvals) in iterate_jkp(Fp,Fi)
            # Initialise column
            for p in Fp[j]:Fp[j+1]-1
                Fjv[Fi[p]] = zero(Tv)
            end
            for p in Ap[j]:Ap[j+1]-1
                Fjv[Ai[p]] = Av[p]
            end

            # Pull updates into L[j:n,j]
            for (k,pvals) in kvals
                f = Fv[Fp[k]]*conj(Fv[first(pvals)])
                for p in pvals
                    # We compute a few dropped fill-ins here. It turns out computing 
                    # and discarding is faster than introducing a branch. 
                    Fjv[Fi[p]] -= Fv[p]*f
                end
            end

            # Copy temporary column into F
            d = Fjv[j]
            Fv[Fp[j]] = d
            for p in Fp[j]+1:Fp[j+1]-1
                Fv[p] = Fjv[Fi[p]]/d
            end
        end
        return Fv
    end
end


export τldlt
function τldlt(Ap,Ai,Av, τ=0; conj = Base.conj)
    checkmat(Ap,Ai,Av)

    @inbounds begin
        Ti = eltype(Ap)
        n = length(Ap)-1
        Tv = eltype(Av)

        # Return variables
        Fp = Vector{Ti}(undef, n+1); Fp[1] = 1
        Fi = Vector{Ti}(undef, 0)
        Fv = Vector{Tv}(undef, 0)

        # Workspace for a single column
        Fji = SortedIntSet(n)
        Fjv = Vector{Tv}(undef, n)

        # Main algorithm
        for (j,kvals) in iterate_jkp(Fp,Fi)
            # Initialise column
            init!(Fji,j)
            Fjv[j] = zero(Tv)
            lasti = j
            for p in Ap[j]:Ap[j+1]-1
                i = Ai[p]
                if i < j; continue; end
                insert!(Fji,i,lasti)
                Fjv[Ai[p]] = Av[p]
                lasti = i
            end

            # Pull updates into L[j:n,j]
            for (k,pvals) in kvals
                lasti = n+1
                Fvkj = conj(Fv[first(pvals)])
                if abs(Fvkj) < τ; continue; end
                Fvjj = Fv[Fp[k]]
                f = Fvjj*Fvkj
                for p in pvals
                    i = Fi[p]
                    Fvp = Fv[p]
                    if abs(Fvp) < τ; continue; end
                    if insert!(Fji,i,lasti)
                        Fjv[i] = -Fvp*f
                    else
                        Fjv[i] -= Fvp*f
                    end
                    lasti = i
                end
            end

            # Copy temporary column into F
            push!(Fi,j)
            d = Fjv[j]
            push!(Fv,d)
            for i in drop(Fji,1)
                push!(Fi,i)
                push!(Fv,Fjv[i]/d)
            end
            Fp[j+1] = length(Fi)+1
        end
        return Fp,Fi,Fv
    end
end

export selinv
function selinv(Fp,Fi,Fv; conj = Base.conj)
    checkmat(Fp,Fi,Fv)

    @inbounds begin
        Ti = eltype(Fp)
        Tv = eltype(Fv)
        n = length(Fp)-1

        # Return variables
        Bv = Vector{Tv}(undef, length(Fi))

        # Workspace for a single column
        Fjv = Vector{Tv}(undef, n)
        Bjv = Vector{Tv}(undef, n)

        # Main algorithm
        for j in reverse(1:n)
            # Initialise column
            for p in Fp[j]+1:Fp[j+1]-1
                Fjv[Fi[p]] = Fv[p]
                Bjv[Fi[p]] = zero(Tv)
            end

            # Pull updates into B[j+1:n,j]
            for p in Fp[j]+1:Fp[j+1]-1
                k = Fi[p]
                Fkj = Fjv[k]
                Bkj = Bjv[k] - Bv[Fp[k]]*Fkj
                for p in Fp[k]+1:Fp[k+1]-1
                    i = Fi[p]
                    Fij = Fjv[i]
                    Bik = Bv[p]
                    Bjv[i] -=      Bik *Fkj
                    Bkj    -= conj(Bik)*Fij
                    # ^ Not factoring out Bkj = Bjv[k] completely destroys loop performance
                end
                Bjv[k] = Bkj
            end

            # Copy temporary column into B
            for p in Fp[j]+1:Fp[j+1]-1
                Fjv[Fi[p]] = zero(Tv)
                Bv[p] = Bjv[Fi[p]]
            end

            # Deal with diagonal
            d = inv(Fv[Fp[j]])
            for p in Fp[j]+1:Fp[j+1]-1
                d -= conj(Bv[p])*Fv[p]
            end
            Bv[Fp[j]] = d
        end
        return Bv
    end
end


export ldlt, cldlt, τldlt
function ldlt(Ap,Ai,Av; conj = Base.conj)
    Fp,Fi = symbolic_ldlt(Ap,Ai)
    Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
    return Fp,Fi,Fv
end
function cldlt(Ap,Ai,Av, c; conj = Base.conj)
    Fp,Fi,Fl = symbolic_cldlt(Ap,Ai, c)
    Fv = numeric_ldlt(Ap,Ai,Av,Fp,Fi; conj=conj)
    return Fp,Fi,Fl,Fv
end
function ldlt(A; conj = Base.conj)
    Ap,Ai,Av = unpacksparse(A)
    Fp,Fi,Fv = ldlt(Ap,Ai,Av; conj = conj)
    return packsparse(Fp,Fi,Fv)
end
function cldlt(A, c; conj = Base.conj)
    Ap,Ai,Av = unpacksparse(A)
    Fp,Fi,Fl,Fv = cldlt(Ap,Ai,Av, c; conj = conj)
    return packsparse(Fp,Fi,Fv)
end
function τldlt(A, τ; conj = Base.conj)
    Ap,Ai,Av = unpacksparse(A)
    Fp,Fi,Fv = τldlt(Ap,Ai,Av, τ; conj = conj)
    return packsparse(Fp,Fi,Fv)
end
function selinv(F; conj = Base.conj)
    Fp,Fi,Fv = unpacksparse(F)
    Bv = selinv(Fp,Fi,Fv; conj=conj)
    return packsparse(Fp,Fi,Bv)
end



#=
 Utility functions
=#

export dropfillin
function dropfillin(Fp,Fi,Fl,c)
    Ti = eltype(Fp)
    n = length(Fp)-1
    F̃p = Vector{Ti}(undef, n+1)
    F̃p[1] = 1
    F̃i = Vector{Ti}(undef, 0)
    F̃l = Vector{Ti}(undef, 0)
    for j = 1:n
        for p in Fp[j]:Fp[j+1]-1
            if Fl[p] <= c
                push!(F̃i,Fi[p])
                push!(F̃l,Fl[p])
            end
        end
        F̃p[j+1] = length(F̃i)+1
    end
    return F̃p,F̃i,F̃l
end

export unpacksparse, packsparse
unpacksparse(A) = A.colptr,A.rowval,A.nzval
function packsparse(p,i,v)
    n = length(p)-1
    return SparseMatrixCSC(n,n,p,i,v)
end
