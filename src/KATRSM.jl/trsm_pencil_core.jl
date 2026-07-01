# computes zB - A for a particular choice of i and j
function zBAij(i, j, z, A, B)
    (z * B[i, j]) - A[i, j]
end

# Pencil element selectors, specialized at COMPILE TIME on whether B = I (the "eye" case). These are
# the single source of truth for the pencil arithmetic shared by the column / tiled-panel / warp
# solves. `_piv_elem` is the diagonal element M[j,j] (the pivot); `_offd_elem` is a strictly
# off-diagonal element M[i,j] (the update). For the eye case (`Val{true}`) the z·B term reduces — the
# diagonal to `z` (B[j,j]=1) and the off-diagonal to 0 (B[i,j]=0) — so the eye methods NEVER reference
# B. Because the choice is `Val` dispatch (not a runtime `if`), the eye-specialized kernel contains no
# `B[...]` index in its IR → no B DRAM load (the eye bandwidth win), and B may even be `nothing` there.
@inline _piv_elem(::Val{false}, j, z, A, B) = @inline zBAij(j, j, z, A, B)
@inline _piv_elem(::Val{true},  j, z, A, B) = z - A[j, j]
@inline _offd_elem(::Val{false}, i, j, z, A, B) = @inline zBAij(i, j, z, A, B)
@inline _offd_elem(::Val{true},  i, j, z, A, B) = -A[i, j]

function forward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for i = 1:m
        for j = 1:i-1
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
        b[i] = _pdiv(b[i], @inline zBAij(i, i, z, A, B))
    end
end

function backward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for i = m:-1:1
        for j = i+1:m
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
        b[i] = _pdiv(b[i], @inline zBAij(i, i, z, A, B))
    end
end