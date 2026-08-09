# (zB - A)[i, j]
function zBAij(i, j, z, A, B)
    (z * B[i, j]) - A[i, j]
end

# Pencil element selectors, `Val`-dispatched for B = I ("eye"), shared by the column /
# tiled-panel / warp solves. `_piv_elem` is the diagonal M[j,j] (pivot); `_offd_elem` is an
# off-diagonal M[i,j]. Eye methods never reference B (z·B reduces to z on-diagonal, 0 off it), so
# B may be `nothing` there.
@inline _piv_elem(::Val{false}, j, z, A, B) = @inline zBAij(j, j, z, A, B)
@inline _piv_elem(::Val{true}, j, z, A, B) = z - A[j, j]
@inline _offd_elem(::Val{false}, i, j, z, A, B) = @inline zBAij(i, j, z, A, B)
@inline _offd_elem(::Val{true}, i, j, z, A, B) = -A[i, j]

function forward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for i in 1:m
        for j in 1:(i - 1)
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
        b[i] = _pdiv(b[i], @inline zBAij(i, i, z, A, B))
    end
end

function backward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for i in m:-1:1
        for j in (i + 1):m
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
        b[i] = _pdiv(b[i], @inline zBAij(i, i, z, A, B))
    end
end
