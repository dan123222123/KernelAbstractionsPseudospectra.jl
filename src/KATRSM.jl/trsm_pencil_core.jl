# computes zB - A for a particular choice of i and j
function zBAij(i, j, z, A, B)
    (z * B[i, j]) - A[i, j]
end

function forward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for i = 1:m
        for j = 1:i-1
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
        b[i] = _pdiv(b[i], @inline zBAij(i, i, z, A, B))
    end
end
function column_oriented_forward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for j = 1:m
        b[j] = _pdiv(b[j], @inline zBAij(j, j, z, A, B))
        for i = j+1:m
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
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
function column_oriented_backward_solve_pencil!(b, z, A, B)
    m = size(A, 1)
    for j = m:-1:1
        b[j] = _pdiv(b[j], @inline zBAij(j, j, z, A, B))
        for i = 1:j-1
            b[i] -= b[j] * @inline zBAij(i, j, z, A, B)
        end
    end
end