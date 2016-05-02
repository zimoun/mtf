
function fun_cart2sph(F)
    function f(n, z)
        sqrt(0.5 * pi ./ z) .* F(n + 0.5, z);
    end
    return f
end

function fun_derivative(F)
    function f(n, z)
        (n / z) * F(n, z) - F(n + 1, z)
    end
    return f
end

J = fun_cart2sph(besselj)
Jp = fun_derivative(J)

Y = fun_cart2sph(bessely)
Yp = fun_derivative(Y)

H1(n, z) = J(n, z) + 1im * Y(n, z)
H1p(n, z) = Jp(n, z) + 1im * Yp(n, z)


function get_size(Nmodes)
    size = 1
    for m in 1:Nmodes
        size += length(-m:m)
    end
    return size, size
end

function self_interaction(Nmodes, k, rad, formula)
    size = get_size(Nmodes)
    N, _ = size
    Op = zeros(Complex, size)
    for i in 1:N
        Op[i, i] = formula(i-1, k, rad)
    end
    return Op
end

formula_single(l, k, rad) = im * k * (rad^4) * J(l, k * rad) * H1(l, k * rad)
formula_double(l, k, rad) = im * (k^2) * (rad^4) * Jp(l, k * rad) * H1(l, k * rad)
formula_adjoint(l, k, rad) = im * (k^2) * (rad^4) * J(l, k * rad) * H1p(l, k * rad)
formula_hyper(l, k, rad) = -im * (k^3) * (rad^4) * Jp(l, k * rad) * H1p(l, k * rad)
formula_identity(l, k, rad) = rad^2

for what in [:single :double :adjoint :hyper :identity]
    @eval function $(symbol(string(what, "_layer")))(Nmodes, k, rad=1)
        """
        Normal is outward of the domain.
        Therefore
            K0 = -K ; K1 = +K
            Q0 = -Q ; Q1 = +Q
        """
        Op = self_interaction(Nmodes, k, rad, $(symbol(string("formula_", what))))
        return Op
    end
end

k0 = 1.
k1 = 2.

rad = 1.

Nmodes = 8
shape = get_size(Nmodes)
N, _ = shape

V0 = single_layer(Nmodes, k0, rad)
K0 = double_layer(Nmodes, k0, rad)
Q0 = adjoint_layer(Nmodes, k0, rad)
W0 = hyper_layer(Nmodes, k0, rad)

V1 = single_layer(Nmodes, k1, rad)
K1 = double_layer(Nmodes, k1, rad)
Q1 = adjoint_layer(Nmodes, k1, rad)
W1 = hyper_layer(Nmodes, k1, rad)

Id = identity_layer(Nmodes, 1., rad)

A = zeros(Complex, 4*N, 4*N)
X = zeros(Complex, 4*N, 4*N)

############################
A[1:N, 1:N] = K0
A[1:N, N+1:2N] = V0
X[1:N, 2N+1:3N] = Id

A[N+1:2N, 1:N] = W0
A[N+1:2N, N+1:2N] = -Q0
X[N+1:2N, 3N+1:4N] = -Id
############################
X[2N+1:3N, 1:N] = Id
A[2N+1:3N, 2N+1:3N] = -K1
A[2N+1:3N, 3N+1:4N] = V1

X[3N+1:4N, N+1:2N] = Id
A[3N+1:4N, 2N+1:3N] = W1
A[3N+1:4N, 3N+1:4N] = Q1
############################

A0 = [K0 V0; W0 -Q0]
A1 = [-K1 V1; W1 Q1]

AA = zeros(Complex, 4N ,4N)
AA[1:2N, 1:2N] = A0
AA[2N+1:end, 2N+1:end] = A1

two = 1 + 0im
D0, _ = eig(two*A0)
D1, _ = eig(two*A1)
D, _ = eig(two*A)
DD, _ = eig(two*AA)


done = true
