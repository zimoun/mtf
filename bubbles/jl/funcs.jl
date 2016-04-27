using GSL: sf_legendre_Plm

function jn(z, n)
    j = sqrt(0.5 * pi ./ z) .* besselj(n + 0.5, z);
    return j
end

function hn(z, n)
    h = sqrt(0.5 * pi ./ z) .* besselh( n + 0.5, z);
    return h
end


function dsbesf1(m, z)
    """
      This function is based on information given in "Vibration and Sound",
      by Philip M. Morse, 2nd Edition., pp. 316-317.

      It uses recursion relations to compute the first derivative
      of spherical Bessel functions
      """

    if m == 0
        y = -sqrt(0.5 * pi ./ z) .* besselj(1.5, z);
	    return y
    end

    y = ((m / (2*m+1)) * sqrt(0.5 * pi ./ z) .* besselj(m-0.5, z));
    t = ((m+1) / (2*m+1)) * sqrt(0.5 * pi ./ z ) .* besselj(m+1.5, z);
    y -= t;
    return y
end

function dshaf11(m, z)
    """
      This function is based on information given in "Vibration and Sound",
      by Philip M. Morse, 2nd Edition., pp. 316-317.
    """

    if m == 0
        y = -sqrt(0.5 * pi ./ z) .* besselh(1.5, z);
	    return y
    end

    y = (m / (2*m+1)) * sqrt(0.5 * pi ./ z) .* besselh(m-0.5, z);
    t = ((m+1) / (2*m+1)) * sqrt(0.5 * pi ./ z ) .* besselh(m+1.5, z);
    y -= t;
    return y
end

function legendre(l, m, z)
    P = sf_legendre_Plm(l, m, z)
    return P
end
function Ylm(l, m, theta, phi)
    P = legendre(l, abs(m), cos(theta));
    Y = sqrt((2*l+1)*factorial((l-abs(m)))/(4*pi*factorial((l+abs(m)))));
    Y *= P * exp(1im * m * phi);
    return Y
end

function cart2sph(x, y, z)
    xy2 = x^2 + y^2
    r = sqrt(xy2 + z^2)
    elev = atan2(z, sqrt(xy2))
    az = atan2(y, x)
    return az, elev, r
end
