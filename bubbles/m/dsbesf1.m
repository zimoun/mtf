function y = dsbesf1(m,z)
%
%  This function is based on information given in "Vibration and Sound",
%  by Philip M. Morse, 2nd Edition., pp. 316-317.
%
%  It uses recursion relations to compute the first derivative
%  of spherical Bessel functions


if m==0
    y=-sqrt(0.5*pi ./z).*besselj(1.5,z);
	return
end

y=((m/(2*m+1))*sqrt(0.5*pi ./z).*besselj(m-0.5,z))-(((m+1)/(2*m+1))*sqrt(0.5*pi ./z).*besselj(m+1.5,z));
