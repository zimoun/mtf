function y = dshaf11(m,z)
%
%  This function is based on information given in "Vibration and Sound",
%  by Philip M. Morse, 2nd Edition., pp. 316-317.
%

if m==0
    y=-sqrt(0.5*pi ./z).*besselh(1.5,z);
	return
end

y=((m/(2*m+1))*sqrt(0.5*pi ./z).*besselh(m-0.5,z))-(((m+1)/(2*m+1))*sqrt(0.5*pi ./z).*besselh(m+1.5,z));
