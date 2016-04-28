function Y=Ylm(l,m,theta,phi)
P = legendre(l, cos(theta));
Y = sqrt( (2*l+1)*factorial((l-abs(m))) / (4*pi*factorial((l+abs(m))))) * ...
    P(abs(m)+1,:) .' * exp(1i * m * phi);
end
