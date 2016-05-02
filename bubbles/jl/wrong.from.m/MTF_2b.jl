
include("funcs.jl")

a = 0.005;
freq = 650;
N = 2;
ww = 2*pi*freq;
rho0 = 1030; c0 = 1500; k0 = ww / c0;
rho1 = 1.18; c1 = 344.4475;  k1=ww / c1;
a1 = rho1 / rho0;    a1inv=1 / a1;
PSM = 0;
Amp = 1;

PSL = Array(Complex, 4)
for l in 0:3
    z0 = a * k0;
    K0 = -1im * k0^2 * a^4 * dsbesf1(l, z0) * hn(z0, l);
    W0 = -1im * k0^3 * a^4 * dsbesf1(l, z0) * dshaf11(l, z0);
    Ks0 = -1im * k0^2 * a^4 * jn(z0, l) * dshaf11(l, z0);
    V0= 1im * k0 * a^4 * hn(z0, l) * jn(z0, l);

    A0 = [-K0 V0; W0 Ks0];

    z1 = k1 * a;
    K1 = 1im * k1^2 * a^4 * jn(z1, l) * dshaf11(l, z1);
    Ks1 = 1im * k1^2 * a^4 * dsbesf1(l, z1) * hn(z1, l);
    W1 = -1im * k1^3 * a^4 * dshaf11(l, z1) * dsbesf1(l, z1);
    V1 = 1im * k1 * a^4 * jn(z1, l) * hn(z1, l);

    A1 = [-K1 V1; W1 Ks1];

    for m in -l:l
        dth = pi / 5;
        dphi = pi / 5;
        dV0 = 0;
        delta = a^4 * (dth*dphi)^2;

        for th1 in dth:dth:(pi-dth)
            for phi1 in 0:dphi:(2*pi-dphi)
                sc = sin(th1) * cos(phi1) - 40
                ss = sin(th1) * sin(phi1)
                co = cos(th1)
                Y1_delta =  conj(Ylm(l,m,th1,phi1)) * sin(th1) * delta
                for th2 in dth:dth:(pi-dth)
                    for phi2 in 0:dphi:(2*pi-dphi)
                        SC = sc - sin(th2) * cos(phi2)
                        SS = ss - sin(th2) * sin(phi2)
                        CO = co - cos(th2)
                        dd = a * sqrt(SC^2 + SS^2 + CO^2)
                        g0 = exp(1im * k0 * dd) / (4*pi * dd);
                        dV0 += Ylm(l,m,th2,phi2) * g0 *sin(th2) * Y1_delta
                    end
                end
            end
        end
        print(dV0)
        print("\n")

        V01 = dV0;
        K01 = 0;
        W01 = 0;
        Ks01 = 0;
        A01 = [-K01 V01; W01 Ks01];

        Xx = a^2 * [1 0; 0 -a1];
        Xinv = a^2 * [1 0; 0 -a1inv];

        N_A0 = zeros(Complex, 2*N, 2*N)
        N_A1 = zeros(Complex, 2*N, 2*N)
        N_Xx = zeros(Complex, 2*N, 2*N)
        N_Xinv = zeros(Complex, 2*N, 2*N)


        for iN in 1:N
            N_A0[2*iN-1:2*iN,2*iN-1:2*iN] = A0;
            N_A1[2*iN-1:2*iN,2*iN-1:2*iN] = A1;
            N_Xx[2*iN-1:2*iN,2*iN-1:2*iN] = Xx;
            N_Xinv[2*iN-1:2*iN,2*iN-1:2*iN] = Xinv;
        end

        N_A0[1:2, 3:4] = A01;
        N_A0[3:4, 1:2] = A01;
        M = [N_A0 -N_Xinv; -N_Xx N_A1];

        SrcX = -10;

        phs1 = exp(-1im * k0 * (SrcX + (20*a)));
        phs2 = exp(-1im * k0 * (SrcX - (20*a)));

        b = 2*sqrt((2*l+1)*pi)*Amp*(1im)^l*a^2*[-jn(z0,l)*phs1; dsbesf1(l,z0)*k0*phs1;-jn(z0,l)*phs2; dsbesf1(l,z0)*k0*phs2; jn(z0,l)*phs1; a1*dsbesf1(l,z0)*k0*phs1;jn(z0,l)*phs2; a1*dsbesf1(l,z0)*k0*phs2];

        pos =[(-20*a) 0 0;(20*a) 0 0];

        x = M\b;

        D1 = dsbesf1(l, k0*a);
        J1 = jn(k0*a, l);
        R = [-1 0 0];
        DRecN = Array(Float64, N)
        psb2 = Array(Complex, N)
        for ii in 1:N
            th, phi, DRec = cart2sph(R[1]-pos[ii,1],R[2]-pos[ii,2],R[3]-pos[ii,3]);
            DRecN[ii] = DRec;

            P = legendre(l, 0, cos(th));
            Y = sqrt((2*l+1)*factorial((l-abs(m)))/(4*pi*factorial((l+abs(m)))))*exp(1im*m*phi)*P;
            z2 = hn(k0*DRec, l);

            psb2[ii]=1im*k0*a^2*Y*(k0*x[2*ii-1]*D1*z2+x[2*ii]*J1*z2);
            psb2[ii]=abs(psb2[ii]) * exp(-1im*angle(psb2[ii]));
        end
  ps = sum(psb2);
  PSM += ps;

   end
PSL[l+1]=PSM;
end

conv=abs(PSL-PSL[end])/abs(PSL[end])
print("\n")
print(conv)
print("\n")
