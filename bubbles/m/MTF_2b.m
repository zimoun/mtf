clc
clear all
% close all
%%

a=0.005;
freq=650;
N=2;
ww=2*pi*freq;
rho0=1030; c0=1500; k0=ww/c0;
rho1=1.18; c1=344.4475;  k1=ww/c1;
a1=rho1/rho0;    a1inv=1/a1;
PSM=0;
Amp=1;

for l=0:3;
    l
  z0=a*k0;
  K0=-1i*k0^2*a^4*dsbesf1(l,z0)*hn(z0,l);
  W0=-1i*k0^3*a^4*dsbesf1(l,z0)*dshaf11(l,z0);
  Ks0=-1i*k0^2*a^4*jn(z0,l)*dshaf11(l,z0);
  V0=1i*k0*a^4*hn(z0,l)*jn(z0,l);

  A0=[-K0 V0; W0 Ks0];

  z1=k1*a;
  K=1i*k1^2*a^4*jn(z1,l)*dshaf11(l,z1);
  Ks=1i*k1^2*a^4*dsbesf1(l,z1)*hn(z1,l);
  W=-1i*k1^3*a^4*dshaf11(l,z1)*dsbesf1(l,z1);
  V=1i*k1*a^4*jn(z1,l)*hn(z1,l);

  A=[-K V; W Ks];

  for m=-l:l;

    dth=pi/5;
    dphi=pi/5;
    dV0=0;
    for th1=dth:dth:(pi-dth)
     th1/dth
     for phi1=0:dphi:(2*pi-dphi)
      for th2=dth:dth:(pi-dth) %elevacion
       for phi2=0:dphi:(2*pi-dphi)
         dd=sqrt((a*sin(th1)*cos(phi1)-20*a-a*sin(th2)*cos(phi2)-20*a)^2+ (a*sin(th1)*sin(phi1)-a*sin(th2)*sin(phi2))^2+ (a*cos(th1)-a*cos(th2))^2);
         g0=exp(1i*k0*dd)/4/pi/dd;
         dV0=dV0+Ylm(l,m,th2,phi2)*g0*a^4*sin(th2)*dth*dphi*conj(Ylm(l,m,th1,phi1))*sin(th1)*dth*dphi;
       end
      end
     end
    end
    display(dV0)

  V01=dV0;
  K01=0;
  W01=0;
  Ks01=0;
  A01=[-K01 V01; W01 Ks01 ];


%%
  Xx=a^2*[1 0;0 -a1];
  Xinv=a^2*[1 0;0 -a1inv];

  for iN=1:N
    N_A0(2*iN-1:2*iN,2*iN-1:2*iN)=A0;
    N_A(2*iN-1:2*iN,2*iN-1:2*iN)=A;
    N_Xx(2*iN-1:2*iN,2*iN-1:2*iN)=Xx;
    N_Xinv(2*iN-1:2*iN,2*iN-1:2*iN)=Xinv;
  end
  N_A0(1:2,3:4)=A01;
  N_A0(3:4,1:2)=A01;
  M=[N_A0 -N_Xinv; -N_Xx N_A];

% calculando el  b ,con fases

  SrcX=-10;

  phs1=exp(-1i*k0*(SrcX+(20*a)));
  phs2=exp(-1i*k0*(SrcX-(20*a)));

  b= 2*sqrt((2*l+1)*pi)*Amp*(1i)^l*a^2*[-jn(z0,l)*phs1; dsbesf1(l,z0)*k0*phs1;-jn(z0,l)*phs2; dsbesf1(l,z0)*k0*phs2; jn(z0,l)*phs1; a1*dsbesf1(l,z0)*k0*phs1;jn(z0,l)*phs2; a1*dsbesf1(l,z0)*k0*phs2];

  pos=[(-20*a) 0 0;(20*a) 0 0];

  x = M\b;


  D1=dsbesf1(l,k0*a);
  J1=jn(k0*a,l);
  R=[-1 0 0];
  for ii=1:N
   [th,phi,DRec]=cart2sph(R(1)-pos(ii,1),R(2)-pos(ii,2),R(3)-pos(ii,3));
   DRecN(ii)=DRec;

   P=legendre(l,cos(th));
   Y=sqrt((2*l+1)*factorial((l-abs(m)))/(4*pi*factorial((l+abs(m)))))*exp(1i*m*phi)*P(1,1);
   z2=hn(k0*DRec,l);

   psb2(ii)=1i*k0*a^2*Y*(k0*x(2*ii-1)*D1*z2+x(2*ii)*J1*z2);
   psb2(ii)=abs(psb2(ii)).*exp(-1i*angle(psb2(ii)));
  end
  ps=sum(psb2);
  PSM=PSM+ps;

   end
PSL(l+1)=PSM;
end

conv=abs(PSL-PSL(end))/abs(PSL(end))
display(conv)
% figure(7)
% hold on
% plot(log10(conv),'k')
