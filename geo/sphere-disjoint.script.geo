Include "params.geo";
//// params.geo is written by python script
//// otherwise,
//// the parameters: k, alpha, eps, L, rad
//// !!!!! MUST be given !!!!!
//// for example:
// alpha = 10;
// k = 0.1;
// eps = { 2, 3, 4 };
// rad = { 1, 1, 0.5 };
// L = { 0, 0.5, 1 };
// name = 'my.msh';


Printf("k= %f  ,   alpha= %f  ,  Ndom= %f", k, alpha, #eps[]);

If (#eps[] != #rad[])
  Printf("========= ERROR ========= #eps=%f  | %f=#rad", #eps[], #rad[]);
EndIf
If (#eps[] != #L[])
  Printf("========= ERROR ========= #eps=%f  | %f=#L", #eps[], #L[]);
EndIf

e = L[0];

//

tag = news;

For ii In {1:#eps[]}
p = newp;
l = newl;
ll = newll;
Ss = news;

If (ii != 1)
  e = e + L[ii-1] + rad[ii-2] + rad[ii-1];
EndIf

If (ii != #eps[])
  perm = Sqrt(eps[ii]);
EndIf
If (ii == #eps[])
  perm = Sqrt(eps[ii-1]);
EndIf
r = rad[ii-1];

lc = 2*Pi / (alpha * perm * k);
Printf("eps=%f  ;  lc=%f", perm, lc);

X = 1;
Y = 0;
Z = 0;

x = e*Sqrt(X)/Sqrt(X+Y+Z);
y = e*Sqrt(Y)/Sqrt(X+Y+Z);
z = e*Sqrt(Z)/Sqrt(X+Y+Z);


Point(p+1) = {x, y, z, 1.};

Point(p+2) = {r+x, y, z, lc};
Point(p+3) = {x, r+y, z, lc};

Circle(l+1) = {p+2, p+1, p+3};

Point(p+4) = {-r+x, y, z, lc};
Point(p+5) = {x, -r+y, z, lc};

Circle(l+2) = {p+3, p+1, p+4};

Circle(l+3) = {p+4, p+1, p+5};

Circle(l+4) = {p+5, p+1, p+2};

Point(p+6) = {x, y, -r+z, lc};
Point(p+7) = {x, y, r+z, lc};
Circle(l+5) = {p+3, p+1, p+6};
Circle(l+6) = {p+6, p+1, p+5};
Circle(l+7) = {p+5, p+1, p+7};
Circle(l+8) = {p+7, p+1, p+3};

Circle(l+9) = {p+2, p+1, p+7};

Circle(l+10) = {p+7, p+1, p+4};

Circle(l+11) = {p+4, p+1, p+6};

Circle(l+12) = {p+6, p+1, p+2};

Line Loop(ll+13) = {l+2, l+8, -(l+10)};
Ruled Surface(Ss+14) = {ll+13};

Line Loop(ll+15) = {l+10, l+3, l+7};
Ruled Surface(Ss+16) = {ll+15};

Line Loop(ll+17) = {-(l+8), -(l+9), l+1};
Ruled Surface(Ss+18) = {ll+17};

Line Loop(ll+19) = {-(l+11), -(l+2), l+5};
Ruled Surface(Ss+20) = {ll+19};

Line Loop(ll+21) = {-(l+5), -(l+12), -(l+1)};
Ruled Surface(Ss+22) = {ll+21};

Line Loop(ll+23) = {-(l+3), l+11, l+6};
Ruled Surface(Ss+24) = {ll+23};

Line Loop(ll+25) = {-(l+7), l+4, l+9};
Ruled Surface(Ss+26) = {ll+25};

Line Loop(ll+27) = {-(l+4), l+12, -(6+l)};
Ruled Surface(Ss+28) = {ll+27};


Physical Surface(tag) = {Ss+28, Ss+26, Ss+16, Ss+14, Ss+20, Ss+24, Ss+22, Ss+18};
tag++;

EndFor



Mesh 2;
Save Sprintf(Str(name));
