alpha = 20 ;
k = 1;
rad = 1.;

eps = {2, 4, 16} ;

div = #eps[];

Printf("k= %f  ,   alpha= %f  ,  Ndom= %f", k, alpha, #eps[]);

E = rad / div;
e = 0;

//

tag = news;

Point(1) = {0.0, 0.0, 0.0, 1.};

For ii In {1:#eps[]}
p = newp;
l = newl;
ll = newll;
Ss = news;

If (ii != 1)
  e = e + E;
EndIf

If (ii != #eps[])
  perm = Sqrt(eps[ii]);
EndIf
If (ii == #eps[])
  perm = Sqrt(eps[ii-1]);
EndIf


lc = 2*Pi / (alpha * perm * k);


Point(p+2) = {rad-e, 0.0, 0.0, lc};
Point(p+3) = {0, rad-e, 0.0, lc};

Circle(l+1) = {p+2, 1, p+3};

Point(p+4) = {-rad+e, 0, 0.0, lc};
Point(p+5) = {0, -rad+e, 0.0, lc};

Circle(l+2) = {p+3, 1, p+4};

Circle(l+3) = {p+4, 1, p+5};

Circle(l+4) = {p+5, 1, p+2};

Point(p+6) = {0, 0, -rad+e, lc};
Point(p+7) = {0, 0, rad-e, lc};
Circle(l+5) = {p+3, 1, p+6};
Circle(l+6) = {p+6, 1, p+5};
Circle(l+7) = {p+5, 1, p+7};
Circle(l+8) = {p+7, 1, p+3};

Circle(l+9) = {p+2, 1, p+7};

Circle(l+10) = {p+7, 1, p+4};

Circle(l+11) = {p+4, 1, p+6};

Circle(l+12) = {p+6, 1, p+2};

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
Save "sphere-concentric.msh" ;
