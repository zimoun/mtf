alpha = 10 ;
k = 20;
rad = 0.3;

eps = {1} ;

div = #eps[];

Printf("k= %f  ,   alpha= %f  ,  Ndom= %f", k, alpha, #eps[]);

A = 1;
B = 0.5;

EA = A*rad / div;
EB = B*rad / div;
ea = 0;
eb = 0;


//

Point(1) = {0.0, 0.0, 0.0, 1.};

For ii In {1:#eps[]}
p = newp;
l = newl;
ll = newll;
Ss = news;

If (ii != 1)
  ea = ea + EA;
  eb = eb + EB;
EndIf

If (ii != #eps[])
  perm = eps[ii];
EndIf
If (ii == #eps[])
  perm = eps[ii-1];
EndIf


lc = 2*Pi / (alpha * perm * k);


Point(p+2) = {A*rad-ea, 0.0, 0.0, lc};
Point(p+3) = {0, B*rad-eb, 0.0, lc};

Point(p+4) = {-A*rad+ea, 0, 0.0, lc};
Point(p+5) = {0, -B*rad+eb, 0.0, lc};

Point(p+6) = {0, 0, -B*rad+eb, lc};
Point(p+7) = {0, 0, B*rad-eb, lc};

Ellipse(l+1) = {p+2, 1, 1, p+3};

Ellipse(l+2) = {p+3, 1, 1, p+4};

Ellipse(l+3) = {p+4, 1, 1, p+5};

Ellipse(l+4) = {p+5, 1, 1, p+2};

Ellipse(l+5) = {p+3, 1, 1, p+6};
Ellipse(l+6) = {p+6, 1, 1, p+5};
Ellipse(l+7) = {p+5, 1, 1, p+7};
Ellipse(l+8) = {p+7, 1, 1, p+3};

Ellipse(l+9) = {p+2, 1, 1, p+7};

Ellipse(l+10) = {p+7, 1, 1, p+4};

Ellipse(l+11) = {p+4, 1, 1, p+6};

Ellipse(l+12) = {p+6, 1, 1, p+2};

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


Physical Surface(ii) = {Ss+28, Ss+26, Ss+16, Ss+14, Ss+20, Ss+24, Ss+22, Ss+18};
EndFor



Mesh 2;
Save "ellipse-concentric.msh" ;
