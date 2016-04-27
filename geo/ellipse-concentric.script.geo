Include "params_tmp.geo";
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

div = #eps[];

Printf("k= %f  ,   alpha= %f  ,  Ndom= %f", k, alpha, #eps[]);

If (#rad[] != 1)
  rad = rad[1];
EndIf
If (#A[] != 1)
  A = A[1];
EndIf
If (#B[] != 1)
  B = B[1];
EndIf

EA = A*rad / div;
EB = B*rad / div;
ea = 0;
eb = 0;


//
If (tag != 0)
  tag = tag - 1;
EndIf
tag = tag + news;
Printf("first tag= %f  (expected last tag: %f)", tag, tag+#eps[]-1);

po = OFFSET + newp;
Point(po) = {xo, yo, zo, 1.};

p = newp; Point(p) = {xo+1, yo, zo, 1};
pp = newp; Point(pp) = {zo, yo+1, zo, 1};
l = OFFSET + newl; Line(l) = {po, p};
lb = newl; Line(lb) = {p, pp};
lc = newl; Line(lc) = {pp, p};
ll = OFFSET + newll; Line Loop(ll) = {l, lb, lc};
//Ss = OFFSET + news; Ruled Surface(sS) = {ll};
Printf("(newp:%f) (newl:%f) (newll:%f) (news:None)", p, l, ll);

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
  perm = Sqrt(eps[ii]);
EndIf
If (ii == #eps[])
  perm = Sqrt(eps[ii-1]);
EndIf


lc = 2*Pi / (alpha * perm * k);


Point(p+2) = {xo+A*rad-ea, yo, zo, lc};
Point(p+3) = {xo, yo+B*rad-eb, zo, lc};

Point(p+4) = {xo-A*rad+ea, yo, zo, lc};
Point(p+5) = {xo, yo-B*rad+eb, zo, lc};

Point(p+6) = {xo, yo, zo-B*rad+eb, lc};
Point(p+7) = {xo, yo, zo+B*rad-eb, lc};

Ellipse(l+1) = {p+2, po, po, p+3};

Ellipse(l+2) = {p+3, po, po, p+4};

Ellipse(l+3) = {p+4, po, po, p+5};

Ellipse(l+4) = {p+5, po, po, p+2};

Ellipse(l+5) = {p+3, po, po, p+6};
Ellipse(l+6) = {p+6, po, po, p+5};
Ellipse(l+7) = {p+5, po, po, p+7};
Ellipse(l+8) = {p+7, po, po, p+3};

Ellipse(l+9) = {p+2, po, po, p+7};

Ellipse(l+10) = {p+7, po, po, p+4};

Ellipse(l+11) = {p+4, po, po, p+6};

Ellipse(l+12) = {p+6, po, po, p+2};

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

p = newp;
l = newl;
ll = newll;
sS = news;

Printf("(newp:%f) (newl:%f) (newll:%f) (news:%f)", p, l, ll, sS);
max = -1;
If (max < p)
  max = p;
EndIf
If (max < l)
  max = l;
EndIf
If (max < ll)
  max = ll;
EndIf
If (max < sS)
  max = sS;
EndIf
Printf("OFFSET = %f; // fix about Gmsh confusion", max) > "offset_tmp.geo";
