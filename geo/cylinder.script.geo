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

//
If (tag != 0)
  tag = tag - 1;
EndIf
tag = tag + news;
Printf("first tag= %f  (expected last tag: %f)", tag, tag+#eps[]-1);

po = OFFSET + newp;
Point(po) = {xo+42, yo, zo-42, 1.};

p = newp; Point(p) = {xo+42, yo, zo, 1};
pp = newp; Point(pp) = {zo, yo+42, zo, 1};
l = OFFSET + newl; Line(l) = {po, p};
lb = newl; Line(lb) = {p, pp};
lc = newl; Line(lc) = {pp, p};
ll = OFFSET + newll; Line Loop(ll) = {l, lb, lc};
//Ss = OFFSET + news; Ruled Surface(sS) = {ll};
Printf("(newp:%f) (newl:%f) (newll:%f) (news:None)", p, l, ll);

rad = rad[0];

perm = Sqrt(eps[0]);
lc = 2*Pi / (alpha * perm * k);

Printf("k= %f  ,   alpha= %f , eps=%f [perm=%f]", k, alpha, eps[0], perm);

///
centerA=newp;
lA = newl;
llA = newll;
sA = news;

Point(centerA) = {xo, yo, zo, lc};

Point(centerA+1) = {xo, yo+rad, zo, lc};
Point(centerA+2) = {xo, yo, yo+rad, lc};
Point(centerA+3) = {xo, yo-rad, zo, lc};
Point(centerA+4) = {xo, yo, yo-rad, lc};

For ii In {1:4}
  jj = Fmod(ii, 4);
  Circle(lA+ii) = {centerA+ii, centerA, centerA+jj+1};
EndFor

Line Loop(llA) = {-(lA+1), -(lA+2), -(lA+3), -(lA+4)};

Ruled Surface(sA) = {-llA};

out[] = Extrude{Lc,0,0}{ Surface{sA}; };

jj = 0;
cyl[] = {out[0]};
For ii In {2:#out[]-1}
  cyl[jj++] = out[ii];
EndFor

Physical Surface(tag) = {-llA, cyl[]};


Mesh 2;
Delete Physicals;
Physical Surface(tag) = {-sA, cyl[]};
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
