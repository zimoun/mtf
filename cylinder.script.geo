alpha = 20 ;
Printf("%f", alpha);

eps = 1.5 ;
lambda = 20. ;

rad = 1.3;

L = 1.;


///
k = 2*Pi* eps*eps / lambda ;
lc = 2*Pi / (alpha * k);

///
centerA=newp;
lA = newl;
llA = newll;
sA = news;

Point(centerA) = {0, 0, 0, lc};

Point(centerA+1) = {0, rad, 0, lc};
Point(centerA+2) = {0, 0, rad, lc};
Point(centerA+3) = {0, -rad, 0, lc};
Point(centerA+4) = {0, 0, -rad, lc};

For ii In {1:4}
  jj = Fmod(ii, 4);
  Circle(lA+ii) = {centerA+ii, centerA, centerA+jj+1};
EndFor

Line Loop(llA) = {-(lA+1), -(lA+2), -(lA+3), -(lA+4)};

Ruled Surface(sA) = {llA};

centerB=newp;
lB = newl;
llB = newll;
sB = news;

Point(centerB) = {L, 0, 0, lc};

Point(centerB+1) = {L, rad, 0, lc};
Point(centerB+2) = {L, 0, rad, lc};
Point(centerB+3) = {L, -rad, 0, lc};
Point(centerB+4) = {L, 0, -rad, lc};

For ii In {1:4}
  jj = Fmod(ii, 4);
  Circle(lB+ii) = {centerB+ii, centerB, centerB+jj+1};
EndFor
Line Loop(llB) = {lB+1, lB+2, lB+3, lB+4};
 
Ruled Surface(sB) = {llB};

l = newl;
For ii In {1:4}
  Line(l+ii) = {centerA+ii, centerB+ii};
EndFor

ll = newll;
sS = news;
For ii In {1:4}
  jj = Fmod(ii, 4);
  Line Loop(ll+ii) = {l+jj+1, -(lB+ii), -(l+ii), lA+ii};
EndFor

For ii In {1:4}
  Ruled Surface(sS+ii) = {ll+ii};
EndFor

Mesh 2;
Delete Physicals;
Physical Surface(10) = {sA, sB, sS+1, sS+2, sS+3, sS+4};
Save "cylinder-simple.msh";

