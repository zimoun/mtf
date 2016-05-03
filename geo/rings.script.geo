
lc1 = 1;
rad1 = 1;

lc2 = 1;
rad2 = 1;

lc3 = 1;
rad3 = 1;


Function CreateCircle
  counter = -1;
  For ii In {0:#pts[]-2}
    l = newl; Circle(l) = {pts[ii], po, pts[ii+1]}; lines[counter++] = l;
  EndFor
  l = newl; Circle(l) = {pts[#pts[]-1], po, pts[0]}; lines[counter++] = l;

  ll = newll; Line Loop(ll) = {lines[]};
  ss = news; Plane Surface(ss) = {ll};
Return

Function CreateSquare
  counter = -1;
  For ii In {0:#pts[]-2}
    l = newl; Line(l) = {pts[ii], pts[ii+1]}; lines[counter++] = l;
  EndFor
  l = newl; Line(l) = {pts[#pts[]-1], pts[0]}; lines[counter++] = l;

  ll = newll; Line Loop(ll) = {lines[]};
  ss = news; Plane Surface(ss) = {ll};
Return


Function ArcExtrudeDonut
   ext[] = Extrude{ {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, Pi/2}{ Surface{surf}; };
   For ii In {2:#ext[]-1}
     donut[cdonut++] = ext[ii];
   EndFor
Return

Function ArcExtrudeSquare
   ext[] = Extrude{ {0, 0, 0}, {1, 0, 0}, {0, 0, 5}, Pi/2}{ Surface{surf}; };
   For ii In {2:#ext[]-1}
     square[csquare++] = ext[ii];
   EndFor
Return

Function TransExtrudeTwist
   ext[] = Extrude{ {-4, 0, 0}, {1, 0, 0}, {4, 0, 7}, Pi/4}{ Surface{surf}; };
   For ii In {2:#ext[]-1}
     twist[ctwist++] = ext[ii];
   EndFor
Return

////////////
// DONUT //
////////////

donut[] = {};
xo = -4; yo = 0; zo = 0;

counter = -1;
pts[] = {}; lines[] = {};
po = newp; Point(po) = {xo, yo, zo, lc1};

p = newp; Point(p) = {xo+rad1, yo, zo, lc1}; pts[counter++] = p;
p = newp; Point(p) = {xo, yo+rad1, zo, lc1}; pts[counter++] = p;
p = newp; Point(p) = {xo-rad1, yo, zo, lc1}; pts[counter++] = p;
p = newp; Point(p) = {xo, yo-rad1, zo, lc1}; pts[counter++] = p;

Call CreateCircle;

cdonut = -1;

surf = ss;
Call ArcExtrudeDonut;
// BUG gmsh!! does not allow For-loop with Function/Macro
surf = ext[0];
Call ArcExtrudeDonut;
surf = ext[0];
Call ArcExtrudeDonut;
surf = ext[0];
Call ArcExtrudeDonut;

Physical Surface(1) = {donut[]};

////////////
// Square //
////////////

square[] = {};
xo = 0; yo = 0; zo = 0;

counter = -1;
pts[] = {}; lines[] = {};
po = newp; Point(po) = {xo, yo, zo, lc1};

p = newp; Point(p) = {xo-rad2, yo, zo-rad2, lc2}; pts[counter++] = p;
p = newp; Point(p) = {xo+rad2, yo, zo-rad2, lc2}; pts[counter++] = p;
p = newp; Point(p) = {xo+rad2, yo, zo+rad2, lc2}; pts[counter++] = p;
p = newp; Point(p) = {xo-rad2, yo, zo+rad2, lc2}; pts[counter++] = p;

Call CreateSquare;

csquare = -1;

surf = ss;
Call ArcExtrudeSquare;
// BUG gmsh!! does not allow For-loop with Function/Macro
surf = ext[0];
Call ArcExtrudeSquare;
surf = ext[0];
Call ArcExtrudeSquare;
surf = ext[0];
Call ArcExtrudeSquare;

Physical Surface(2) = {-square[]};

////////////
// Twist  //
////////////

twist[] = {};
xo = 6; yo = 0; zo = 7;

counter = -1;
pts[] = {}; lines[] = {};
po = newp; Point(po) = {xo, yo, zo, lc1};

p = newp; Point(p) = {xo, yo-rad3, zo-rad3, lc3}; pts[counter++] = p;
p = newp; Point(p) = {xo, yo+rad3, zo-rad3, lc3}; pts[counter++] = p;
p = newp; Point(p) = {xo, yo+rad3, zo+rad3, lc3}; pts[counter++] = p;
p = newp; Point(p) = {xo, yo-rad3, zo+rad3, lc3}; pts[counter++] = p;

Call CreateSquare;

ctwist = -1;

twist[ctwist++] = -ss;

surf = ss;
Call TransExtrudeTwist;
// BUG gmsh!! does not allow For-loop with Function/Macro
surf = ext[0];
Call TransExtrudeTwist;
surf = ext[0];
Call TransExtrudeTwist;
twist[ctwist++] = ext[0];

Physical Surface(3) = {-twist[]};

Mesh 2;
Save "rings.msh";
