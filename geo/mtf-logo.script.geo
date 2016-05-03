Include "params_tmp.geo";

tag = 0;

perm1 =  Sqrt(eps[0]);
lc1 = 2*Pi / (alpha * perm1 * k);

perm2 =  Sqrt(eps[1]);
lc2 = 2*Pi / (alpha * perm2 * k);

perm3 =  Sqrt(eps[2]);
lc3 = 2*Pi / (alpha * perm3 * k);

Lx = 0; Ly = 0; Lz = -0.5;
vx = 0; vy = 0; vz = 1;
ax = 0; ay = 1; az = 0;
thetaRAD = 0.15;

Function MyExtrude

count = -1;
lines[] = {};

For ii In {0:#pts[]-2}
    l = newl; Line(l) = {pts[ii], pts[ii+1]};
    lines[count++] = l;
EndFor
l = newl; Line(l) = {pts[#pts[]-1], pts[0]};
lines[count++] = l;

ll = newll; Line Loop(ll) = {lines[]};
ss = news; Plane Surface(ss) = {-ll};

//ext[] = Extrude{0, 0, 0.5}{Surface{ss};};
ext[] = Extrude{ {Lx, Ly, Lz}, {vx, vy, vz}, {ax, ay, az}, thetaRAD}{Surface{ss};};

jj = 0;
tmp[] = {ext[0]};
For ii In {1:#ext[]-1}
  tmp[jj++] = ext[ii];
EndFor

Physical Surface(tag++) = {ss, -tmp[]};

Return


/// M

count = -1;
pts[] = {};
p = newp; Point(p) = {0, 0, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0, 1, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.15, 1, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.45, 0.5, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.55, 0.5, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.85, 1, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {1, 1, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {1, 0, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.75, 0, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.75, 0.3, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.25, 0.3, 0, lc1}; pts[count++] = p;
p = newp; Point(p) = {0.25, 0., 0, lc1}; pts[count++] = p;

Lx = 0; Ly = 0; Lz = -0.5;
vx = 0; vy = 0; vz = 1;
ax = 0; ay = 0; az = 0;
thetaRAD = 0.25;

Call MyExtrude;

/// T

x = 2;

count = -1;
pts[] = {};
p = newp; Point(p) = {x-0.1, 0, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x-0.1, 0.75, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x-0.75, 0.75, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x-0.75, 1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.75, 1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.75, 0.75, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.1, 0.75, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.1, 0, 0., lc2}; pts[count++] = p;

Rotate{ {0, 1, 0}, {x, 0.0, 0}, -0.2}{ Point{pts[]}; }

Lx = 0; Ly = 0; Lz = -0.5;
vx = 0; vy = 0; vz = 1;
ax = x; ay = 0; az = 0;
thetaRAD = -0.25;

Call MyExtrude;

/// F

x = 3.25;

count = -1;
pts[] = {};
p = newp; Point(p) = {x, -0.3, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x, 1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+1, 1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+1, 0.55, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.35, 0.80, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.35, 0.30, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+1, 0.35, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+1, 0.1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.3, 0.1, 0., lc2}; pts[count++] = p;
p = newp; Point(p) = {x+0.3, -0.5, 0., lc2}; pts[count++] = p;

Rotate{ {0, 1, 0}, {x, 0.0, 0}, 0.2}{ Point{pts[]}; }

Lx = 0; Ly = 0; Lz = 0.3;
vx = 0; vy = 0; vz = 1;
ax = x; ay = 0; az = 0;
thetaRAD = 0.25;

Call MyExtrude;


Mesh 2;
Save "mtf-logo.msh";
