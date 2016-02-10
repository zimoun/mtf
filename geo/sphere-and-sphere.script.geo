alpha = 20 ;
Printf("%f", alpha);

eps = 3 ;

lambda = 9.5 ;
k = 2*Pi* Sqrt(eps) / lambda ;

rad1 = 1.;
rad2 = 1.;

L = 1 ;

//
lc = 2*Pi / (alpha * k);
a = 1;
s = L + rad1 + rad2;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {rad1, 0.0, 0.0, lc/a};
Point(3) = {0, rad1, 0.0, lc};

Circle(1) = {2, 1, 3};
Line(100) = {1, 3};

Point(4) = {-rad1, 0, 0.0, lc};
Point(5) = {0, -rad1, 0.0, lc};

Circle(2) = {3, 1, 4};

Circle(3) = {4, 1, 5};
Line(300) = {1, 5};

Circle(4) = {5, 1, 2};

Point(6) = {0, 0, -rad1, lc};
Point(7) = {0, 0, rad1, lc};
Circle(5) = {3, 1, 6};
Circle(6) = {6, 1, 5};
Circle(7) = {5, 1, 7};
Circle(8) = {7, 1, 3};

Circle(9) = {2, 1, 7};
Line(900) = {1, 7};

Circle(10) = {7, 1, 4};

Circle(11) = {4, 1, 6};
Line(1100) = {1, 6};

Circle(12) = {6, 1, 2};

Line Loop(13) = {2, 8, -10};
Ruled Surface(14) = {13};

// Line Loop(130) = {-100, 8, 900};
// Ruled Surface(140) = {130};
// Ruled Surface(141) = {-130};

Line Loop(15) = {10, 3, 7};
Ruled Surface(16) = {15};

// Line Loop(150) = {-900, 300, 7};
// Ruled Surface(160) = {150};
// Ruled Surface(161) = {-150};

Line Loop(17) = {-8, -9, 1};
Ruled Surface(18) = {17};

Line Loop(19) = {-11, -2, 5};
Ruled Surface(20) = {19};

// Line Loop(190) = {-1100, 100, 5};
// Ruled Surface(200) = {190};
// Ruled Surface(201) = {-190};

Line Loop(21) = {-5, -12, -1};
Ruled Surface(22) = {21};

Line Loop(23) = {-3, 11, 6};
Ruled Surface(24) = {23};

// Line Loop(230) = {-300, 1100, 6};
// Ruled Surface(240) = {230};
// Ruled Surface(241) = {-230};

Line Loop(25) = {-7, 4, 9};
Ruled Surface(26) = {25};

Line Loop(27) = {-4, 12, -6};
Ruled Surface(28) = {27};

Physical Surface(10) = {28, 26, 16, 14, 20, 24, 22, 18};

//// end surface


////////
// bis//
////////
p = newp;
l = newl;
ll = newll;
Ss = news;

Point(p+1) = {s+0.0, 0.0, 0.0, lc};
Point(p+2) = {s+rad2, 0.0, 0.0, lc/a};
Point(p+3) = {s+0, rad2, 0.0, lc};

Circle(l+1) = {p+2, p+1, p+3};
Line(l+100) = {p+1, p+3};

Point(p+4) = {s-rad2, 0, 0.0, lc};
Point(p+5) = {s+0, -rad2, 0.0, lc};

Circle(l+2) = {p+3, p+1, p+4};

Circle(l+3) = {p+4, p+1, p+5};
Line(l+300) = {p+1, p+5};

Circle(l+4) = {p+5, p+1, p+2};

Point(p+6) = {s+0, 0, -rad2, lc};
Point(p+7) = {s+0, 0, rad2, lc};
Circle(l+5) = {p+3, p+1, p+6};
Circle(l+6) = {p+6, p+1, p+5};
Circle(l+7) = {p+5, p+1, p+7};
Circle(l+8) = {p+7, p+1, p+3};

Circle(l+9) = {p+2, p+1, p+7};
Line(l+900) = {p+1, p+7};

Circle(l+10) = {p+7, p+1, p+4};

Circle(l+11) = {p+4, p+1, p+6};
Line(l+1100) = {p+1, p+6};

Circle(l+12) = {p+6, p+1, p+2};

Line Loop(ll+13) = {l+2, l+8, -l-10};
Ruled Surface(Ss + 14) = {ll+13};

// Line Loop(130) = {-100, 8, 900};
// Ruled Surface(140) = {130};
// Ruled Surface(141) = {-130};

Line Loop(ll+15) = {l+10, l+3, l+7};
Ruled Surface(Ss+16) = {ll+15};

// Line Loop(150) = {-900, 300, 7};
// Ruled Surface(160) = {150};
// Ruled Surface(161) = {-150};

Line Loop(ll+17) = {-l-8, -l-9, l+1};
Ruled Surface(Ss+18) = {ll+17};

Line Loop(ll+19) = {-l-11, -l-2, l+5};
Ruled Surface(Ss+20) = {ll+19};

// Line Loop(190) = {-1100, 100, 5};
// Ruled Surface(200) = {190};
// Ruled Surface(201) = {-190};

Line Loop(ll+21) = {-l-5, -l-12, -l-1};
Ruled Surface(Ss+22) = {ll+21};

Line Loop(ll+23) = {-l-3, l+11, l+6};
Ruled Surface(Ss+24) = {ll+23};

// Line Loop(230) = {-300, 1100, 6};
// Ruled Surface(240) = {230};
// Ruled Surface(241) = {-230};

Line Loop(ll+25) = {-l-7, l+4, l+9};
Ruled Surface(Ss+26) = {ll+25};

Line Loop(ll+27) = {-l-4, l+12, -l-6};
Ruled Surface(Ss+28) = {ll+27};


Physical Surface(100) = {ll+28, ll+26, ll+16, ll+14, ll+20, ll+24, ll+22, ll+18};



Mesh 2;
Save "sphere-sphere.msh" ;

// Delete Physicals;
// Physical Surface(10) = {28, 26, 16, 14, 20, 24, 22, 18};
// Save "sphere.msh" ;

// Delete Physicals;
// Physical Surface(20) = {1000, 2000, 3000, 4000, 5000, 6000};
// Save "cube.msh" ;
