alpha = 80 ;
Printf("%f", alpha);

eps = 1.5 ;
lambda = 10. ;
k = 2*Pi* eps / lambda ;

rad = 1.;
lc = 2*Pi / (alpha * k);

a = 1;

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {rad, 0.0, 0.0, lc/a};
Point(3) = {0, rad, 0.0, lc};

Circle(1) = {2, 1, 3};
Line(100) = {1, 3};

Point(4) = {-rad, 0, 0.0, lc};
Point(5) = {0, -rad, 0.0, lc};

Circle(2) = {3, 1, 4};

Circle(3) = {4, 1, 5};
Line(300) = {1, 5};

Circle(4) = {5, 1, 2};

Point(6) = {0, 0, -rad, lc};
Point(7) = {0, 0, rad, lc};
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

s = rad+rad;

x = 1;
y = 1;
z = 1;

alpha = 1;

Point(1000) = {s+0, 0, 0, lc};
Point(2000) = {s+0, 0, z, lc};
Point(3000) = {s+x, 0, z, lc};
Point(4000) = {s+x, 0, 0, lc};

Point(5000) = {s+0, -y, 0, lc};
Point(6000) = {s+0, -y, z, lc};
Point(7000) = {s+x, -y, z, lc};
Point(8000) = {s+x, -y, 0, lc};

// interface
Line(1000) = {1000, 2000};
Line(2000) = {2000, 3000};
Line(3000) = {3000, 4000};
Line(4000) = {4000, 1000};
Line Loop(1000) = {1000, 2000, 3000, 4000};
Plane Surface(1000) = {1000}; 

// opposite to interface (left)
Line(5000) = {5000, 8000};
Line(6000) = {8000, 7000};
Line(7000) = {7000, 6000};
Line(8000) = {6000, 5000};
Line Loop(2000) = {5000, 6000, 7000, 8000};
Plane Surface(2000) = {2000};

// top (left)
Line(9000) = {2000, 6000};
Line(10000) = {6000, 7000};
Line(11000) = {7000, 3000};
Line(12000) = {3000, 2000};
Line Loop(3000) = {9000, 10000, 11000, 12000};
Plane Surface(3000) = {3000};

// bottom (left)
Line(13000) = {1000, 4000};
Line(14000) = {4000, 8000};
Line(15000) = {8000, 5000};
Line(16000) = {5000, 1000};
Line Loop(4000) = {13000, 14000, 15000, 16000};
Plane Surface(4000) = {4000};

// front (left: x>0)
Line(17000) = {4000, 3000};
Line(18000) = {3000, 7000};
Line(19000) = {7000, 8000};
Line(20000) = {8000, 4000};
Line Loop(5000) = {17000, 18000, 19000, 20000};
Plane Surface(5000) = {5000};

// opposite front (x=0000)
Line(21000) = {1000, 5000};
Line(22000) = {5000, 6000};
Line(23000) = {6000, 2000};
Line(24000) = {2000, 1000};
Line Loop(6000) = {21000, 22000, 23000, 24000};
Plane Surface(6000) = {6000};



Physical Surface(20) = {1000, 2000, 3000, 4000, 5000, 6000};

// end cube


Mesh 2;
Save "sphere-cube.msh" ;

Delete Physicals;
Physical Surface(10) = {28, 26, 16, 14, 20, 24, 22, 18};
Save "sphere.msh" ;

Delete Physicals;
Physical Surface(20) = {1000, 2000, 3000, 4000, 5000, 6000};
Save "cube.msh" ;
