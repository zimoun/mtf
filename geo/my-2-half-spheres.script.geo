
alpha = 10 ;
Printf("%f", alpha);

eps = 1. ;
lambda = 10.5 ;
k = 2*Pi* Sqrt(eps) / lambda ;

//

rad = 1.0;
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
Ruled Surface(1400) = {-13};

Line Loop(130) = {-100, 8, 900};
Ruled Surface(140) = {130};
Ruled Surface(141) = {-130};

Line Loop(15) = {10, 3, 7};
Ruled Surface(16) = {15};
Ruled Surface(1600) = {-15};

Line Loop(150) = {-900, 300, 7};
Ruled Surface(160) = {150};
Ruled Surface(161) = {-150};

Line Loop(17) = {-8, -9, 1};
Ruled Surface(18) = {17};
Ruled Surface(1800) = {-17};

Line Loop(19) = {-11, -2, 5};
Ruled Surface(20) = {19};
Ruled Surface(2000) = {-19};

Line Loop(190) = {-1100, 100, 5};
Ruled Surface(200) = {190};
Ruled Surface(201) = {-190};

Line Loop(21) = {-5, -12, -1};
Ruled Surface(22) = {21};
Ruled Surface(2200) = {-21};

Line Loop(23) = {-3, 11, 6};
Ruled Surface(24) = {23};
Ruled Surface(2400) = {-23};

Line Loop(230) = {-300, 1100, 6};
Ruled Surface(240) = {230};
Ruled Surface(241) = {-230};

Line Loop(25) = {-7, 4, 9};
Ruled Surface(26) = {25};
Ruled Surface(2600) = {-25};

Line Loop(27) = {-4, 12, -6};
Ruled Surface(28) = {27};
Ruled Surface(2800) = {-27};


Physical Surface(10) = {2800, 2600, 1600, 1400, 2000, 2400, 2200, 1800};
Physical Surface(1) = {241, 161, 16, 14, 20, 24, 201, 141};
Physical Surface(2) = {28, 26, 160, 140, 200, 240, 22, 18};

Mesh 2;
Save "full-oriented.msh" ;

Delete Physicals;
Physical Surface(21) = {240, 160, 200, 140};
Physical Surface(110) = {16, 14, 20, 24};
Physical Surface(210) = {28, 26, 22, 18};
Save "full-interfaces.msh" ;


Delete Physicals;
Physical Surface(10) = {2800, 2600, 1600, 1400, 2000, 2400, 2200, 1800};
Save "0.msh" ;

Delete Physicals;
Physical Surface(1) = {241, 161, 16, 14, 20, 24, 201, 141};
Save "1.msh" ;

Delete Physicals;
Physical Surface(2) = {28, 26, 160, 140, 200, 240, 22, 18};
Save "2.msh" ;
