cl = 1;
Point(1) = {0, 0, 0, 1.0};
Point(2) = {10, 0, 0, 1.0};
Point(3) = {10, 10, 0, 1.0};
Point(4) = {0, 10, 0, 1.0};
Line(1) = {4, 3};
Line(2) = {3, 2};
Line(3) = {2, 1};
Line(4) = {1, 4};
Line Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
Physical Line(1) = {4};
Physical Line(2) = {1};
Physical Line(3) = {2};
Physical Line(4) = {3};
Physical Surface(1) = {1};
Transfinite Surface {1} = {1, 2, 3, 4};
Transfinite Line {1, 3} = 11 Using Progression 1;
Transfinite Line {2, 4} = 11 Using Progression 1;
Recombine Surface {1};

