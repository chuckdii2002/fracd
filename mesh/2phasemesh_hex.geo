//gmsh kathoff.geo -2
cl = 1;
lx = 6000.0;
ly = 6000.0;
lz = 0.;
ncell = 10;
res = lx/ncell;
xgrid = 11;
ygrid = 11;
Point(1) = {0, 0, lz, res};
Point(2) = {lx, 0, lz, res};
Point(3) = {lx, ly, lz, res};
Point(4) = {0, ly, lz, res};
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
Transfinite Line {1, 3} = xgrid Using Progression 1;
Transfinite Line {2, 4} = ygrid Using Progression 1;
Recombine Surface {1};

