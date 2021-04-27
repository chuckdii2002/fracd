xdim = 6000;
ydim = 6000;
zdim = 600;
gridsize = 0.5;
xgrids = 21;
ygrids = 21;
zlayers = 5;

Point(1) = {0, 0, 0, gridsize};
Point(2) = {xdim, 0, 0, gridsize};
Point(3) = {xdim, ydim, 0, gridsize};
Point(4) = {0, ydim, 0, gridsize};

Line(5) = {1, 2};
Line(6) = {2, 3};
Line(7) = {3, 4};
Line(8) = {4, 1};
Line Loop(9) = {5, 6, 7, 8};
Plane Surface(10) = {9};

Transfinite Line{5,7} = xgrids;
Transfinite Line{6,8} = ygrids;
Transfinite Surface{10};
Recombine Surface{10};


//Now make 3D by extrusion.
newEntities[] = 
Extrude { 0,0, -zdim }
{
	Surface{10};
	Layers{zlayers};
	Recombine;
};
Physical Surface(4) = {10};
Physical Surface(3) = {newEntities[0]};
Physical Surface(5) = {newEntities[2]};
Physical Surface(6) = {newEntities[4]};
Physical Surface(1) = {newEntities[3]};
Physical Surface(2) = {newEntities[5]};
Physical Volume(100) = {newEntities[1]};