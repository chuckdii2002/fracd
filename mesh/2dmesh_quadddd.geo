Point(1) = {0, 0, 0};
Point(2) = {0, 1, 0};
Point(3) = {1, 1, 0};
Point(4) = {1, 0, 0};
Point(5) = {.5, .2, 0};
Point(6) = {.2, .5, 0};
Point(7) = {.7, .5, 0};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Line(5) = {5, 7};
Line(6) = {7, 6};
Line(7) = {6, 5};
Line Loop(8) = {2, 3, 4, 1};
Line Loop(9) = {6, 7, 5};
Plane Surface(10) = {8, 9};
Physical Line("Dirichlet") = {2,3,4,1};
Physical Line("Neumann") = {6, 7, 5};
Physical Surface("Allsurface") = {10};
