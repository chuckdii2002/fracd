/*
 test1.c:
 
 (c) 2016--- Chukwudi Chukwudozie chdozie@gmail.com
 
 ./test1 -meshfilename trial.msh -dim 2 -meshrefine -meshrefinetype SIZECONSTRAINED -meshinterpolate  -simplex
 ./test1  -dim 2 -meshrefine -meshinterpolate -grid_size 3,3 -meshrefinetype SIZECONSTRAINED
 ./test1  -dim 2 -meshrefine -meshrefinetype SIZECONSTRAINED -meshfilename trial.msh
 
 ./test1  -dim 2 -meshfilename trial.msh -ubc -numubclabels 1 -UBClabels 1 -pbc -numpbclabels 1 -pBClabels 1 -Tbc -numTbclabels 1 -TBClabels 1 -Tractionbc -numTractionbclabels 1 -TractionBClabels 1 -flowfluxbc -numflowfluxbclabels 1 -flowfluxBClabels 1 -heatfluxbc -numheatfluxbclabels 1 -heatfluxBClabels 1
 
 
 
 mpiexec -np 2 ./test1  -dim 2 -meshfilename trial.msh -ubc -numubclabels 1 -UBClabels 1 -pbc -numpbclabels 1 -pBClabels 1 -Tbc -numTbclabels 1 -TBClabels 1 -Tractionbc -numTractionbclabels 1 -TractionBClabels 1 -flowfluxbc -numflowfluxbclabels 1 -flowfluxBClabels 1 -heatfluxbc -numheatfluxbclabels 1 -heatfluxBClabels 1 -UbcComponentsperlabel 2 -UbcComponents 0,1 -Ubcvalues 0,0 -tractionbcComponentsperlabel 2 -tractionBCComponents 0,1 -tractionBCvalues 0,-0.2 -PBCValues 4 -FlowfluxBCValues -2 -TBCValues 56 -HeatfluxBCValues 3.4
 
 
 
 ./test1  -dim 2 -meshfilename trial.msh -ubc -numubclabels 1 -UBClabels 1 -pbc -numpbclabels 1 -pBClabels 1 -Tbc -numTbclabels 2 -TBClabels 1,2 -Tractionbc -numTractionbclabels 1 -TractionBClabels 1 -flowfluxbc -numflowfluxbclabels 1 -flowfluxBClabels 1 -heatfluxbc -numheatfluxbclabels 1 -heatfluxBClabels 1 -UbcComponentsperlabel 2 -UbcComponents 0,1 -Ubcvalues 0,0 -tractionbcComponentsperlabel 2 -tractionBCComponents 0,1 -tractionBCvalues 0,-0.2 -PBCValues 4 -FlowfluxBCValues -2 -TBCValues 56,45 -HeatfluxBCValues 3.4
 
 
 ./test1  -dim 2 -meshfilename how.msh -ubc -numubclabels 1 -UBClabels 1 -UbcComponentsperlabel 2 -UbcComponents 0,1 -Ubcvalues 0,0 -pbc -numpbclabels 1 -pBClabels 1 -PBCValues 4 -Tbc -numTbclabels 2 -TBClabels 1,2 -TBCValues 56,45 -Tractionbc -numTractionbclabels 1 -TractionBClabels 1 -tractionbcComponentsperlabel 2 -tractionBCComponents 0,1 -tractionBCvalues 0,-0.2  -flowfluxbc -numflowfluxbclabels 1 -flowfluxBClabels 1 -FlowfluxBCValues -2 -heatfluxbc -numheatfluxbclabels 1 -heatfluxBClabels 1 -HeatfluxBCValues 3.4
 
 
 ./test1  -dim 2 -meshfilename how.msh -ubc -numubclabels 1 -UBClabels 1 -UbcComponentsperlabel 2 -UbcComponents 0,1 -Ubcvalues 0,0 -Tractionbc -numTractionbclabels 1 -TractionBClabels 1 -tractionbcComponentsperlabel 2 -tractionBCComponents 0,1 -tractionBCvalues 0,-0.2
 
 ./test1 -options_file test1_2da.opts
 ./test1 -options_file test1_2db.opts
 ./test1 -options_file test1_2da_quad.opts
 ./test1 -options_file test1_2db_quad.opts
 ./test1 -options_file test1_3da.opts
 ./test1 -options_file test1_3da_hex.opts
 ./test1 -options_file test1_3db_hex.opts
 mpiexec -np 2 ./test1 -options_file test1_3db.opts
 */

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDMechanics.h"
#include "FracDHeatFlow.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
    AppCtx          bag;
    PetscErrorCode  ierr;
    PetscInt            vStart, vEnd,eStart, eEnd,fStart, fEnd, cStart, cEnd;
    
    ierr = PetscInitialize(&argc,&argv,(char*)0,banner);CHKERRQ(ierr);
    ierr = FracDInitialize(&bag);CHKERRQ(ierr);
    ierr = FracDSolveU(&bag);CHKERRQ(ierr);
    
    
    PetscReal b;
    b = acos(0.5);
    printf("pi = %g, sin %g= %g, %g\n\n\n",PETSC_PI, b, sin(b),360*b/(2*PETSC_PI));
    PetscViewer viewer;
    const char* fn = "output.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fn,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.U,viewer);
    //    VecView(bag.ppties.E,viewer);
    PetscViewerDestroy(&viewer);
    
    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 2, &eStart, &eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag.plexVecNode, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    
    printf("cells: %i \t %i \n", cStart,cEnd);
    printf("faces: %i \t %i \n", fStart,fEnd);
    printf("edges: %i \t %i \n", eStart,eEnd);
    printf("vertices: %i \t %i \n\n\n\n", vStart,vEnd);
    
    //    ierr = DMView(bag.plexVecNode, viewer);CHKERRQ(ierr);
    
    
    
    ierr = FracDFinalize(&bag);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return(0);
}

