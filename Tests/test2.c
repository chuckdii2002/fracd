/*
 test2.c:
 (c) 2016--- Chukwudi Chukwudozie chdozie@gmail.com
 ./test2 -options_file test2_2da.opts -T_snes_type test  -T_snes_view -T_snes_test_display -T_snes_view -T_snes_linesearch_monitor -T_snes_monitor
 
 mpiexec -np 5 ./test2 -options_file test2_2da.opts
 mpiexec -np 5 ./test2 -options_file test2_2db.opts
 mpiexec -np 5 ./test2 -options_file test2_2dc.opts
 mpiexec -np 5 ./test2 -options_file test2_3da.opts
 mpiexec -np 5 ./test2 -options_file test2_3db.opts
 mpiexec -np 5 ./test2 -options_file test2_3dc.opts
 
 mpiexec -np 5 ./test2 -options_file test2_2da_quad.opts
 mpiexec -np 5 ./test2 -options_file test2_2db_quad.opts
 mpiexec -np 5 ./test2 -options_file test2_2dc_quad.opts
 
 mpiexec -np 5 ./test2 -options_file test2_3da_hex.opts
 mpiexec -np 5 ./test2 -options_file test2_3db_hex.opts
 mpiexec -np 5 ./test2 -options_file test2_3dc_hex.opts
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
  ierr = VecSet(bag.fields.QT,1);CHKERRQ(ierr);
    ierr = VecSet(bag.fields.V,0.5);CHKERRQ(ierr);
    ierr = VecSet(bag.fields.V,0.);CHKERRQ(ierr);

    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag.plexVecNode, 2, &eStart, &eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag.plexVecNode, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    
    printf("cells: %i \t %i \n", cStart,cEnd);
    printf("faces: %i \t %i \n", fStart,fEnd);
    printf("edges: %i \t %i \n", eStart,eEnd);
    printf("vertices: %i \t %i \n\n\n\n", vStart,vEnd);
    
    PetscReal b;
    b = acos(0.5);
    PetscViewer viewer;
 
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Volume.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.ppties.dualCellVolume,viewer);CHKERRQ(ierr);
    
    bag.timevalue = 1;
    bag.maxtimestep = 5;

    
    ierr = FracDSolveT(&bag);CHKERRQ(ierr);
    ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);

    const char* fna = "output2a.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fna,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.T,viewer);
    PetscViewerDestroy(&viewer);
    
//    ierr = FracDSolveT(&bag);CHKERRQ(ierr);
//    ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);
//    const char* fnb = "output2b.vtk";
//    PetscViewerVTKOpen(PETSC_COMM_WORLD,fnb,FILE_MODE_WRITE,&viewer);
//    VecView(bag.fields.T,viewer);
//    PetscViewerDestroy(&viewer);
//    
//    ierr = FracDSolveT(&bag);CHKERRQ(ierr);
//    ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);
//    
//    const char* fn = "output2.vtk";
//    PetscViewerVTKOpen(PETSC_COMM_WORLD,fn,FILE_MODE_WRITE,&viewer);
//    VecView(bag.fields.T,viewer);
//    PetscViewerDestroy(&viewer);
    
  ierr = FracDFinalize(&bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(0);
}

