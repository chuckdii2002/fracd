/*
 test3.c:
 (c) 2016--- Chukwudi Chukwudozie chdozie@gmail.com
 ./test3 -options_file test3_2da.opts -p_snes_view
 
 mpiexec -np 5 ./test3 -options_file test3_2da.opts -options_file test3_2dapvt.opts
 mpiexec -np 5 ./test3 -options_file test3_3da.opts -options_file test3_2dapvt.opts
 mpiexec -np 1 ./test3 -options_file test3_2phase_2da.opts -options_file test3_2phase_2dapvt.opts
 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2da.opts -options_file test3_2phase_2dapvt.opts -p_snes_view -p_snes_monitor  -p_snes_converged_reason -p_ksp_monitor_true_residual -p_ksp_converged_reason -p_fieldsplit_0_ksp_type fgmres -p_fieldsplit_1_ksp_type fgmres -P_fieldsplit_0_sub_ksp_type gmres -P_fieldsplit_1_sub_ksp_type gmres -P_fieldsplit_0_sub_pc_type bjacobi -P_fieldsplit_1_sub_pc_type bjacobi -p_snes_max_it 35 -P_fieldsplit_0_sub_sub_ksp_type fgmres -P_fieldsplit_1_sub_sub_ksp_type fgmres -P_fieldsplit_0_sub_sub_pc_type bjacobi -P_fieldsplit_1_sub_sub_pc_type bjacobi -p_snes_type newtonls -p_fieldsplit_2_ksp_type fgmres  -P_fieldsplit_2_pc_type bjacobi
 */

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDMechanics.h"
#include "FracDHeatFlow.h"
#include "FracDFluidFlow.h"
#include "FracDComputations.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
    AppCtx          bag;
    PetscErrorCode  ierr;
    PetscViewer viewer;
    PetscInt            rank,size,i;
    
    ierr = PetscInitialize(&argc,&argv,(char*)0,banner);CHKERRQ(ierr);
    
    
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
    
    
    ierr = FracDInitialize(&bag);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold.txt",&viewer);CHKERRQ(ierr);
    VecView(bag.fields.oP,viewer);
    
    ierr = VecSet(bag.fields.QP,0.);CHKERRQ(ierr);
    
    for(i = 0; i < 300; i++){
        ierr = FracDSolveP(&bag);CHKERRQ(ierr);
        ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold1.txt",&viewer);CHKERRQ(ierr);
        VecView(bag.fields.P,viewer);
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold3.txt",&viewer);CHKERRQ(ierr);
        VecView(bag.fields.Sw,viewer);
        
        
        
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PResvolume.txt",&viewer);CHKERRQ(ierr);
        ierr = VecView(bag.ppties.dualCellVolume,viewer);CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PFlowRatew.txt",&viewer);CHKERRQ(ierr);
        ierr = VecView(bag.fields.Qwbh,viewer);CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PFlowRateo.txt",&viewer);CHKERRQ(ierr);
        ierr = VecView(bag.fields.Qobh,viewer);CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PFlowPressure.txt",&viewer);CHKERRQ(ierr);
        ierr = VecView(bag.fields.Pbh,viewer);CHKERRQ(ierr);
        
        
        
        
        
        const char* fna = "output3.vtk";
        PetscViewerVTKOpen(PETSC_COMM_WORLD,fna,FILE_MODE_WRITE,&viewer);
        VecView(bag.fields.P,viewer);
        PetscViewerDestroy(&viewer);
        
        const char* fna_1 = "output3_1.vtk";
        PetscViewerVTKOpen(PETSC_COMM_WORLD,fna_1,FILE_MODE_WRITE,&viewer);
        VecView(bag.fields.Sw,viewer);
        PetscViewerDestroy(&viewer);
    }
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Swnew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.Sw,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pnew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.P,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold2.txt",&viewer);CHKERRQ(ierr);
    VecView(bag.fields.oP,viewer);
    
    printf("\n rank: %d/%d \t NEW TIME STEP \n\n\n\n",rank,size);
    
    ierr = FracDSolveP(&bag);CHKERRQ(ierr);
    ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);
    
    const char* fna1 = "output3a.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fna1,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.P,viewer);
    PetscViewerDestroy(&viewer);
    
    const char* fna1_1 = "output3a_1.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fna1_1,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.Sw,viewer);
    PetscViewerDestroy(&viewer);
    
    
    //    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PResvolume.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.ppties.dualCellVolume,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PFlowRate.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.Qwbh,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PFlowPressure.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.Pbh,viewer);CHKERRQ(ierr);
    
    ierr = FracDFinalize(&bag);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return(0);
}

