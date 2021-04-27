/*
 test3.c:
 (c) 2016--- Chukwudi Chukwudozie chdozie@gmail.com
 ./test3 -options_file test3_2da.opts -p_snes_view
 
 mpiexec -np 5 ./test3 -options_file test3_2da.opts -options_file test3_2dapvt.opts
 mpiexec -np 5 ./test3 -options_file test3_3da.opts -options_file test3_2dapvt.opts
 mpiexec -np 1 ./test3 -options_file test3_2phase_2da.opts -options_file test3_2phase_2dapvt.opts
 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2da.opts -options_file test3_2phase_2dapvt.opts -p_snes_view -p_snes_monitor  -p_snes_converged_reason -p_ksp_monitor_true_residual -p_ksp_converged_reason -p_fieldsplit_0_ksp_type fgmres -p_fieldsplit_1_ksp_type fgmres -P_fieldsplit_0_sub_ksp_type gmres -P_fieldsplit_1_sub_ksp_type gmres -P_fieldsplit_0_sub_pc_type bjacobi -P_fieldsplit_1_sub_pc_type bjacobi -p_snes_max_it 35 -P_fieldsplit_0_sub_sub_ksp_type fgmres -P_fieldsplit_1_sub_sub_ksp_type fgmres -P_fieldsplit_0_sub_sub_pc_type bjacobi -P_fieldsplit_1_sub_sub_pc_type bjacobi -p_snes_type newtonls -p_fieldsplit_2_ksp_type fgmres  -P_fieldsplit_2_pc_type bjacobi
 
 
mpiexec -np 2 ./test3 -options_file test3_2phase_2dd.opts -options_file test3_2phase_2dapvt.opts -p_snes_view -p_snes_monitor  -p_snes_converged_reason -p_ksp_monitor_true_residual -p_ksp_converged_reason -p_ksp_type fgmres  -p_pc_fieldsplit_type multiplicative -P_fieldsplit_0_pc_type bjacobi -P_fieldsplit_1_pc_type bjacobi -P_fieldsplit_2_pc_type bjacobi
 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2dd.opts -options_file test3_2phase_2dapvt.opts -p_snes_view -p_snes_monitor  -p_snes_converged_reason -p_ksp_monitor_true_residual -p_ksp_converged_reason -p_ksp_type gcr  -p_pc_fieldsplit_type multiplicative -P_fieldsplit_0_pc_type bjacobi -P_fieldsplit_1_pc_type bjacobi -P_fieldsplit_2_pc_type bjacobi -p_snes_linesearch_type basic
 
 
 
 
 
 
 
 
 
 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt.opts -p_snes_monitor -p_snes_type newtonls -p_snes_rtol 1e-5 -p_snes_atol 1e-5 -p_snes_stol 1e-5 -p_ksp_converged_reason -p_snes_linesearch_alpha 0.8 -p_snes_max_it 100
 

 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2dd.opts -options_file test3_2phase_2dapvt.opts
 mpiexec -np 2 ./test3 -options_file test3_2phase_2dd.opts -options_file test3_2phase_2dapvt1.opts
 mpiexec -np 2 ./test3 -options_file test3_3phase.opts -options_file test3_3phase_pvt.opts
 
 
 ./test3 -options_file test3_2da_quad.opts
 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt.opts -p_ksp_type pipegcr -p_snes_rtol 1e-5 -p_snes_atol 1e-5 -p_ksp_converged_reason -p_snes_monitor
 
 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt.opts -p_snes_rtol 1e-5 -p_snes_atol 1e-5 -p_snes_stol 1e-5 -p_snes_linesearch_alpha 0.8 -p_snes_max_it 100
 
 mpiexec -np 2 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt1.opts
 mpiexec -np 2 ./test3 -options_file test3_3phase_quad.opts -options_file test3_3phase_pvt.opts
 
 ./test3 -options_file test3_3phase_quad.opts -options_file test3_3phase_pvt.opts -p_snes_rtol 1e-5 -p_snes_atol 1e-5 -p_snes_stol 1e-5 -p_snes_linesearch_alpha 0.8 -p_snes_max_it 100
 
 
 ./test3 -options_file SPE.opts -options_file SPE_pvt.opts
 
 
 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt1.opts -p_fieldsplit_0_ksp_type fgmres -p_fieldsplit_1_ksp_type fgmres -p_fieldsplit_2_ksp_type fgmres -P_fieldsplit_2_ksp_monitor_true_residual -P_fieldsplit_0_pc_type jacobi -P_fieldsplit_1_pc_type jacobi -P_fieldsplit_2_pc_type lu
 
 
 
 ./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt1.opts  -p_snes_monitor  -p_snes_linesearch_type l2 -p_ksp_monitor -p_fieldsplit_ksp_type fgmres
 
 
 
./test3 -options_file test3_2phase_2dd_quad.opts -options_file test3_2phase_2dapvt1.opts  -p_snes_monitor  -p_snes_linesearch_type l2  -p_fieldsplit_ksp_type fgmres -p_fieldsplit_pc_type ilu -p_ksp_max_it 20000 -p_snes_lag_jacobian 100 -p_snes_max_it 200
 
 
  ./test3 -options_file test3_3phase_quad.opts -options_file test3_3phase_pvt.opts -p_fieldsplit_ksp_type fgmres
  ./test3 -options_file test3_3phase_quad.opts -options_file test3_3phase_pvt1.opts -p_fieldsplit_ksp_type fgmres
 

 https://epubs.siam.org/doi/pdf/10.1137/17M1133208
 
 https://www.mcs.anl.gov/petsc/petsc-current/src/snes/examples/tutorials/ex70.c.html
 https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex43.c.html
 http://www.mufits.imec.msu.ru/examples.html
 https://opm-project.org/?page_id=197
 */

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDMechanics.h"
#include "FracDFlow.h"
#include "FracDHeatFlow.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
    AppCtx          bag;
    PetscErrorCode  ierr;
    PetscViewer viewer;
    PetscInt            rank,size;
    
    ierr = PetscInitialize(&argc,&argv,(char*)0,banner);CHKERRQ(ierr);
    
    
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
    
    ierr = FracDInitialize(&bag);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold.txt",&viewer);CHKERRQ(ierr);
    VecView(bag.fields.oP,viewer);
    
    ierr = VecSet(bag.fields.QP,0.);CHKERRQ(ierr);
    
    for(bag.timestep = 0; bag.timestep < bag.maxtimestep; bag.timestep++){
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
        const char* fnaa = "output3Sg.vtk";
        PetscViewerVTKOpen(PETSC_COMM_WORLD,fnaa,FILE_MODE_WRITE,&viewer);
        VecView(bag.fields.Sg,viewer);
        PetscViewerDestroy(&viewer);
        
        const char* fna = "output3P.vtk";
        PetscViewerVTKOpen(PETSC_COMM_WORLD,fna,FILE_MODE_WRITE,&viewer);
        VecView(bag.fields.P,viewer);
        PetscViewerDestroy(&viewer);
        
        const char* fna_1 = "outputSw.vtk";
        PetscViewerVTKOpen(PETSC_COMM_WORLD,fna_1,FILE_MODE_WRITE,&viewer);
        VecView(bag.fields.Sw,viewer);
        PetscViewerDestroy(&viewer);
        ierr = ThisismyVTKwriter(&bag);CHKERRQ(ierr);
        printf("\n Time Step: %d \n",bag.timestep);
    }
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Swnew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.Sw,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pnew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.P,viewer);CHKERRQ(ierr);
    
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"SgNew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag.fields.SaturatedStateIndicator,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Pressureold2.txt",&viewer);CHKERRQ(ierr);
    VecView(bag.fields.oP,viewer);
    
    printf("\n rank: %d/%d \t NEW TIME STEP \n\n\n\n",rank,size);
    
//    ierr = FracDSolveP(&bag);CHKERRQ(ierr);
//    ierr = FracDTimeStepUpdate(&bag);CHKERRQ(ierr);
    
    const char* fna1 = "output3a.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fna1,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.Pb,viewer);
    PetscViewerDestroy(&viewer);
    
    const char* fna1_1 = "output3Sgg.vtk";
    PetscViewerVTKOpen(PETSC_COMM_WORLD,fna1_1,FILE_MODE_WRITE,&viewer);
    VecView(bag.fields.Sg,viewer);
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

