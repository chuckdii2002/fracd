/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */

#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDWellFluidFlow.h"
#include "FracDFlow.h"
#include "FracDComputations.h"


//Check to delete cols in residual functions

//http://lists.mcs.anl.gov/pipermail/petsc-users/2016-July/029956.html
//http://lists.mcs.anl.gov/pipermail/petsc-users/2014-December/023926.html
//http://lists.mcs.anl.gov/pipermail/petsc-dev/2013-May/012206.html
//http://www.mcs.anl.gov/petsc/petsc-dev/src/mat/examples/tests/ex159.c
//http://www.mcs.anl.gov/petsc/petsc-current/src/snes/examples/tutorials/ex70.c.html
//https://github.com/erdc-cm/petsc-dev/blob/master/src/snes/examples/tests/ex17.c
//    http://www.mcs.anl.gov/petsc/petsc-dev/src/ksp/ksp/examples/tests/ex11.c
//http://lists.mcs.anl.gov/pipermail/petsc-users/2015-January/023993.html


#undef __FUNCT__
#define __FUNCT__ "FracDdRpbh_dP"
extern PetscErrorCode FracDdRpbh_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *matvalue=NULL,zero = 0;
    PetscInt       i,ii,j,l,c,w,pt1,numclpts;
    PetscInt       vStart,vEnd,cStart,cEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability;
    Vec            local_perm;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL;
    PetscReal      **coords;
    PetscInt       *closurept=NULL;
    PetscReal      *wellcoords;;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,local_CV,localPbh,localRs;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL,*Rs_array=NULL;
    PetscReal      G,effectiveCellPerm;
    PetscReal      dervR,R1,R,Qs;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc3(ncol,&cols,nrow,&rows,ncol,&matvalue);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    
    for(l = 0; l < bag->CVFEface.elemnodes; l++)    matvalue[l] = 0;
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        coldofIndex = 0;
        for(ii = 0; ii < numclpts; ii++){
            pt1 = closurept[2*ii];
            if(pt1 >= vStart && pt1 < vEnd){
                ierr = PetscSectionGetOffset(globalSection, pt1, &goffset);CHKERRQ(ierr);
                goffset = goffset < 0 ? -(goffset+1):goffset;
                cols[coldofIndex] = goffset;
                coldofIndex++;
            }
        }
        for(ii = 0; ii < bag->CVFEface.dim; ii++){
            Permeability[ii] = Perm_array[ii];
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][ii] = coord_array[ii+j*bag->CVFEface.dim];
            }
        }
        for(j = 0; j < bag->dim; j++) wellcoords[j] = bag->well[w].coordinates[j];
        ierr = bag->FracDCreateDPointFEElement(coords, wellcoords, &bag->epD);CHKERRQ(ierr);
        effectiveCellPerm = 1.;
        for(j = 0; j < bag->dim; j++) {
            effectiveCellPerm = effectiveCellPerm * Permeability[j];
        }
        effectiveCellPerm = PetscPowScalar(effectiveCellPerm,1./bag->dim);
        for(l = 0; l < bag->CVFEface.elemnodes; l++){
            bag->ppties.OilPVTData.B_ModelData[2] = bag->ppties.OilPVTData.rho_ModelData[2] = bag->ppties.OilPVTData.mu_ModelData[2] = Pb_array[l];
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDWellModel(&R,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            ierr = FracDWellModel(&R1,&Qs,bag->well[w],G,Pbh_array[w],P_array[l]+bag->SMALL_PRESSURE,Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            dervR = (R1-R)/(bag->SMALL_PRESSURE);
            matvalue[l] =  bag->epD.phi[l] * dervR;
        }
        ierr = MatSetValues(K, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
        if (KPC != K) {
            ierr = MatSetValues(KPC, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyWellBottomHolePressureCondition(K,bag->well,bag->numWells,zero);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = FracDMatrixApplyWellBottomHolePressureCondition(KPC,bag->well,bag->numWells,zero);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,matvalue);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixWP1.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDdRpbh_dSw"
extern PetscErrorCode FracDdRpbh_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *matvalue=NULL,zero = 0;
    PetscInt       i,ii,j,l,c,w,pt1,numclpts;
    PetscInt       vStart,vEnd,cStart,cEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability;
    Vec            local_perm;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL;
    PetscReal      **coords;
    PetscInt       *closurept=NULL;
    PetscReal      *wellcoords;;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,local_CV,localPbh,localRs;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL,*Rs_array=NULL;
    PetscReal      G,effectiveCellPerm;
    PetscReal      dervR,R1,R,Qs;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc3(ncol,&cols,nrow,&rows,ncol,&matvalue);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    
    for(l = 0; l < bag->CVFEface.elemnodes; l++)        matvalue[l] = 0;
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        coldofIndex = 0;
        for(ii = 0; ii < numclpts; ii++){
            pt1 = closurept[2*ii];
            if(pt1 >= vStart && pt1 < vEnd){
                ierr = PetscSectionGetOffset(globalSection, pt1, &goffset);CHKERRQ(ierr);
                goffset = goffset < 0 ? -(goffset+1):goffset;
                cols[coldofIndex] = goffset;
                coldofIndex++;
            }
        }
        for(ii = 0; ii < bag->CVFEface.dim; ii++){
            Permeability[ii] = Perm_array[ii];
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][ii] = coord_array[ii+j*bag->CVFEface.dim];
            }
        }
        for(j = 0; j < bag->dim; j++) wellcoords[j] = bag->well[w].coordinates[j];
        ierr = bag->FracDCreateDPointFEElement(coords, wellcoords, &bag->epD);CHKERRQ(ierr);
        effectiveCellPerm = 1.;
        for(j = 0; j < bag->dim; j++) {
            effectiveCellPerm = effectiveCellPerm * Permeability[j];
        }
        effectiveCellPerm = PetscPowScalar(effectiveCellPerm,1./bag->dim);
        for(l = 0; l < bag->CVFEface.elemnodes; l++){
            bag->ppties.OilPVTData.B_ModelData[2] = bag->ppties.OilPVTData.rho_ModelData[2] = bag->ppties.OilPVTData.mu_ModelData[2] = Pb_array[l];
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDWellModel(&R,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            ierr = FracDWellModel(&R1,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l]+bag->SMALL_SATURATION,Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            dervR = (R1-R)/(bag->SMALL_SATURATION);
            matvalue[l] =  bag->epD.phi[l] * dervR;
        }
        ierr = MatSetValues(K, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
        if (KPC != K) {
            ierr = MatSetValues(KPC, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyWellBottomHolePressureCondition(K,bag->well,bag->numWells,zero);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = FracDMatrixApplyWellBottomHolePressureCondition(KPC,bag->well,bag->numWells,zero);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,matvalue);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixWP2.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDdRpbh_dSg"
extern PetscErrorCode FracDdRpbh_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *matvalue=NULL,zero = 0;
    PetscInt       i,j,l,c,w,pt1,numclpts;
    PetscInt       vStart,vEnd,cStart,cEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability;
    Vec            local_perm;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL;
    PetscReal      **coords;
    PetscInt       *closurept=NULL;
    PetscReal      *wellcoords;;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,local_CV,localPbh,localRs;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL,*Rs_array=NULL;
    PetscReal      G,effectiveCellPerm,Volumecheck;
    PetscReal      dervR,R1,R,Qs;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc3(ncol,&cols,nrow,&rows,ncol,&matvalue);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    
    for(l = 0; l < bag->CVFEface.elemnodes; l++)
    matvalue[l] = 0;
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Permeability[i] = Perm_array[i];
        coldofIndex = 0;
        for(i = 0; i < numclpts; i++){
            pt1 = closurept[2*i];
            if(pt1 >= vStart && pt1 < vEnd){
                ierr = PetscSectionGetOffset(globalSection, pt1, &goffset);CHKERRQ(ierr);
                goffset = goffset < 0 ? -(goffset+1):goffset;
                cols[coldofIndex] = goffset;
                coldofIndex++;
            }
        }
        for(i = 0; i < bag->CVFEface.dim; i++){
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][i] = coord_array[i+j*bag->CVFEface.dim];
            }
        }
        for(w = 0; w < bag->numWells; w++){
            Volumecheck = 0;
            effectiveCellPerm = 1.;
            for(j = 0; j < bag->dim; j++) wellcoords[j] = bag->well[w].coordinates[j];
            ierr = bag->FracDCreateDPointFEElement(coords, wellcoords, &bag->epD);CHKERRQ(ierr);
            for(l = 0; l < bag->CVFEface.elemnodes; l++)   Volumecheck += bag->epD.phi[l];
            if( PetscAbs(1.0-Volumecheck) < PETSC_SMALL){
                for(j = 0; j < bag->dim; j++) {
                    effectiveCellPerm = effectiveCellPerm * Permeability[j];
                }
                effectiveCellPerm = PetscPowScalar(effectiveCellPerm,1./bag->dim);
                for(l = 0; l < bag->CVFEface.elemnodes; l++){
                    bag->ppties.OilPVTData.B_ModelData[2] = bag->ppties.OilPVTData.rho_ModelData[2] = bag->ppties.OilPVTData.mu_ModelData[2] = Pb_array[l];
                    bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
                    G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
                    ierr = FracDWellModel(&R,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
                    ierr = FracDWellModel(&R1,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l]+bag->SMALL_SATURATION,Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
                    dervR = (R1-R)/(bag->SMALL_SATURATION);
                    matvalue[l] =  bag->epD.phi[l] * dervR;
//                    
//                    if(c == 1759 || c == 1194){
//                        printf("\n\n\n I am checking for dRbp_Sg %g %g %g %g %g %g \n\n\n", matvalue[l],dervR,Pbh_array[w],P_array[l],R, R-R1);
//                    }
                }
                ierr = MatSetValues(K, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
                if (KPC != K) {
                    ierr = MatSetValues(KPC, 1, &w, ncol, cols, matvalue, ADD_VALUES);CHKERRQ(ierr);
                }
            }
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyWellBottomHolePressureCondition(K,bag->well,bag->numWells,zero);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = FracDMatrixApplyWellBottomHolePressureCondition(KPC,bag->well,bag->numWells,zero);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,matvalue);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixWP3.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDdRpbh_dPbh"
extern PetscErrorCode FracDdRpbh_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscReal      matvalue, one = 1.0;
    PetscInt       i,ii,j,l,c,w;
    PetscInt       cStart,cEnd;
    PetscSection   vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability;
    Vec            local_perm;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL;
    PetscReal      **coords;
    PetscReal      *wellcoords;;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,local_CV,localPbh,localRs;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL,*Rs_array=NULL;
    PetscReal      G,effectiveCellPerm;
    PetscReal      dervR,R1,R,Qs;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);

    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        for(ii = 0; ii < bag->CVFEface.dim; ii++){
            Permeability[ii] = Perm_array[ii];
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][ii] = coord_array[ii+j*bag->CVFEface.dim];
            }
        }
        for(j = 0; j < bag->dim; j++) wellcoords[j] = bag->well[w].coordinates[j];
        ierr = bag->FracDCreateDPointFEElement(coords, wellcoords, &bag->epD);CHKERRQ(ierr);
        effectiveCellPerm = 1.;
        for(j = 0; j < bag->dim; j++) {
            effectiveCellPerm = effectiveCellPerm * Permeability[j];
        }
        effectiveCellPerm = PetscPowScalar(effectiveCellPerm,1./bag->dim);
        matvalue = 0;
        for(l = 0; l < bag->CVFEface.elemnodes; l++){
            bag->ppties.OilPVTData.B_ModelData[2] = bag->ppties.OilPVTData.rho_ModelData[2] = bag->ppties.OilPVTData.mu_ModelData[2] = Pb_array[l];
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDWellModel(&R,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            ierr = FracDWellModel(&R1,&Qs,bag->well[w],G,Pbh_array[w]+bag->SMALL_PRESSURE,P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            dervR = (R1-R)/(bag->SMALL_PRESSURE);
            matvalue +=  bag->epD.phi[l] * dervR;
        }
        ierr = MatSetValues(K, 1, &w, 1, &w, &matvalue, ADD_VALUES);CHKERRQ(ierr);
        if (KPC != K) {
            ierr = MatSetValues(KPC, 1, &w, 1, &w, &matvalue, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
    }

    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyWellBottomHolePressureCondition(K,bag->well,bag->numWells,one);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = FracDMatrixApplyWellBottomHolePressureCondition(KPC,bag->well,bag->numWells,one);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixWP4.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDRpbh"
extern PetscErrorCode FracDRpbh(void *user, Vec RPbh, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       i,ii,j,l,c,w,numclpts,rank;
    PetscInt       cStart,cEnd;
    PetscSection   vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability;
    Vec            local_perm;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL;
    PetscReal      **coords;
    PetscInt       *closurept=NULL;
    PetscReal      *wellcoords;;
    Vec            localResidual,localP,localPb,localPcow,localPcog,localSw,localSg,localPbh,local_CV,localRs;
    Vec            localQwbh,localQobh,localQgbh;
    PetscScalar    *Residual_array=NULL,*Res_array=NULL,*P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Rs_array=NULL;
    PetscScalar    *Qwbh_array=NULL,*Qobh_array=NULL,*Qgbh_array=NULL;
    const PetscScalar    *Pbh_array=NULL;
    PetscReal      G,effectiveCellPerm;
    PetscReal      R,Qs,Qw,Qo,Qg;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    ierr = VecSet(RPbh,0.);CHKERRQ(ierr);
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Rs,INSERT_VALUES,localRs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArrayRead(localPbh,&Pbh_array);CHKERRQ(ierr);
    
    ierr = VecSet(bag->fields.Qwbh,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(bag->WellRedun,&localQwbh);CHKERRQ(ierr);
    ierr = VecSet(localQwbh,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localQwbh,&Qwbh_array);CHKERRQ(ierr);
    
    ierr = VecSet(bag->fields.Qobh,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(bag->WellRedun,&localQobh);CHKERRQ(ierr);
    ierr = VecSet(localQobh,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localQobh,&Qobh_array);CHKERRQ(ierr);
    
    ierr = VecSet(bag->fields.Qgbh,0.);CHKERRQ(ierr);
    ierr = DMGetLocalVector(bag->WellRedun,&localQgbh);CHKERRQ(ierr);
    ierr = VecSet(localQgbh,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localQgbh,&Qgbh_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localResidual,&Residual_array);CHKERRQ(ierr);
    
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(ii = 0; ii < bag->CVFEface.dim; ii++){
            Permeability[ii] = Perm_array[ii];
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][ii] = coord_array[ii+j*bag->CVFEface.dim];
            }
        }
        for(j = 0; j < bag->dim; j++) wellcoords[j] = bag->well[w].coordinates[j];
        ierr = bag->FracDCreateDPointFEElement(coords, wellcoords, &bag->epD);CHKERRQ(ierr);
        effectiveCellPerm = 1.;
        for(j = 0; j < bag->dim; j++) {
            effectiveCellPerm = effectiveCellPerm * Permeability[j];
        }
        effectiveCellPerm = PetscPowScalar(effectiveCellPerm,1./bag->dim);
        for(l = 0; l < bag->CVFEface.elemnodes; l++){
            bag->ppties.OilPVTData.B_ModelData[2] = bag->ppties.OilPVTData.rho_ModelData[2] = bag->ppties.OilPVTData.mu_ModelData[2] = Pb_array[l];
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDWellModel(&R,&Qs,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            Residual_array[w] += bag->epD.phi[l] * (R-Qs);
            ierr = FracDWellModelRates(&Qw,&Qo,&Qg,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData,bag->ppties.SolutionGasOilData);CHKERRQ(ierr);
            Qwbh_array[w] += bag->epD.phi[l] * Qw/bag->WellinMeshData.Count[w];
            Qobh_array[w] += bag->epD.phi[l] * Qo/bag->WellinMeshData.Count[w];
            Qgbh_array[w] += bag->epD.phi[l] * Qg/bag->WellinMeshData.Count[w];
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localRs, c, NULL, &Rs_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    
    ierr = VecRestoreArray(localQwbh,&Qwbh_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->WellRedun,localQwbh,ADD_VALUES,bag->fields.Qwbh);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->WellRedun,localQwbh,ADD_VALUES,bag->fields.Qwbh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localQwbh);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(localQobh,&Qobh_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->WellRedun,localQobh,ADD_VALUES,bag->fields.Qobh);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->WellRedun,localQobh,ADD_VALUES,bag->fields.Qobh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localQobh);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(localQgbh,&Qgbh_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->WellRedun,localQgbh,ADD_VALUES,bag->fields.Qgbh);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->WellRedun,localQgbh,ADD_VALUES,bag->fields.Qgbh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localQgbh);CHKERRQ(ierr);
    
    ierr = VecRestoreArray(localResidual,&Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->WellRedun,localResidual,ADD_VALUES,RPbh);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->WellRedun,localResidual,ADD_VALUES,RPbh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localResidual);CHKERRQ(ierr);
    
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localRs);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    
    if(!rank){
        ierr = VecGetArray(RPbh,&Res_array);CHKERRQ(ierr);
        for(w = 0; w < bag->numWells; w++)  {
            if(bag->well[w].condition == PRESSURE){
                Res_array[w] = Pbh_array[w] - bag->well[w].Pbh;
            }
        }
        ierr = VecRestoreArray(RPbh,&Res_array);CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    
    PetscViewer viewer;
    //    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PRes1.txt",&viewer);CHKERRQ(ierr);
    //    ierr = VecView(P,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Rpbh.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(RPbh,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Psol1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(Pbh,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PRatew.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag->fields.Qwbh,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PRateg.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag->fields.Qgbh,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"PRateo.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(bag->fields.Qobh,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDWellModel"
extern PetscErrorCode FracDWellModel(PetscReal *Q,PetscReal *Qconstr,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilData)
{
    PetscErrorCode ierr;
    PetscReal      Bw,muw,krw;
    PetscReal      Bo,muo,kro,Rs;
    PetscReal      Bg,mug,krg;
    PetscReal      MobilityWater,MobilityOil,MobilityGas;
    
    PetscFunctionBegin;
    ierr = RelPermData.FracDUpDateKrw(&krw,Sw,PETSC_NULL,RelPermData.Krw_TableData,PETSC_NULL,RelPermData.numwaterdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKro(&kro,Sw,Sg,PETSC_NULL,RelPermData.Krow_TableData,RelPermData.Krog_TableData,RelPermData.Krw_TableData,RelPermData.stone_model_data,RelPermData.numwaterdatarow,RelPermData.numgasdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKrg(&krg,Sg,PETSC_NULL,RelPermData.Krg_TableData,PETSC_NULL,RelPermData.numgasdatarow);CHKERRQ(ierr);
    
    ierr = WaterPVTData.FracDUpDateFVF(&Bw,P,PETSC_NULL,WaterPVTData.B_TableData,WaterPVTData.B_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    ierr = WaterPVTData.FracDUpDateViscosity(&muw,P,PETSC_NULL,WaterPVTData.mu_TableData,WaterPVTData.mu_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = OilPVTData.FracDUpDateFVF(&Bo,P,PETSC_NULL,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    ierr = OilPVTData.FracDUpDateViscosity(&muo,P,PETSC_NULL,OilPVTData.mu_TableData,OilPVTData.mu_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = GasPVTData.FracDUpDateFVF(&Bg,P,PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    ierr = GasPVTData.FracDUpDateViscosity(&mug,P,PETSC_NULL,GasPVTData.mu_TableData,GasPVTData.mu_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = SolutionGasOilData.FracDUpDateSolutionGasOilRatio(&Rs,P,PETSC_NULL,SolutionGasOilData.TableData,SolutionGasOilData.ModelData,SolutionGasOilData.numdatarow);CHKERRQ(ierr);

    MobilityWater = G*krw*(Pbh-P+Pcow)/(muw*Bw);
    MobilityOil = G*kro*(Pbh-P)/(muo*Bo);
    MobilityGas = G*krg*(Pbh-P-Pcog)/(mug*Bg);
    switch (well.condition) {
        case 0:
        {
            *Q = *Qconstr = 1.;/* meaningless for both options*/
        }
            break;
        case 1:
        {
            if(well.type == INJECTORWATER)    *Q = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P+Pcow)/Bw;
            else    *Q = MobilityWater;
            *Qconstr = well.Qws;
        }
            break;
        case 2:
        {
            *Q = MobilityOil;
            *Qconstr = well.Qos;
        }
            break;
        case 3:
        {
            if(well.type == INJECTORGAS)    *Q = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P-Pcog)/Bg;
            else    *Q = MobilityGas + Rs*MobilityOil;
            *Qconstr = well.Qgs;
        }
            break;
        case 4:
        {
            *Q = MobilityWater+MobilityOil;
            *Qconstr = well.QL;
        }
            break;
        case 5:
        {
            *Q = MobilityWater+MobilityOil+MobilityGas;
            *Qconstr = well.QT;
        }
            break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDQw"
extern PetscErrorCode FracDQw(PetscReal *Qw,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData)
{
    PetscErrorCode ierr;
    PetscReal      Bw,muw,krw;
    PetscReal      Bo,muo,kro;
    PetscReal      Bg,mug,krg;
    PetscReal      MobilityWater,MobilityOil,MobilityGas;
    
    PetscFunctionBegin;
    ierr = RelPermData.FracDUpDateKrw(&krw,Sw,PETSC_NULL,RelPermData.Krw_TableData,PETSC_NULL,RelPermData.numwaterdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKro(&kro,Sw,Sg,PETSC_NULL,RelPermData.Krow_TableData,RelPermData.Krog_TableData,RelPermData.Krw_TableData,RelPermData.stone_model_data,RelPermData.numwaterdatarow,RelPermData.numgasdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKrg(&krg,Sg,PETSC_NULL,RelPermData.Krg_TableData,PETSC_NULL,RelPermData.numgasdatarow);CHKERRQ(ierr);
    
    ierr = WaterPVTData.FracDUpDateFVF(&Bw,P,PETSC_NULL,WaterPVTData.B_TableData,WaterPVTData.B_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    ierr = WaterPVTData.FracDUpDateViscosity(&muw,P,PETSC_NULL,WaterPVTData.mu_TableData,WaterPVTData.mu_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = OilPVTData.FracDUpDateFVF(&Bo,P,PETSC_NULL,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    ierr = OilPVTData.FracDUpDateViscosity(&muo,P,PETSC_NULL,OilPVTData.mu_TableData,OilPVTData.mu_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = GasPVTData.FracDUpDateFVF(&Bg,P,PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    ierr = GasPVTData.FracDUpDateViscosity(&mug,P,PETSC_NULL,GasPVTData.mu_TableData,GasPVTData.mu_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    
    MobilityWater = krw*(Pbh-P+Pcow)/(muw*Bw);
    MobilityOil = kro*(Pbh-P)/(muo*Bo);
    MobilityGas = krg*(Pbh-P-Pcog)/(mug*Bg);
    
    switch (well.condition) {
        case 0:
        {
            if(well.type == INJECTORWATER)
                *Qw = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P+Pcow)/Bw;
            else
                *Qw = G * MobilityWater;
        }
            break;
        case 1:
            *Qw = well.Qws;
            break;
        case 2:
            *Qw = MobilityWater/MobilityOil * well.Qos;
            break;
        case 3:
        {
            *Qw = MobilityWater/MobilityGas * well.Qgs;
        }
            break;
        case 4:
            *Qw = MobilityWater/(MobilityWater+MobilityOil) * well.QL;
            break;
        case 5:
            *Qw = MobilityWater/(MobilityWater+MobilityOil+MobilityGas) * well.QT;
            break;
    }
    if(well.type == INJECTORGAS)    *Qw = 0;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDQo"
extern PetscErrorCode FracDQo(PetscReal *Qo,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData)
{
    PetscErrorCode ierr;
    PetscReal      Bw,muw,krw;
    PetscReal      Bo,muo,kro;
    PetscReal      Bg,mug,krg;
    PetscReal      MobilityWater,MobilityOil,MobilityGas;
    
    PetscFunctionBegin;
    ierr = RelPermData.FracDUpDateKrw(&krw,Sw,PETSC_NULL,RelPermData.Krw_TableData,PETSC_NULL,RelPermData.numwaterdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKro(&kro,Sw,Sg,PETSC_NULL,RelPermData.Krow_TableData,RelPermData.Krog_TableData,RelPermData.Krw_TableData,RelPermData.stone_model_data,RelPermData.numwaterdatarow,RelPermData.numgasdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKrg(&krg,Sg,PETSC_NULL,RelPermData.Krg_TableData,PETSC_NULL,RelPermData.numgasdatarow);CHKERRQ(ierr);
    
    ierr = WaterPVTData.FracDUpDateFVF(&Bw,P,PETSC_NULL,WaterPVTData.B_TableData,WaterPVTData.B_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    ierr = WaterPVTData.FracDUpDateViscosity(&muw,P,PETSC_NULL,WaterPVTData.mu_TableData,WaterPVTData.mu_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = OilPVTData.FracDUpDateFVF(&Bo,P,PETSC_NULL,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    ierr = OilPVTData.FracDUpDateViscosity(&muo,P,PETSC_NULL,OilPVTData.mu_TableData,OilPVTData.mu_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = GasPVTData.FracDUpDateFVF(&Bg,P,PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    ierr = GasPVTData.FracDUpDateViscosity(&mug,P,PETSC_NULL,GasPVTData.mu_TableData,GasPVTData.mu_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    
    MobilityWater = krw*(Pbh-P+Pcow)/(muw*Bw);
    MobilityOil = kro*(Pbh-P)/(muo*Bo);
    MobilityGas = krg*(Pbh-P-Pcog)/(mug*Bg);
    switch (well.condition) {
        case 0:
            *Qo = G * MobilityOil;
            break;
        case 1:
            *Qo = MobilityOil/MobilityWater * well.Qws;
            break;
        case 2:
            *Qo = well.Qos;
            break;
        case 3:
            *Qo = MobilityOil/MobilityGas * well.Qgs;
            break;
        case 4:
            *Qo = MobilityOil/(MobilityWater+MobilityOil) * well.QL;
            break;
        case 5:
            *Qo = MobilityOil/(MobilityWater+MobilityOil+MobilityGas) * well.QT;
            break;
    }
    if(well.type == INJECTORWATER || well.type == INJECTORGAS)    *Qo = 0;

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDQg"
extern PetscErrorCode FracDQg(PetscReal *Qg,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilData)
{
    PetscErrorCode ierr;
    PetscReal      Bw,muw,krw;
    PetscReal      Bo,muo,kro,Qo,Rs;
    PetscReal      Bg,mug,krg;
    PetscReal      MobilityWater,MobilityOil,MobilityGas;
    
    PetscFunctionBegin;
    ierr = RelPermData.FracDUpDateKrw(&krw,Sw,PETSC_NULL,RelPermData.Krw_TableData,PETSC_NULL,RelPermData.numwaterdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKro(&kro,Sw,Sg,PETSC_NULL,RelPermData.Krow_TableData,RelPermData.Krog_TableData,RelPermData.Krw_TableData,RelPermData.stone_model_data,RelPermData.numwaterdatarow,RelPermData.numgasdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKrg(&krg,Sg,PETSC_NULL,RelPermData.Krg_TableData,PETSC_NULL,RelPermData.numgasdatarow);CHKERRQ(ierr);
    
    ierr = WaterPVTData.FracDUpDateFVF(&Bw,P,PETSC_NULL,WaterPVTData.B_TableData,WaterPVTData.B_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    ierr = WaterPVTData.FracDUpDateViscosity(&muw,P,PETSC_NULL,WaterPVTData.mu_TableData,WaterPVTData.mu_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = OilPVTData.FracDUpDateFVF(&Bo,P,PETSC_NULL,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    ierr = OilPVTData.FracDUpDateViscosity(&muo,P,PETSC_NULL,OilPVTData.mu_TableData,OilPVTData.mu_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = GasPVTData.FracDUpDateFVF(&Bg,P,PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    ierr = GasPVTData.FracDUpDateViscosity(&mug,P,PETSC_NULL,GasPVTData.mu_TableData,GasPVTData.mu_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = SolutionGasOilData.FracDUpDateSolutionGasOilRatio(&Rs,P,PETSC_NULL,SolutionGasOilData.TableData,SolutionGasOilData.ModelData,SolutionGasOilData.numdatarow);CHKERRQ(ierr);

    MobilityWater = krw*(Pbh-P+Pcow)/(muw*Bw);
    MobilityOil = kro*(Pbh-P)/(muo*Bo);
    MobilityGas = krg*(Pbh-P-Pcog)/(mug*Bg);
    switch (well.condition) {
        case 0:
        {
            if(well.type == INJECTORGAS)
            *Qg = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P-Pcog)/Bg;
            else
            *Qg = G * MobilityGas;
        }
        break;
        case 1:
        *Qg = MobilityGas/MobilityWater * well.Qws;
        break;
        case 2:
        *Qg = MobilityGas/MobilityOil * well.Qos;
        break;
        case 3:
        *Qg = well.Qgs;
        break;
        case 4:
        *Qg = MobilityGas/(MobilityWater+MobilityOil) * well.QL;
        break;
        case 5:
        *Qg = MobilityGas/(MobilityWater+MobilityOil+MobilityGas) * well.QT;
        break;
    }
    if(well.type == INJECTORWATER)    *Qg = 0;
    
    switch (well.condition) {
        case 0:
        Qo = G * MobilityOil;
        break;
        case 1:
        Qo = MobilityOil/MobilityWater * well.Qws;
        break;
        case 2:
        Qo = well.Qos;
        break;
        case 3:
        Qo = MobilityOil/MobilityGas * well.Qgs;
        break;
        case 4:
        Qo = MobilityOil/(MobilityWater+MobilityOil) * well.QL;
        break;
        case 5:
        Qo = MobilityOil/(MobilityWater+MobilityOil+MobilityGas) * well.QT;
        break;
    }
    if(well.type == INJECTORWATER || well.type == INJECTORGAS)    Qo = 0;
    *Qg = *Qg + Rs*Qo;    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDWellModelRates"
extern PetscErrorCode FracDWellModelRates(PetscReal *Qw,PetscReal *Qo,PetscReal *Qg,FracDWell well,PetscReal G,PetscReal Pbh,PetscReal P,PetscReal Pb,PetscReal Sw,PetscReal Sg,PetscReal Pcow,PetscReal Pcog,FracDPVT WaterPVTData,FracDPVT OilPVTData,FracDPVT GasPVTData,FracDRelPerm RelPermData,FracDPbRs SolutionGasOilData)
{
    PetscErrorCode ierr;
    PetscReal      Bw,muw,krw;
    PetscReal      Bo,muo,kro,Rs;
    PetscReal      Bg,mug,krg;
    PetscReal      MobilityWater,MobilityOil,MobilityGas;
    
    PetscFunctionBegin;
    ierr = RelPermData.FracDUpDateKrw(&krw,Sw,PETSC_NULL,RelPermData.Krw_TableData,PETSC_NULL,RelPermData.numwaterdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKro(&kro,Sw,Sg,PETSC_NULL,RelPermData.Krow_TableData,RelPermData.Krog_TableData,RelPermData.Krw_TableData,RelPermData.stone_model_data,RelPermData.numwaterdatarow,RelPermData.numgasdatarow);CHKERRQ(ierr);
    ierr = RelPermData.FracDUpDateKrg(&krg,Sg,PETSC_NULL,RelPermData.Krg_TableData,PETSC_NULL,RelPermData.numgasdatarow);CHKERRQ(ierr);
    
    ierr = WaterPVTData.FracDUpDateFVF(&Bw,P,PETSC_NULL,WaterPVTData.B_TableData,WaterPVTData.B_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    ierr = WaterPVTData.FracDUpDateViscosity(&muw,P,PETSC_NULL,WaterPVTData.mu_TableData,WaterPVTData.mu_ModelData,WaterPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = OilPVTData.FracDUpDateFVF(&Bo,P,PETSC_NULL,OilPVTData.B_TableData,OilPVTData.B_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    ierr = OilPVTData.FracDUpDateViscosity(&muo,P,PETSC_NULL,OilPVTData.mu_TableData,OilPVTData.mu_ModelData,OilPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = GasPVTData.FracDUpDateFVF(&Bg,P,PETSC_NULL,GasPVTData.B_TableData,GasPVTData.B_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    ierr = GasPVTData.FracDUpDateViscosity(&mug,P,PETSC_NULL,GasPVTData.mu_TableData,GasPVTData.mu_ModelData,GasPVTData.numdatarow);CHKERRQ(ierr);
    
    ierr = SolutionGasOilData.FracDUpDateSolutionGasOilRatio(&Rs,P,PETSC_NULL,SolutionGasOilData.TableData,SolutionGasOilData.ModelData,SolutionGasOilData.numdatarow);CHKERRQ(ierr);

    MobilityWater = G*krw*(Pbh-P+Pcow)/(muw*Bw);
    MobilityOil = G*kro*(Pbh-P)/(muo*Bo);
    MobilityGas = G*krg*(Pbh-P-Pcog)/(mug*Bg);
    
    


    
    if(well.type == INJECTORWATER)
        *Qw = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P+Pcow)/Bw;
    else if(well.type == INJECTORGAS)
        *Qw = 0.;
    else
        *Qw = MobilityWater;
    
    if(well.type == INJECTORWATER || well.type == INJECTORGAS)
        *Qo = 0;
    else
        *Qo = MobilityOil;
    
    if(well.type == INJECTORGAS)
        *Qg = G * (krw/muw+kro/muo+krg/mug)*(Pbh-P-Pcog)/Bg;
    else if(well.type == INJECTORWATER)
        *Qg = 0.;
    else
        *Qg = MobilityGas;
    
    *Qg += *Qo * Rs;
    
//    printf("\nWater rate: %g \t %g \t %g \t %g \t %g \n",*Qw,krw, Sw, Pbh, P);
//    printf("Oil rate: %g \t %g \t %g \t %g \t %g\n",*Qo,kro,muo,Bo, P);
//    printf("gas rate: %g \t %g \t %g  \t %g \t %g \n",MobilityGas,*Qg,mug,Bg, P);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDMatrixApplyWellBottomHolePressureCondition"
extern PetscErrorCode FracDMatrixApplyWellBottomHolePressureCondition(Mat K, FracDWell *well, PetscInt numwells, PetscScalar diagonalvalue)
{
    PetscErrorCode      ierr;
    PetscInt            w;
    
    PetscFunctionBegin;
    for(w = 0; w < numwells; w++){
        if(well[w].condition == PRESSURE){
            ierr = MatZeroRows(K,1,&w,diagonalvalue,NULL,NULL);CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}
