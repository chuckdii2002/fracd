/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */

#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDWaterFluidFlow.h"
#include "FracDFlow.h"
#include "FracDComputations.h"
#include "FracDOilFluidFlow.h"
#include "FracDWellFluidFlow.h"

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
#define __FUNCT__ "FracDdRo_dP"
extern PetscErrorCode FracDdRo_dP(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *K_local,*K1_local,matvalue;
    PetscInt       d,i,j,k,ii,jj,kk,l,c,cj,w;
    PetscInt       pt,rpt,pt1,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability,scale;
    Vec            local_perm,local_phi,local_CV;
    PetscScalar    *Perm_array=NULL,*Phi_array=NULL,*CV_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords,*wellcoords;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,localPbh;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL;
    PetscScalar    *P1_array=NULL,*Sw1_array=NULL,*Sg1_array=NULL;
    PetscReal      dt,theta,effectiveCellPerm;
    PetscReal      G,lambda,phi,phiData[3],fpress,fbubblepress;
    PetscReal      Bo,BoData[3],muo,muoData[3],kro,Qo,Qo1,dervQo,kr_check,kr[2];
    PetscReal      invBo_derv,invmuo_derv,phi_derv;
    PetscReal      CFbeta, CFalpha, CFgamma;
    PetscErrorCode      (*FracDProjectFaceCoordinateDimensions)(PetscReal**,PetscReal**, PetscInt, PetscInt) = NULL;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    switch (bag->dim) {
        case 2:
            FracDProjectFaceCoordinateDimensions = FracD1DProjectFaceCoordinateDimensions;
            break;
        case 3:
            FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensions;
            break;
    }
    dt = bag->timevalue;
    theta = bag->theta;
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,2,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    edgecentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    facecentroid = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    edgecoords = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    for(i = 0; i < 2; i++)
    {
        facecentroid[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
        edgecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    if(bag->CVFEface.dim == 2)
    {
        gStart = 1;
        gStart  = fStart;   gEnd  = fEnd;
        ggStart = cStart;   ggEnd = cEnd;
    }
    if(bag->CVFEface.dim == 3)
    {
        gStart  = eStart;   gEnd  = eEnd;
        ggStart = fStart;   ggEnd = fEnd;
    }
    cvfacesize = bag->CVFEface.nodes;
    scale = bag->CVFEface.scale;
    CVfacecoords = (PetscReal **)malloc(cvfacesize * sizeof(PetscReal *));
    for(i = 0; i < cvfacesize; i++)
    {
        CVfacecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc4(ncol,&cols,nrow,&rows,ncol*nrow,&K_local,ncol*nrow,&K1_local);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
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
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Permeability[i] = Perm_array[i];
        bag->ppties.PhiData[0] = Phi_array[0];
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
            cellcentroid[i] = 0;
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][i] = coord_array[i+j*bag->CVFEface.dim];
                cellcentroid[i] += coord_array[i+j*bag->CVFEface.dim];
            }
            cellcentroid[i] = cellcentroid[i]/bag->CVFEface.elemnodes;
        }
        for(k = 0; k < numclpts; k++){
            pt = closurept[2*k];
            if(pt >= gStart && pt < gEnd){
                ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                for(l = 0; l < bag->CVFEface.dim; l++){
                    edgecoords[0][l] = ecoord_array[l];
                    edgecoords[1][l] = ecoord_array[l+bag->dim];
                    edgecentroid[l] = 1/2.*(ecoord_array[l]+ecoord_array[l+bag->dim]);
                }
                if(bag->dim == 3){
                    d = 0;
                    ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                    for(kk = 0; kk < rnumclpts; kk++){
                        rpt = rclosurept[2*kk];
                        if(rpt >= ggStart && rpt < ggEnd){
                            for(ii = 0; ii < numclpts; ii++){
                                if(rpt == closurept[2*ii]){
                                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    for(l = 0; l < bag->CVFEface.dim; l++){
                                        facecentroid[d][l] = 0.;
                                        for(jj = 0; jj < bag->elD.nodes; jj++){
                                            facecentroid[d][l] += fcoord_array[l+jj*bag->dim];
                                        }
                                        facecentroid[d][l] = 1./bag->elD.nodes * facecentroid[d][l];
                                    }
                                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    d++;
                                }
                            }
                        }
                    }
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = facecentroid[0][i];
                        CVfacecoords[2][i] = edgecentroid[i];
                        CVfacecoords[3][i] = facecentroid[1][i];
                    }
                    ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                }
                else{
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = edgecentroid[i];
                    }
                }
                ierr = bag->FracDCreateCVFEFace(coords, CVfacecoords, &bag->CVFEface);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                //                Upwinding scheme is implemented below
                fbubblepress = fpress = 0;
                for(i = 0; i < bag->CVFEface.elemnodes; i++){
                    fpress += P_array[i]*bag->CVFEface.phi[i];
                    fbubblepress += Pb_array[i]*bag->CVFEface.phi[i];
                }

                
                
                
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[0],Sw1_array[0],Sg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);

                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[1],Sw1_array[1],Sg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);

                ierr = FracDQuantityAndDerivativeComputation(BoData,fpress,bag->SMALL_PRESSURE,fbubblepress,bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow,bag->ppties.OilPVTData.FracDUpDateFVF);CHKERRQ(ierr);
                
                ierr = FracDQuantityAndDerivativeComputation(muoData,fpress,bag->SMALL_PRESSURE,fbubblepress,bag->ppties.OilPVTData.mu_TableData,bag->ppties.OilPVTData.mu_ModelData,bag->ppties.OilPVTData.numdatarow,bag->ppties.OilPVTData.FracDUpDateViscosity);CHKERRQ(ierr);
                
                Bo= BoData[0]; invBo_derv = BoData[2];
                muo = muoData[0]; invmuo_derv = muoData[2];
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
//                lambda = kro/(muo*Bo);
                jj = 0;
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    kr_check = 0;
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = PetscSectionGetOffset(globalSection, rpt1, &goffset);CHKERRQ(ierr);
                        goffset = goffset < 0 ? -(goffset+1):goffset;
                        rows[0] = goffset;
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Permeability, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++)   kr_check += K_local[cj] * (P_array[cj]);
                        if(kr_check > 0) kro = kr[jj];
                        else    kro = kr[(jj+1)%2];
                        lambda = kro/(muo*Bo);
                        for(l = 0; l < nrow*ncol; l++) {
                            K1_local[l] = CFbeta * theta * lambda * K_local[l];
                        }
                        matvalue = 0;
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++) {
                            matvalue += CFbeta * theta * lambda * (Bo * invBo_derv + muo * invmuo_derv) * K_local[cj] * (P_array[cj]);
                        }
                        ierr = MatSetValues(K, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                        ierr = MatSetValues(K, nrow, rows, ncol, cols, K1_local, ADD_VALUES);CHKERRQ(ierr);
                        if (KPC != K) {
                            ierr = MatSetValues(KPC, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                            ierr = MatSetValues(KPC, nrow, rows, ncol, cols, K1_local, ADD_VALUES);CHKERRQ(ierr);
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        jj++;
                    }
                }
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            ierr = FracDQuantityAndDerivativeComputation(BoData,P_array[i],bag->SMALL_PRESSURE,Pb_array[i],bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow,bag->ppties.OilPVTData.FracDUpDateFVF);CHKERRQ(ierr);
            Bo = BoData[0]; invBo_derv = BoData[2];
            ierr = FracDQuantityAndDerivativeComputation(phiData,P_array[i],bag->SMALL_PRESSURE,PETSC_NULL,PETSC_NULL,bag->ppties.PhiData,1,FracDInterpolateUsingAnalyticalModel);CHKERRQ(ierr);
            phi = phiData[0]; phi_derv = phiData[1];
            matvalue = (1./dt) * 1./CFalpha * scale * bag->CVFEface.elemVolume * (1.0-Sw_array[i]-Sg_array[i]) * (phi*invBo_derv+phi_derv/Bo);
            ierr = MatSetValues(K, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }

    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
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
        matvalue = 0;
        for(l = 0; l < bag->CVFEface.elemnodes; l++){
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDQo(&Qo,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            ierr = FracDQo(&Qo1,bag->well[w],G,Pbh_array[w],P_array[l]+bag->SMALL_PRESSURE,Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            dervQo = (Qo1-Qo)/bag->SMALL_PRESSURE;
            matvalue =  -1. * theta * bag->epD.phi[l] * dervQo;
            ierr = MatSetValues(K, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }

    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree4(cols,rows,K_local,K1_local);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    for(i = 0; i < cvfacesize; i++)  free(CVfacecoords[i]);
    for(i = 0; i < 2; i++) {
        free(facecentroid[i]);
        free(edgecoords[i]);
    }
    free(facecentroid);
    free(wellcoords);
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Permeability);
    
    /*
     PetscViewer viewer;
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOP1.txt",&viewer);CHKERRQ(ierr);
     ierr = MatView(K,viewer);CHKERRQ(ierr);
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOPPCU1.txt",&viewer);CHKERRQ(ierr);
     ierr = MatView(KPC,viewer);CHKERRQ(ierr);
    */
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDdRo_dSw"
extern PetscErrorCode FracDdRo_dSw(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *K_local,matvalue;
    PetscInt       d,i,j,k,ii,jj,kk,l,c,cj,w;
    PetscInt       pt,rpt,pt1,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability,scale;
    Vec            local_perm,local_phi,local_CV;
    PetscScalar    *Perm_array=NULL,*Phi_array=NULL,*CV_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords,*wellcoords;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,localPbh;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL;
    PetscScalar    *P1_array=NULL,*Sw1_array=NULL,*Sg1_array=NULL;
    PetscReal      dt,theta,effectiveCellPerm;
    PetscReal      G,phi,fpress,fbubblepress;
    PetscReal      Bo,muo,kro,kro1,Qo,Qo1,dervQo,kro_derv,kr_check,kr[2],kr1[2];
    PetscReal      CFbeta, CFalpha, CFgamma;
    PetscErrorCode      (*FracDProjectFaceCoordinateDimensions)(PetscReal**,PetscReal**, PetscInt, PetscInt) = NULL;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    switch (bag->dim) {
        case 2:
            FracDProjectFaceCoordinateDimensions = FracD1DProjectFaceCoordinateDimensions;
            break;
        case 3:
            FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensions;
            break;
    }
    dt = bag->timevalue;
    theta = bag->theta;

    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,2,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    edgecentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    facecentroid = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    edgecoords = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    for(i = 0; i < 2; i++)
    {
        facecentroid[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
        edgecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    if(bag->CVFEface.dim == 2)
    {
        gStart = 1;
        gStart  = fStart;   gEnd  = fEnd;
        ggStart = cStart;   ggEnd = cEnd;
    }
    if(bag->CVFEface.dim == 3)
    {
        gStart  = eStart;   gEnd  = eEnd;
        ggStart = fStart;   ggEnd = fEnd;
    }
    cvfacesize = bag->CVFEface.nodes;
    scale = bag->CVFEface.scale;
    CVfacecoords = (PetscReal **)malloc(cvfacesize * sizeof(PetscReal *));
    for(i = 0; i < cvfacesize; i++)
    {
        CVfacecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc3(ncol,&cols,nrow,&rows,ncol*nrow,&K_local);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
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
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Permeability[i] = Perm_array[i];
        bag->ppties.PhiData[0] = Phi_array[0];
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
            cellcentroid[i] = 0;
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][i] = coord_array[i+j*bag->CVFEface.dim];
                cellcentroid[i] += coord_array[i+j*bag->CVFEface.dim];
            }
            cellcentroid[i] = cellcentroid[i]/bag->CVFEface.elemnodes;
        }
        for(k = 0; k < numclpts; k++){
            pt = closurept[2*k];
            if(pt >= gStart && pt < gEnd){
                ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                for(l = 0; l < bag->CVFEface.dim; l++){
                    edgecoords[0][l] = ecoord_array[l];
                    edgecoords[1][l] = ecoord_array[l+bag->dim];
                    edgecentroid[l] = 1/2.*(ecoord_array[l]+ecoord_array[l+bag->dim]);
                }
                if(bag->dim == 3){
                    d = 0;
                    ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                    for(kk = 0; kk < rnumclpts; kk++){
                        rpt = rclosurept[2*kk];
                        if(rpt >= ggStart && rpt < ggEnd){
                            for(ii = 0; ii < numclpts; ii++){
                                if(rpt == closurept[2*ii]){
                                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    for(l = 0; l < bag->CVFEface.dim; l++){
                                        facecentroid[d][l] = 0.;
                                        for(jj = 0; jj < bag->elD.nodes; jj++){
                                            facecentroid[d][l] += fcoord_array[l+jj*bag->dim];
                                        }
                                        facecentroid[d][l] = 1./bag->elD.nodes * facecentroid[d][l];
                                    }
                                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    d++;
                                }
                            }
                        }
                    }
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = facecentroid[0][i];
                        CVfacecoords[2][i] = edgecentroid[i];
                        CVfacecoords[3][i] = facecentroid[1][i];
                    }
                    ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                }
                else{
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = edgecentroid[i];
                    }
                }
                ierr = bag->FracDCreateCVFEFace(coords, CVfacecoords, &bag->CVFEface);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                //                Upwinding scheme is implemented below
                fpress = 0;
                fbubblepress = 0;
                for(i = 0; i < bag->CVFEface.elemnodes; i++){
                    fpress += P_array[i]*bag->CVFEface.phi[i];
                    fbubblepress += Pb_array[i]*bag->CVFEface.phi[i];
                }
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[0],Sw1_array[0],Sg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[1],Sw1_array[1],Sg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr1[0],Sw1_array[0]+bag->SMALL_SATURATION,Sg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr1[1],Sw1_array[1]+bag->SMALL_SATURATION,Sg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,fpress,fbubblepress,bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.OilPVTData.FracDUpDateViscosity(&muo,fpress,fbubblepress,bag->ppties.OilPVTData.mu_TableData,bag->ppties.OilPVTData.mu_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                jj = 0;
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    kr_check = 0;
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = PetscSectionGetOffset(globalSection, rpt1, &goffset);CHKERRQ(ierr);
                        goffset = goffset < 0 ? -(goffset+1):goffset;
                        rows[0] = goffset;
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Permeability, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++)   kr_check += K_local[cj] * (P_array[cj]);
                        if(kr_check > 0){
                            kro = kr[jj];
                            kro1 = kr1[jj];
                        }
                        else{
                            kro = kr[(jj+1)%2];
                            kro1 = kr1[(jj+1)%2];
                        }
                        kro_derv = (kro1-kro)/(bag->SMALL_SATURATION);
                        matvalue = 0;
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++) {
                            matvalue += CFbeta * theta * kro_derv/(muo*Bo) * K_local[cj] * (P_array[cj]);
                        }
                        ierr = MatSetValues(K, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                        if (KPC != K) {
                            ierr = MatSetValues(KPC, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        jj++;
                    }
                }
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,P_array[i],Pb_array[i],bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
            ierr = FracDInterpolateUsingAnalyticalModel(&phi,P_array[i],PETSC_NULL,PETSC_NULL,bag->ppties.PhiData,PETSC_NULL);CHKERRQ(ierr);
            matvalue = -1. * (1./dt) * 1./CFalpha * scale * bag->CVFEface.elemVolume * phi/Bo;
            ierr = MatSetValues(K, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }

    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
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
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDQo(&Qo,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            ierr = FracDQo(&Qo1,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l]+bag->SMALL_SATURATION,Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            dervQo = (Qo1-Qo)/bag->SMALL_SATURATION;
            matvalue =  -1. * theta * bag->epD.phi[l] * dervQo;
            ierr = MatSetValues(K, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }

    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,K_local);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    for(i = 0; i < cvfacesize; i++)  free(CVfacecoords[i]);
    for(i = 0; i < 2; i++) {
        free(facecentroid[i]);
        free(edgecoords[i]);
    }
    free(facecentroid);
    free(wellcoords);
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Permeability);
    
    /*
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOP2.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOPPCU2.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(KPC,viewer);CHKERRQ(ierr);
    */
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDdRo_dSg"
extern PetscErrorCode FracDdRo_dSg(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *K_local,matvalue;
    PetscInt       d,i,j,k,ii,jj,kk,l,c,cj,w;
    PetscInt       pt,rpt,pt1,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability,scale;
    Vec            local_perm,local_CV;
    PetscScalar    *Perm_array=NULL,*CV_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords,*wellcoords;
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,localPbh;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL;
    PetscScalar    *P1_array=NULL,*Sw1_array=NULL,*Sg1_array=NULL;
    PetscReal      dt,theta,effectiveCellPerm;
    PetscReal      G,phi,fpress,fbubblepress;
    PetscReal      Bo,muo,kro,kro1,Qo,Qo1,dervQo,kro_derv,kr_check,kr[2],kr1[2];
    PetscReal      CFbeta, CFalpha, CFgamma;
    PetscErrorCode      (*FracDProjectFaceCoordinateDimensions)(PetscReal**,PetscReal**, PetscInt, PetscInt) = NULL;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    switch (bag->dim) {
        case 2:
        FracDProjectFaceCoordinateDimensions = FracD1DProjectFaceCoordinateDimensions;
        break;
        case 3:
        FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensions;
        break;
    }
    dt = bag->timevalue;
    theta = bag->theta;
    
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,2,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexScalNode,&globalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    edgecentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    facecentroid = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    edgecoords = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    for(i = 0; i < 2; i++)
    {
        facecentroid[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
        edgecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    if(bag->CVFEface.dim == 2)
    {
        gStart = 1;
        gStart  = fStart;   gEnd  = fEnd;
        ggStart = cStart;   ggEnd = cEnd;
    }
    if(bag->CVFEface.dim == 3)
    {
        gStart  = eStart;   gEnd  = eEnd;
        ggStart = fStart;   ggEnd = fEnd;
    }
    cvfacesize = bag->CVFEface.nodes;
    scale = bag->CVFEface.scale;
    CVfacecoords = (PetscReal **)malloc(cvfacesize * sizeof(PetscReal *));
    for(i = 0; i < cvfacesize; i++)
    {
        CVfacecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc3(ncol,&cols,nrow,&rows,ncol*nrow,&K_local);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
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
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
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
            cellcentroid[i] = 0;
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][i] = coord_array[i+j*bag->CVFEface.dim];
                cellcentroid[i] += coord_array[i+j*bag->CVFEface.dim];
            }
            cellcentroid[i] = cellcentroid[i]/bag->CVFEface.elemnodes;
        }
        for(k = 0; k < numclpts; k++){
            pt = closurept[2*k];
            if(pt >= gStart && pt < gEnd){
                ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                for(l = 0; l < bag->CVFEface.dim; l++){
                    edgecoords[0][l] = ecoord_array[l];
                    edgecoords[1][l] = ecoord_array[l+bag->dim];
                    edgecentroid[l] = 1/2.*(ecoord_array[l]+ecoord_array[l+bag->dim]);
                }
                if(bag->dim == 3){
                    d = 0;
                    ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                    for(kk = 0; kk < rnumclpts; kk++){
                        rpt = rclosurept[2*kk];
                        if(rpt >= ggStart && rpt < ggEnd){
                            for(ii = 0; ii < numclpts; ii++){
                                if(rpt == closurept[2*ii]){
                                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    for(l = 0; l < bag->CVFEface.dim; l++){
                                        facecentroid[d][l] = 0.;
                                        for(jj = 0; jj < bag->elD.nodes; jj++){
                                            facecentroid[d][l] += fcoord_array[l+jj*bag->dim];
                                        }
                                        facecentroid[d][l] = 1./bag->elD.nodes * facecentroid[d][l];
                                    }
                                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    d++;
                                }
                            }
                        }
                    }
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = facecentroid[0][i];
                        CVfacecoords[2][i] = edgecentroid[i];
                        CVfacecoords[3][i] = facecentroid[1][i];
                    }
                    ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                }
                else{
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = edgecentroid[i];
                    }
                }
                ierr = bag->FracDCreateCVFEFace(coords, CVfacecoords, &bag->CVFEface);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                //                Upwinding scheme is implemented below
                fpress = 0;
                fbubblepress = 0;
                for(i = 0; i < bag->CVFEface.elemnodes; i++){
                    fpress += P_array[i]*bag->CVFEface.phi[i];
                    fbubblepress += Pb_array[i]*bag->CVFEface.phi[i];
                }
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[0],Sw1_array[0],Sg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[1],Sw1_array[1],Sg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr1[0],Sw1_array[0],Sg1_array[0]+bag->SMALL_SATURATION,PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr1[1],Sw1_array[1],Sg1_array[1]+bag->SMALL_SATURATION,PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,fpress,fbubblepress,bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.OilPVTData.FracDUpDateViscosity(&muo,fpress,fbubblepress,bag->ppties.OilPVTData.mu_TableData,bag->ppties.OilPVTData.mu_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                jj = 0;
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    kr_check = 0;
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = PetscSectionGetOffset(globalSection, rpt1, &goffset);CHKERRQ(ierr);
                        goffset = goffset < 0 ? -(goffset+1):goffset;
                        rows[0] = goffset;
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Permeability, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++)   kr_check += K_local[cj] * (P_array[cj]);
                        if(kr_check > 0){
                            kro = kr[jj];
                            kro1 = kr1[jj];
                        }
                        else{
                            kro = kr[(jj+1)%2];
                            kro1 = kr1[(jj+1)%2];
                        }
                        kro_derv = (kro1-kro)/(bag->SMALL_SATURATION);
                        matvalue = 0;
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++) {
                            matvalue += CFbeta * theta * kro_derv/(muo*Bo) * K_local[cj] * (P_array[cj]);
                        }
                        ierr = MatSetValues(K, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                        if (KPC != K) {
                            ierr = MatSetValues(KPC, 1, rows, 1, rows, &matvalue, ADD_VALUES);CHKERRQ(ierr);
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        jj++;
                    }
                }
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,P_array[i],Pb_array[i],bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
            ierr = FracDInterpolateUsingAnalyticalModel(&phi,P_array[i],PETSC_NULL,PETSC_NULL,bag->ppties.PhiData,PETSC_NULL);CHKERRQ(ierr);
            matvalue = -1. * (1./dt) * 1./CFalpha * scale * bag->CVFEface.elemVolume * phi/Bo;
            ierr = MatSetValues(K, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[i], 1, &cols[i], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
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
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDQo(&Qo,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            ierr = FracDQo(&Qo1,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l]+bag->SMALL_SATURATION,Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            dervQo = (Qo1-Qo)/(bag->SMALL_SATURATION);
            matvalue =  -1. * theta * bag->epD.phi[l] * dervQo;
            ierr = MatSetValues(K, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[l], 1, &cols[l], &matvalue, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,K_local);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    for(i = 0; i < cvfacesize; i++)  free(CVfacecoords[i]);
    for(i = 0; i < 2; i++) {
        free(facecentroid[i]);
        free(edgecoords[i]);
    }
    free(facecentroid);
    free(wellcoords);
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Permeability);
    
    /*
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOP3.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOPPCU3.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(KPC,viewer);CHKERRQ(ierr);
    */
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDdRo_dPbh"
extern PetscErrorCode FracDdRo_dPbh(void *user, Mat K, Mat KPC, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *matvalue=NULL;
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
    Vec            localP,localPb,localPcow,localPcog,localSw,localSg,local_CV,localPbh;
    PetscScalar    *P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL;
    PetscReal      G,effectiveCellPerm;
    PetscReal      Qo,Qo1,dervQo;
    PetscReal      dt,theta;
    PetscReal      CFbeta, CFalpha, CFgamma;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    dt = bag->timevalue;
    theta = bag->theta;
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
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
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
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
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
            bag->well[w].re = 1./PETSC_PI * PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDQo(&Qo,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            ierr = FracDQo(&Qo1,bag->well[w],G,Pbh_array[w]+bag->SMALL_PRESSURE,P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            dervQo = (Qo1-Qo)/(bag->SMALL_SATURATION);
            matvalue[l] =  -1. * theta * bag->epD.phi[l] * dervQo;
        }
        ierr = MatSetValues(K, ncol, cols, 1, &w, matvalue, ADD_VALUES);CHKERRQ(ierr);
        if (KPC != K) {
            ierr = MatSetValues(KPC, ncol, cols, 1, &w, matvalue, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
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
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree3(cols,rows,matvalue);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    free(wellcoords);
    free(Permeability);
    
    /*
     PetscViewer viewer;
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixOP4.txt",&viewer);CHKERRQ(ierr);
     ierr = MatView(K,viewer);CHKERRQ(ierr);
    */
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDRo"
extern PetscErrorCode FracDRo(void *user, Vec Ro, Vec P, Vec Sw, Vec Sg, Vec Pbh)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscReal      *K_local;
    PetscInt       d,i,j,k,ii,jj,kk,l,c,w;
    PetscInt       pt,rpt,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Permeability,scale;
    Vec            local_perm,local_phi,local_CV;
    PetscScalar    *Perm_array=NULL,*Phi_array=NULL,*CV_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords,*wellcoords;;
    Vec            localResidual,localP,localoP,localPb,olocalPb,localPcow,localPcog,localoPcow,localoPcog,localSw,localoSw,localSg,localoSg,localPbh,localoPbh,localQP;
    PetscScalar    *Residual_array=NULL,*Residual1_array=NULL,*P_array=NULL,*Pb_array=NULL,*Pcow_array=NULL,*Pcog_array=NULL,*Sw_array=NULL,*Sg_array=NULL,*Pbh_array=NULL,*QP_array=NULL;
    PetscScalar    *oP_array=NULL,*oPb_array=NULL,*oPcow_array=NULL,*oPcog_array=NULL,*oSw_array=NULL,*oSg_array=NULL,*oPbh_array=NULL;
    PetscScalar    *P1_array=NULL,*Sw1_array=NULL,*Sg1_array=NULL,*oP1_array=NULL,*oSw1_array=NULL,*oSg1_array=NULL;
    PetscInt       numValues,numValues1,cj,li,QPsize;
    PetscReal      dt,theta,effectiveCellPerm,G;
    PetscReal      Bo,muo,phi,kro,Qo,kr_check,kr[2];
    PetscReal      Bo_o,muo_o,phi_o,kro_o,Qo_o,kr_check_o,kr_o[2];
    PetscReal      lambda,fpress,lambda_o,fpress_o, fbubblepress,fbubblepress_o;
    PetscReal      CFbeta, CFalpha, CFgamma;
    PetscErrorCode      (*FracDProjectFaceCoordinateDimensions)(PetscReal**,PetscReal**, PetscInt, PetscInt) = NULL;
    
    PetscFunctionBegin;
    ierr = VecSet(Ro,0.);CHKERRQ(ierr);
    switch (bag->dim) {
        case 2:
            FracDProjectFaceCoordinateDimensions = FracD1DProjectFaceCoordinateDimensions;
            break;
        case 3:
            FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensions;
            break;
    }
    dt = bag->timevalue;
    theta = bag->theta;
    phi = phi_o = 1.;
    CFbeta = bag->ConversionFactorBeta;
    CFgamma = bag->ConversionFactorGamma;
    CFalpha = bag->ConversionFactorAlpha;
    
    ierr = DMGetCoordinatesLocal(bag->plexScalNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,1,&fStart,&fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalNode,2,&eStart,&eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matvecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matscalSection);CHKERRQ(ierr);
    
    Permeability = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    wellcoords = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    edgecentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    coords = (PetscReal **)malloc(bag->CVFEface.elemnodes * sizeof(PetscReal *));
    for(i = 0; i < bag->CVFEface.elemnodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    facecentroid = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    edgecoords = (PetscReal **)malloc(2 * sizeof(PetscReal *));
    for(i = 0; i < 2; i++)
    {
        facecentroid[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
        edgecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    if(bag->CVFEface.dim == 2)
    {
        gStart = 1;
        gStart  = fStart;   gEnd  = fEnd;
        ggStart = cStart;   ggEnd = cEnd;
    }
    if(bag->CVFEface.dim == 3)
    {
        gStart  = eStart;   gEnd  = eEnd;
        ggStart = fStart;   ggEnd = fEnd;
    }
    cvfacesize = bag->CVFEface.nodes;
    scale = bag->CVFEface.scale;
    CVfacecoords = (PetscReal **)malloc(cvfacesize * sizeof(PetscReal *));
    for(i = 0; i < cvfacesize; i++)
    {
        CVfacecoords[i] = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc1(ncol*nrow,&K_local);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.perm,INSERT_VALUES,local_perm);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localoP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oP,INSERT_VALUES,localoP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oP,INSERT_VALUES,localoP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pb,INSERT_VALUES,localPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&olocalPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oPb,INSERT_VALUES,olocalPb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oPb,INSERT_VALUES,olocalPb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcow,INSERT_VALUES,localPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localoPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oPcow,INSERT_VALUES,localoPcow);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oPcow,INSERT_VALUES,localoPcow);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.Pcog,INSERT_VALUES,localPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localoPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oPcog,INSERT_VALUES,localoPcog);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oPcog,INSERT_VALUES,localoPcog);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,Sw,INSERT_VALUES,localSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localoSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oSg,INSERT_VALUES,localoSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oSg,INSERT_VALUES,localoSg);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localoSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oSw,INSERT_VALUES,localoSw);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oSw,INSERT_VALUES,localoSw);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localQP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.QP,INSERT_VALUES,localQP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.QP,INSERT_VALUES,localQP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->ppties.dualCellVolume,INSERT_VALUES,local_CV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,Pbh,INSERT_VALUES,localPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localoPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->WellRedun,bag->fields.oPbh,INSERT_VALUES,localoPbh);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->WellRedun,bag->fields.oPbh,INSERT_VALUES,localoPbh);CHKERRQ(ierr);
    ierr = VecGetArray(localoPbh,&oPbh_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    
    ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localResidual, vStart, &numValues, NULL);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localResidual, cStart, &numValues1, NULL);CHKERRQ(ierr);
    
    ierr = DMGetWorkArray(bag->plexScalNode, numValues, MPIU_SCALAR, &Residual_array);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexScalNode, numValues1, MPIU_SCALAR, &Residual1_array);CHKERRQ(ierr);
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoP, c, NULL, &oP_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, olocalPb, c, NULL, &oPb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoPcow, c, NULL, &oPcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoPcog, c, NULL, &oPcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSg, c, NULL, &oSg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSw, c, NULL, &oSw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localQP, c, &QPsize, &QP_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Permeability[i] = Perm_array[i];
        bag->ppties.PhiData[0] = Phi_array[0];
        for(i = 0; i < bag->CVFEface.dim; i++){
            cellcentroid[i] = 0;
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                coords[j][i] = coord_array[i+j*bag->CVFEface.dim];
                cellcentroid[i] += coord_array[i+j*bag->CVFEface.dim];
            }
            cellcentroid[i] = cellcentroid[i]/bag->CVFEface.elemnodes;
        }
        for(k = 0; k < numclpts; k++){
            pt = closurept[2*k];
            if(pt >= gStart && pt < gEnd){
                ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
                for(l = 0; l < bag->CVFEface.dim; l++){
                    edgecoords[0][l] = ecoord_array[l];
                    edgecoords[1][l] = ecoord_array[l+bag->dim];
                    edgecentroid[l] = 1/2.*(ecoord_array[l]+ecoord_array[l+bag->dim]);
                }
                if(bag->dim == 3){
                    d = 0;
                    ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                    for(kk = 0; kk < rnumclpts; kk++){
                        rpt = rclosurept[2*kk];
                        if(rpt >= ggStart && rpt < ggEnd){
                            for(ii = 0; ii < numclpts; ii++){
                                if(rpt == closurept[2*ii]){
                                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    for(l = 0; l < bag->CVFEface.dim; l++){
                                        facecentroid[d][l] = 0.;
                                        for(jj = 0; jj < bag->elD.nodes; jj++){
                                            facecentroid[d][l] += fcoord_array[l+jj*bag->dim];
                                        }
                                        facecentroid[d][l] = 1./bag->elD.nodes * facecentroid[d][l];
                                    }
                                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt, NULL, &fcoord_array);CHKERRQ(ierr);
                                    d++;
                                }
                            }
                        }
                    }
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = facecentroid[0][i];
                        CVfacecoords[2][i] = edgecentroid[i];
                        CVfacecoords[3][i] = facecentroid[1][i];
                    }
                    ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_FALSE, &rnumclpts, &rclosurept);CHKERRQ(ierr);
                }
                else{
                    for(i = 0; i < bag->CVFEface.dim; i++){
                        CVfacecoords[0][i] = cellcentroid[i];
                        CVfacecoords[1][i] = edgecentroid[i];
                    }
                }
                ierr = bag->FracDCreateCVFEFace(coords, CVfacecoords, &bag->CVFEface);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoP, pt, NULL, &oP1_array);CHKERRQ(ierr);
                
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSg, pt, NULL, &oSg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSw, pt, NULL, &oSw1_array);CHKERRQ(ierr);
                //              Upwinding scheme is implemented below
                fpress = fpress_o = 0;
                fbubblepress = fbubblepress_o = 0;
                for(i = 0; i < bag->CVFEface.elemnodes; i++){
                    fpress += P_array[i]*bag->CVFEface.phi[i];
                    fpress_o += oP_array[i]*bag->CVFEface.phi[i];
                    fbubblepress += Pb_array[i]*bag->CVFEface.phi[i];
                    fbubblepress_o += oPb_array[i]*bag->CVFEface.phi[i];
                }
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[0],Sw1_array[0],Sg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr[1],Sw1_array[1],Sg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr =  bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,fpress,fbubblepress,bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr =  bag->ppties.OilPVTData.FracDUpDateViscosity(&muo,fpress,PETSC_NULL,bag->ppties.OilPVTData.mu_TableData,bag->ppties.OilPVTData.mu_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr_o[0],oSw1_array[0],oSg1_array[0],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr = bag->ppties.RelPermData.FracDUpDateKro(&kr_o[1],oSw1_array[1],oSg1_array[1],PETSC_NULL,bag->ppties.RelPermData.Krow_TableData,bag->ppties.RelPermData.Krog_TableData,bag->ppties.RelPermData.Krw_TableData,bag->ppties.RelPermData.stone_model_data,bag->ppties.RelPermData.numwaterdatarow,bag->ppties.RelPermData.numgasdatarow);CHKERRQ(ierr);
                ierr =  bag->ppties.OilPVTData.FracDUpDateFVF(&Bo_o,fpress_o,PETSC_NULL,bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                ierr =  bag->ppties.OilPVTData.FracDUpDateViscosity(&muo_o,fpress_o,fbubblepress_o,bag->ppties.OilPVTData.mu_TableData,bag->ppties.OilPVTData.mu_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
                
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                jj = 0;
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    kr_check = kr_check_o = 0;
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Permeability, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for (cj = 0; cj < bag->CVFEface.elemnodes; cj++){
                            kr_check += K_local[cj] * (P_array[cj]);
                            kr_check_o += K_local[cj] * (oP_array[cj]);
                        }
                        if(kr_check > 0) kro = kr[jj];
                        else    kro = kr[(jj+1)%2];
                        if(kr_check_o > 0) kro_o = kr_o[jj];
                        else    kro_o = kr_o[(jj+1)%2];
                        lambda = kro/(muo*Bo);
                        lambda_o = kro_o/(muo_o*Bo_o);
                        for (l = 0, li = 0; li < numValues; li++) {
                            Residual_array[li] = 0;
                            for (cj = 0; cj < bag->CVFEface.elemnodes; cj++, l++) {
                                Residual_array[li] += CFbeta * K_local[l] * (theta*lambda*(P_array[cj])+(1.-theta)*lambda_o*(oP_array[cj]));
                            }
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, localResidual, rpt1, Residual_array, ADD_ALL_VALUES);CHKERRQ(ierr);
                        jj++;
                    }
                }
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, pt, NULL, &Sg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSg, pt, NULL, &oSg1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, pt, NULL, &Sw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSw, pt, NULL, &oSw1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, pt, NULL, &P1_array);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoP, pt, NULL, &oP1_array);CHKERRQ(ierr);
                
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            Residual1_array[i] = 0;
            ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo,P_array[i],Pb_array[i],bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
            ierr = FracDInterpolateUsingAnalyticalModel(&phi,P_array[i],PETSC_NULL,PETSC_NULL,bag->ppties.PhiData,PETSC_NULL);CHKERRQ(ierr);

            ierr = bag->ppties.OilPVTData.FracDUpDateFVF(&Bo_o,oP_array[i],oPb_array[i],bag->ppties.OilPVTData.B_TableData,bag->ppties.OilPVTData.B_ModelData,bag->ppties.OilPVTData.numdatarow);CHKERRQ(ierr);
            ierr = FracDInterpolateUsingAnalyticalModel(&phi_o,oP_array[i],PETSC_NULL,PETSC_NULL,bag->ppties.PhiData,PETSC_NULL);CHKERRQ(ierr);
            Residual1_array[i] = 1./CFalpha * scale * bag->CVFEface.elemVolume * ((1./dt) * ((phi*(1.-Sw_array[i]-Sg_array[i])/Bo)-(phi_o*(1.-oSw_array[i]-oSg_array[i])/Bo_o))-QP_array[i]);
        }
        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, localResidual, c, Residual1_array, ADD_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoP, c, NULL, &oP_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, olocalPb, c, NULL, &oPb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoPcow, c, NULL, &oPcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoPcog, c, NULL, &oPcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSg, c, NULL, &oSg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSw, c, NULL, &oSw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localQP, c, &QPsize, &QP_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }

    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++) {
        w = bag->WellinMeshData.WellInfo[i][0];
        c = bag->WellinMeshData.WellInfo[i][1];
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, olocalPb, c, NULL, &oPb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoPcow, c, NULL, &oPcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoPcog, c, NULL, &oPcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoP, c, NULL, &oP_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSg, c, NULL, &oSg_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localoSw, c, NULL, &oSw_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localQP, c, &QPsize, &QP_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        for(ii = 0; ii < bag->CVFEface.dim; ii++){
            for(j = 0; j < bag->CVFEface.elemnodes; j++){
                Permeability[ii] = Perm_array[ii];
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
            bag->well[w].re = 1/PETSC_PI*PetscPowScalar(CV_array[l],(1./bag->dim));
            G = 2*PETSC_PI*CFbeta*effectiveCellPerm*bag->well[w].h/(PetscLogReal(bag->well[w].re/bag->well[w].rw)+bag->well[w].sk);
            ierr = FracDQo(&Qo,bag->well[w],G,Pbh_array[w],P_array[l],Pb_array[l],Sw_array[l],Sg_array[l],Pcow_array[l],Pcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            ierr = FracDQo(&Qo_o,bag->well[w],G,oPbh_array[w],oP_array[l],oPb_array[l],oSw_array[l],oSg_array[l],oPcow_array[l],oPcog_array[l],bag->ppties.WaterPVTData,bag->ppties.OilPVTData,bag->ppties.GasPVTData,bag->ppties.RelPermData);CHKERRQ(ierr);
            Residual1_array[l] = -1. * bag->epD.phi[l] * (theta * Qo + (1-theta) * Qo_o);
        }
        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, localResidual, c, Residual1_array, ADD_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_perm, c, NULL, &  Perm_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPb, c, NULL, &Pb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, olocalPb, c, NULL, &oPb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcow, c, NULL, &Pcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoPcow, c, NULL, &oPcow_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localPcog, c, NULL, &Pcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoPcog, c, NULL, &oPcog_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoP, c, NULL, &oP_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSg, c, NULL, &Sg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSg, c, NULL, &oSg_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localSw, c, NULL, &Sw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localoSw, c, NULL, &oSw_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localQP, c, &QPsize, &QP_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, local_CV, c, NULL, &CV_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
    }

    ierr = DMRestoreWorkArray(bag->plexScalNode, numValues1, PETSC_SCALAR, &Residual1_array);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(bag->plexScalNode, numValues, PETSC_SCALAR, &Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->plexScalNode,localResidual,ADD_VALUES,Ro);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexScalNode,localResidual,ADD_VALUES,Ro);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localResidual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localoP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&olocalPb);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localoPcow);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localoPcog);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localoSg);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localoSw);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localQP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_CV);CHKERRQ(ierr);
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(localoPbh,&oPbh_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localoPbh);CHKERRQ(ierr);
    //    Will need to figure out a way for neumann condition for different phases
    if (bag->FlowFluxBC.hasLabel) {
//        ierr = FracDResidualApplyCVFENeumannBC(Ro,dt,&bag->FlowFluxBC,&bag->elD,bag->FracDCreateDMinusOneFEElement,FracDProjectFaceCoordinateDimensions);CHKERRQ(ierr);
    }
//    ierr = FracDResidualApplyDirichletBC(Ro,P,&bag->PBC);CHKERRQ(ierr);
    ierr = PetscFree(K_local);CHKERRQ(ierr);
    for(i = 0; i < bag->nodes; i++)  free(coords[i]);
    free(coords);
    for(i = 0; i < cvfacesize; i++)  free(CVfacecoords[i]);
    for(i = 0; i < 2; i++) {
        free(facecentroid[i]);
        free(edgecoords[i]);
    }
    free(facecentroid);
    free(wellcoords);
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Permeability);
/*
     PetscViewer viewer;
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Ro.txt",&viewer);CHKERRQ(ierr);
     ierr = VecView(Ro,viewer);CHKERRQ(ierr);
     ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Ro1.txt",&viewer);CHKERRQ(ierr);
     ierr = VecView(Sw,viewer);CHKERRQ(ierr);
    */
    PetscFunctionReturn(0);
}
