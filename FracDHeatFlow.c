/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */

#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDFlow.h"
#include "FracDHeatFlow.h"


#undef __FUNCT__
#define __FUNCT__ "FracDTJacobian"
extern PetscErrorCode FracDTJacobian(SNES snesT,Vec T,Mat K, Mat KPC, void *user)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscInt       coldofIndex,goffset,*rows=NULL,*cols=NULL;
    PetscReal      *K_local,*K1_local;
    PetscInt       d,i,j,k,ii,jj,kk,l,c;
    PetscInt       pt,rpt,pt1,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   globalSection,vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Conductivity,scale;
    Vec            local_cond,local_phi,local_Cp,local_rhos,localV,localVo;
    PetscScalar    *V_array=NULL,*Vo_array=NULL;
    PetscScalar    *Cond_array=NULL,*Phi_array=NULL,*Cp_array=NULL,*Rho_s_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords;
    PetscReal      dt, theta,cvfevolume, rhoCp, rhoCw;
    PetscScalar    one = 1.0;

    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    dt = bag->timevalue;
    theta = bag->theta;
    rhoCw = bag->ppties.rho_w*bag->ppties.Cpw;
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

    Conductivity = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
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
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.cond,INSERT_VALUES,local_cond);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.cond,INSERT_VALUES,local_cond);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalCell,&local_Cp);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.Cp,INSERT_VALUES,local_Cp);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.Cp,INSERT_VALUES,local_Cp);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_rhos);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.rhos,INSERT_VALUES,local_rhos);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.rhos,INSERT_VALUES,local_rhos);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecNode,&localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,bag->fields.V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,bag->fields.V,INSERT_VALUES,localV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecNode,&localVo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,bag->fields.oV,INSERT_VALUES,localVo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,bag->fields.oV,INSERT_VALUES,localVo);CHKERRQ(ierr);
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_cond, c, NULL, &Cond_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_Cp, c, NULL, &Cp_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_rhos, c, NULL, &Rho_s_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localV, c, NULL, &V_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localVo, c, NULL, &Vo_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Conductivity[i] = Cond_array[i];
        rhoCp = Rho_s_array[0]*Cp_array[0]*(1-Phi_array[0])+rhoCw*Phi_array[0];
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
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = PetscSectionGetOffset(globalSection, rpt1, &goffset);CHKERRQ(ierr);
                        goffset = goffset < 0 ? -(goffset+1):goffset;
                        rows[0] = goffset;
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Conductivity, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);

                        ierr = FracDAdvectiveFluxMatrixLocal(K1_local, rhoCw, V_array, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for(l = 0; l < nrow*ncol; l++) {
                            K_local[l] = dt * theta * K_local[l];
                            K1_local[l] = dt * theta * K1_local[l];
                        }
                        ierr = MatSetValues(K, nrow, rows, ncol, cols, K_local, ADD_VALUES);CHKERRQ(ierr);
                        ierr = MatSetValues(K, nrow, rows, ncol, cols, K1_local, ADD_VALUES);CHKERRQ(ierr);
                        if (KPC != K) {
                            ierr = MatSetValues(KPC, nrow, rows, ncol, cols, K_local, ADD_VALUES);CHKERRQ(ierr);
                            ierr = MatSetValues(KPC, nrow, rows, ncol, cols, K1_local, ADD_VALUES);CHKERRQ(ierr);
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                    }
                }
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            cvfevolume = rhoCp*scale * bag->CVFEface.elemVolume;
            ierr = MatSetValues(K, 1, &cols[i], 1, &cols[i], &cvfevolume, ADD_VALUES);CHKERRQ(ierr);
            if (KPC != K) {
                ierr = MatSetValues(KPC, 1, &cols[i], 1, &cols[i], &cvfevolume, ADD_VALUES);CHKERRQ(ierr);
            }
        }
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_cond, c, NULL, &Cond_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_Cp, c, NULL, &Cp_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_rhos, c, NULL, &Rho_s_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localV, c, NULL, &V_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localVo, c, NULL, &Vo_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_Cp);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_rhos);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localV);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localVo);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyDirichletBC(bag->plexScalNode, K,&bag->TBC,one);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = FracDMatrixApplyDirichletBC(bag->plexScalNode, KPC,&bag->TBC,one);CHKERRQ(ierr);
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
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Conductivity);
    
/*
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixT1.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixTPCU1.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(KPC,viewer);CHKERRQ(ierr);

    
    Vec A;
    ierr = DMCreateGlobalVector(bag->plexScalNode, &A);CHKERRQ(ierr);
    ierr = VecSet(A,0.0);CHKERRQ(ierr);
    
    MatMult(K,T,A);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"TRes1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(A,viewer);CHKERRQ(ierr);
    */
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDTResidual"
extern PetscErrorCode FracDTResidual(SNES snesT,Vec T,Vec residual,void *user)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       ncol = bag->CVFEface.elemnodes;
    PetscInt       nrow = 1;
    PetscReal      *K_local,*K1_local;
    PetscInt       d,i,j,k,ii,jj,kk,l,c;
    PetscInt       pt,rpt,rpt1;
    PetscInt       vStart,vEnd,eStart,eEnd,fStart,fEnd,cStart,cEnd,gStart,gEnd,ggStart,ggEnd;
    PetscSection   vecSection,scalSection,cordSection,matvecSection,matscalSection;
    PetscReal      *Conductivity,scale;
    Vec            local_cond,local_phi,local_Cp,local_rhos;
    PetscScalar    *Cond_array=NULL,*Phi_array=NULL,*Cp_array=NULL,*Rho_s_array=NULL;
    PetscInt       numclpts,rnumclpts,rnumclpts1,cvfacesize;
    Vec            coordinates;
    PetscScalar    *coord_array=NULL,*ecoord_array=NULL,*fcoord_array=NULL,*vcoord_array=NULL;
    PetscReal      **coords, **edgecoords;
    PetscInt       *closurept=NULL,*rclosurept=NULL,*rclosurept1=NULL;
    PetscReal      *cellcentroid,*edgecentroid,**facecentroid,**CVfacecoords;
    Vec            localResidual,localT,localTo,localV,localVo,localQT;
    PetscScalar    *Residual_array=NULL,*Residual1_array=NULL,*T_array=NULL,*To_array=NULL,*V_array=NULL,*Vo_array=NULL,*QT_array=NULL;
    PetscInt       numValues,numValues1,cj,li,QTsize;
    PetscReal      dt,theta,rhoCp,rhoCw;

    PetscFunctionBegin;
    ierr = VecSet(residual,0.);CHKERRQ(ierr);
    dt = bag->timevalue;
    theta = bag->theta;
    rhoCw = bag->ppties.rho_w*bag->ppties.Cpw;
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
    
    Conductivity = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
    cellcentroid = (PetscReal *)malloc(bag->CVFEface.dim * sizeof(PetscReal));
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
    ierr = PetscMalloc2(ncol*nrow,&K_local,ncol*nrow,&K1_local);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecCell,bag->ppties.cond,INSERT_VALUES,local_cond);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecCell,bag->ppties.cond,INSERT_VALUES,local_cond);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.phi,INSERT_VALUES,local_phi);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_Cp);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.Cp,INSERT_VALUES,local_Cp);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.Cp,INSERT_VALUES,local_Cp);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_rhos);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.rhos,INSERT_VALUES,local_rhos);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.rhos,INSERT_VALUES,local_rhos);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,T,INSERT_VALUES,localT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,T,INSERT_VALUES,localT);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localTo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.oT,INSERT_VALUES,localTo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.oT,INSERT_VALUES,localTo);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexVecNode,&localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,bag->fields.V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,bag->fields.V,INSERT_VALUES,localV);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecNode,&localVo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,bag->fields.oV,INSERT_VALUES,localVo);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,bag->fields.oV,INSERT_VALUES,localVo);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localQT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.QT,INSERT_VALUES,localQT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.QT,INSERT_VALUES,localQT);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);

    ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localResidual, vStart, &numValues, NULL);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localResidual, cStart, &numValues1, NULL);CHKERRQ(ierr);
    
    ierr = DMGetWorkArray(bag->plexScalNode, numValues, MPIU_SCALAR, &Residual_array);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexScalNode, numValues1, MPIU_SCALAR, &Residual1_array);CHKERRQ(ierr);

    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(bag->plexVecCell, matvecSection, local_cond, c, NULL, &Cond_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_Cp, c, NULL, &Cp_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matscalSection, local_rhos, c, NULL, &Rho_s_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localT, c, NULL, &T_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localTo, c, NULL, &To_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localV, c, NULL, &V_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localVo, c, NULL, &Vo_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localQT, c, &QTsize, &QT_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < bag->CVFEface.dim; i++)  Conductivity[i] = Cond_array[i];
        rhoCp = Rho_s_array[0]*Cp_array[0]*(1-Phi_array[0])+rhoCw*Phi_array[0];
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
                ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                for(i = 0; i < rnumclpts1; i++){
                    rpt1 = rclosurept1[2*i];
                    if(rpt1 >= vStart && rpt1 < vEnd){
                        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = FracDDiffusiveFluxMatrixLocal(K_local, Conductivity, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        ierr = FracDAdvectiveFluxMatrixLocal(K1_local, rhoCw, V_array, vcoord_array, &bag->CVFEface);CHKERRQ(ierr);
                        for (l = 0, li = 0; li < numValues; li++) {
                            Residual_array[li] = 0;
                            for (cj = 0; cj < bag->CVFEface.elemnodes; cj++, l++) {
                                Residual_array[li] += dt * (K_local[l] + K1_local[l]) * (theta*T_array[cj]+(1.-theta)*To_array[cj]);
                            }
                        }
                        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, rpt1, NULL, &vcoord_array);CHKERRQ(ierr);
                        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, localResidual, rpt1, Residual_array, ADD_ALL_VALUES);CHKERRQ(ierr);
                    }
                }
                ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, pt, PETSC_TRUE, &rnumclpts1, &rclosurept1);CHKERRQ(ierr);
                ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pt, NULL, &ecoord_array);CHKERRQ(ierr);
            }
        }
        for(i = 0; i < bag->CVFEface.elemnodes; i++){
            Residual1_array[i] = 0;
            Residual1_array[i] = rhoCp*scale * bag->CVFEface.elemVolume * (T_array[i]-To_array[i]-dt*QT_array[i]);
        }
        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, localResidual, c, Residual1_array, ADD_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecCell, matvecSection, local_cond, c, NULL, &Cond_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_phi, c, NULL, &Phi_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_Cp, c, NULL, &Cp_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matscalSection, local_rhos, c, NULL, &Rho_s_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localT, c, NULL, &T_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localTo, c, NULL, &To_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localV, c, NULL, &V_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localVo, c, NULL, &Vo_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localQT, c, &QTsize, &QT_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(bag->plexScalNode, numValues1, PETSC_SCALAR, &Residual1_array);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(bag->plexScalNode, numValues, PETSC_SCALAR, &Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->plexScalNode,localResidual,ADD_VALUES,residual);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexScalNode,localResidual,ADD_VALUES,residual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localResidual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_phi);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_Cp);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_rhos);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localQT);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localT);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localTo);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localV);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localVo);CHKERRQ(ierr);
    if (bag->HeatFluxBC.hasLabel) {
        ierr = FracDResidualApplyCVFENeumannBC(residual,dt,&bag->HeatFluxBC,&bag->elD,bag->FracDCreateDMinusOneFEElement,bag->FracDProjectFaceCoordinateDimensions);CHKERRQ(ierr);
    }
    ierr = FracDResidualApplyDirichletBC(residual,T,&bag->TBC);CHKERRQ(ierr);
    ierr = PetscFree2(K_local,K1_local);CHKERRQ(ierr);
    
    for(i = 0; i < bag->CVFEface.elemnodes; i++)  free(coords[i]);
    free(coords);
    for(i = 0; i < cvfacesize; i++)  free(CVfacecoords[i]);
    for(i = 0; i < 2; i++) {
        free(facecentroid[i]);
        free(edgecoords[i]);
    }
    free(facecentroid);
    free(edgecoords);
    free(CVfacecoords);
    free(cellcentroid);
    free(edgecentroid);
    free(Conductivity);
   
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"TResidual1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(residual,viewer);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"TSolution1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(T,viewer);CHKERRQ(ierr);
/*        PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"TResidual1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(residual,viewer);CHKERRQ(ierr);
    
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"TSolution1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(T,viewer);CHKERRQ(ierr);
 */
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDSolveT"
extern PetscErrorCode FracDSolveT(AppCtx *bag)
{
    PetscErrorCode          ierr;
    SNESConvergedReason     reason;
    PetscInt                its;
    
    PetscFunctionBegin;
    ierr = SNESSolve(bag->snesT,PETSC_NULL,bag->fields.T);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(bag->snesT,&reason);CHKERRQ(ierr);
    if (reason < 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[ERROR] snesT diverged with reason %d\n",(int)reason);CHKERRQ(ierr);
    } else {
        ierr = SNESGetIterationNumber(bag->snesT,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"      snesT converged in %d iterations %d.\n",(int)its,(int)reason);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}
