/*
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */
#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#include "FracDBasic.h"
#include "FracDMechanics.h"

/*
 http://www.personal.soton.ac.uk/jav/soton/HELM/workbooks/workbook_27/27_4_changing_coords.pdf
 http://mathforum.org/library/drmath/view/52092.html
 
 MatSetValuesLocal goes hand in hand with local indices obtained using DMGetDefaultSection . On the other hand,  MatSetValues goes hand in hand with global indices obtained using DMGetDefaultGlobalSection.
 */
#undef __FUNCT__
#define __FUNCT__ "FracDUJacobian"
extern PetscErrorCode FracDUJacobian(SNES snesU,Vec U,Mat K, Mat KPC, void *user)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       nrow = bag->eD.dim * bag->eD.nodes;
    PetscReal      *K_local;
    PetscInt       i,j,k,l;
    PetscInt       vStart,vEnd,cStart,cEnd,c;
    PetscSection   globalSection,vecSection,cordSection,matSection;
    PetscReal      E, nu;
    Vec            local_E, local_nu;
    PetscScalar    *E_array = NULL, *nu_array = NULL;
    Vec            coordinates;
    PetscScalar    *coord_array = NULL;
    PetscReal      **coords;
    PetscInt       goffset,pt,numclpts,*rows=NULL,*closurept=NULL;
    PetscInt       edofIndex;
    PetscScalar    one = 1.0;
    
    PetscFunctionBegin;
    ierr = MatZeroEntries(K);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatZeroEntries(KPC);CHKERRQ(ierr);
    }
    ierr = PetscMalloc2(nrow,&rows,nrow*nrow,&K_local);CHKERRQ(ierr);
    coords = (PetscReal **)malloc(bag->eD.nodes * sizeof(PetscReal *));
    for(i = 0; i < bag->eD.nodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->eD.dim * sizeof(PetscReal));
    }
    ierr = DMGetCoordinatesLocal(bag->plexVecNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexVecNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexVecNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexVecNode,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matSection);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(bag->plexVecNode,&globalSection);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalCell,&local_E);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.E,INSERT_VALUES,local_E);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.E,INSERT_VALUES,local_E);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_nu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.nu,INSERT_VALUES,local_nu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.nu,INSERT_VALUES,local_nu);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c){
        edofIndex = 0;
        ierr = DMPlexGetTransitiveClosure(bag->plexVecNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_E, c, NULL, &E_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_nu, c, NULL, &nu_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        E = E_array[0]; nu = nu_array[0];
        for(l = 0, i = 0; i < bag->eD.nodes; i++){
            for(j = 0; j < bag->eD.dim; j++, l++){
                coords[i][j] = coord_array[l];
            }
        }
        ierr = bag->FracDCreateDFEElement(coords, &bag->eD);CHKERRQ(ierr);
        ierr = bag->FracDElasticityStiffnessMatrixLocal(K_local, E, nu, &bag->eD);CHKERRQ(ierr);
        for(k = 0; k < numclpts; k++){
            pt = closurept[2*k];
            if(pt >= vStart && pt < vEnd){
                ierr = PetscSectionGetOffset(globalSection, pt, &goffset);CHKERRQ(ierr);
                goffset = goffset < 0 ? -(goffset+1):goffset;
                for(i = 0; i < bag->eD.dim; i++){
                    rows[edofIndex] = goffset + i;
                    edofIndex++;
                }
            }
        }
        ierr = MatSetValues(K, nrow, rows, nrow, rows, K_local, ADD_VALUES);CHKERRQ(ierr);
        if (KPC != K) {
            ierr = MatSetValues(KPC, nrow, rows, nrow, rows, K_local, ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(bag->plexVecNode, c, PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_E, c, NULL, &E_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_nu, c, NULL, &nu_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_E);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_nu);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = FracDMatrixApplyDirichletBC(bag->plexVecNode, K,&bag->UBC,one);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (KPC != K) {
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        
        ierr = FracDMatrixApplyDirichletBC(bag->plexVecNode, KPC,&bag->UBC,one);CHKERRQ(ierr);
        
        ierr = MatAssemblyBegin(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = PetscFree2(rows,K_local);CHKERRQ(ierr);
    for(i = 0; i < bag->eD.nodes; i++){
        free(coords[i]);
    }
    free(coords);
    
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Matrix.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(K,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"MatrixPCU.txt",&viewer);CHKERRQ(ierr);
    ierr = MatView(KPC,viewer);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDUResidual"
extern PetscErrorCode FracDUResidual(SNES snesU,Vec U,Vec residual,void *user)
{
    PetscErrorCode ierr;
    AppCtx         *bag=(AppCtx*)user;
    DM             cdm;
    PetscInt       nrow = bag->eD.dim * bag->eD.nodes;
    PetscReal      *residual_local,*K_local;
    PetscInt       i,j,l,li,lj,ci,cj,numValues;
    PetscInt       cStart, cEnd, c;
    PetscSection   vecSection, scalSection, cordSection, matSection;
    PetscReal      E, nu, beta, alpha;
    Vec            local_E, local_nu, local_beta, local_alpha;
    PetscScalar    *E_array = NULL, *nu_array = NULL, *beta_array = NULL, *alpha_array = NULL;
    Vec            localResidual, localU, localP, localT, localFb;
    PetscScalar    *Residual_array = NULL, *U_array = NULL, *T_array = NULL, *P_array = NULL, *Fb_array = NULL;
    Vec            coordinates;
    PetscScalar    *coord_array = NULL;
    PetscReal      **coords;

    PetscFunctionBegin;
    ierr = VecSet(residual,0.);CHKERRQ(ierr);
    coords = (PetscReal **)malloc(bag->eD.nodes * sizeof(PetscReal *));
    for(i = 0; i < bag->eD.nodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->eD.dim * sizeof(PetscReal));
    }
    ierr = PetscMalloc2(nrow,&residual_local,nrow*nrow,&K_local);CHKERRQ(ierr);

    ierr = DMGetCoordinatesLocal(bag->plexVecNode,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexVecNode, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMPlexGetHeightStratum(bag->plexVecNode,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecNode,&vecSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&matSection);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalCell,&local_E);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.E,INSERT_VALUES,local_E);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.E,INSERT_VALUES,local_E);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_nu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.nu,INSERT_VALUES,local_nu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.nu,INSERT_VALUES,local_nu);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexScalCell,&local_beta);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.beta,INSERT_VALUES,local_beta);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.beta,INSERT_VALUES,local_beta);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_alpha);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,bag->ppties.alpha,INSERT_VALUES,local_alpha);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,bag->ppties.alpha,INSERT_VALUES,local_alpha);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.P,INSERT_VALUES,localP);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.P,INSERT_VALUES,localP);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&localT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalNode,bag->fields.T,INSERT_VALUES,localT);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalNode,bag->fields.T,INSERT_VALUES,localT);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexVecNode,&localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,U,INSERT_VALUES,localU);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,U,INSERT_VALUES,localU);CHKERRQ(ierr);

    ierr = DMGetLocalVector(bag->plexVecNode,&localFb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexVecNode,bag->fields.Fb,INSERT_VALUES,localFb);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexVecNode,bag->fields.Fb,INSERT_VALUES,localFb);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecNode,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);

    ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localResidual, cStart, &numValues, NULL);CHKERRQ(ierr);//My guess is that this does not need to be restored since the array is null (no array is obtained)
 
    ierr = DMGetWorkArray(bag->plexVecNode, numValues, MPIU_SCALAR, &Residual_array);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c){
        for(l = 0; l < numValues; l++)  Residual_array[l] = 0;
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_E, c, NULL, &E_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_nu, c, NULL, &nu_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_beta, c, NULL, &beta_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalCell, matSection, local_alpha, c, NULL, &alpha_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localU, c, NULL, &U_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, localT, c, NULL, &T_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(bag->plexVecNode, vecSection, localFb, c, NULL, &Fb_array);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
        E = E_array[0]; nu = nu_array[0], beta = beta_array[0], alpha = alpha_array[0];
        for(l = 0, i = 0; i < bag->eD.nodes; i++){
            for(j = 0; j < bag->eD.dim; j++, l++){
                coords[i][j] = coord_array[l];
            }
        }
        ierr = bag->FracDCreateDFEElement(coords, &bag->eD);
        ierr = bag->FracDElasticityStiffnessMatrixLocal(K_local, E, nu, &bag->eD);CHKERRQ(ierr);
        for (l = 0, li = 0, i = 0; i < bag->eD.nodes; i++) {
            for (ci = 0; ci < bag->eD.dim; ci++, li++) {
                for (lj = 0, j = 0; j < bag->eD.nodes; j++) {
                    for (cj = 0; cj < bag->eD.dim; cj++, lj++, l++) {
                        Residual_array[li] += K_local[l] * U_array[lj];
                    }
                }
            }
        }
        ierr = FracDThermoPoroelastic_local(Residual_array, P_array, T_array, E, nu, beta, alpha, &bag->eD);//residual_array is updated inside this function
        ierr = FracDApplyBodyForce_local(Residual_array, Fb_array, bag->ppties.g, bag->ppties.rho, &bag->eD);
        ierr = DMPlexVecSetClosure(bag->plexVecNode, vecSection, localResidual, c, Residual_array, ADD_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_E, c, NULL, &E_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_nu, c, NULL, &nu_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_beta, c, NULL, &beta_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalCell, matSection, local_alpha, c, NULL, &alpha_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localU, c, NULL, &U_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localP, c, NULL, &P_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexScalNode, scalSection, localT, c, NULL, &T_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(bag->plexVecNode, vecSection, localFb, c, NULL, &Fb_array);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, NULL, &coord_array);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(bag->plexVecNode, numValues, PETSC_SCALAR, &Residual_array);CHKERRQ(ierr);
 
    ierr = DMLocalToGlobalBegin(bag->plexVecNode,localResidual,ADD_VALUES,residual);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexVecNode,localResidual,ADD_VALUES,residual);CHKERRQ(ierr);
    
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localResidual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_E);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_nu);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_beta);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_alpha);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localP);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&localT);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localU);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecNode,&localFb);CHKERRQ(ierr);
    if (bag->TractionBC.hasLabel) {
        ierr = FracDResidualApplyFENeumannBC(residual, &bag->TractionBC, &bag->elD, bag->FracDCreateDMinusOneFEElement, bag->FracDProjectFaceCoordinateDimensions);CHKERRQ(ierr);
    }
    ierr = FracDResidualApplyDirichletBC(residual,U,&bag->UBC);CHKERRQ(ierr);
    

    
    
    ierr = PetscFree2(residual_local,K_local);CHKERRQ(ierr);

    for(i = 0; i < bag->eD.nodes; i++)  free(coords[i]);
    free(coords);
    
  
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Residual1.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(residual,viewer);CHKERRQ(ierr);
    
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"USolution.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Residual.txt",&viewer);CHKERRQ(ierr);
    ierr = VecView(residual,viewer);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDApplyBodyForce_local"
extern PetscErrorCode FracDApplyBodyForce_local(PetscScalar *residual_local,PetscScalar *Fb_array, PetscReal *grav, PetscReal rho, FracDFEElement *e)
{
    PetscInt       i,j,c,l,g;
    PetscScalar    f_elem[e->ng][e->dim];
    PetscFunctionBegin;
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->dim; i++){
            f_elem[g][i] = 0;
            for(j = 0; j < e->nodes; j++){
                f_elem[g][i] += Fb_array[i+j*e->dim]*e->phi[g][j];
            }
        }
    }
    for (l = 0, i = 0; i < e->nodes; i++) {
        for (c = 0; c < e->dim; c++, l++) {
            for (g = 0; g < e->ng; g++){
            residual_local[l] -= e->detJ[g] * (f_elem[g][c] + rho*grav[c]) * e->phi[g][i] * e->weight[g];
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDThermoPoroelastic_local"
extern PetscErrorCode FracDThermoPoroelastic_local(PetscScalar *residual_local,PetscScalar *P_array, PetscScalar *T_array, PetscReal E, PetscReal nu, PetscReal beta, PetscReal alpha, FracDFEElement *e)
{
    PetscInt       i,c,l,g;
    PetscReal      mu, lambda;
    PetscScalar    P_elem[e->ng], T_elem[e->ng];
    
    PetscFunctionBegin;
    lambda = E*nu/((1+nu)*(1-2*nu));
    mu = E/(2*(1+nu));
    for(g = 0; g < e->ng; g++){
        P_elem[g] = T_elem[g] = 0;
    }
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->nodes; i++){
            P_elem[g] += P_array[i]*e->phi[g][i];
            T_elem[g] += T_array[i]*e->phi[g][i];
        }
    }
    for (l = 0, i = 0; i < e->nodes; i++) {
        for (c = 0; c < e->dim; c++, l++) {
            for (g = 0; g < e->ng; g++){
                residual_local[l] -= e->detJ[g] * (beta*P_elem[g]+alpha*(3*lambda+2*mu)*T_elem[g]) * e->dphi[g][c][i] * e->weight[g];
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDElasticity3D_local"
extern PetscErrorCode FracDElasticity3D_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e)
{
    
    PetscInt       i,j,ci,cj,l,g;
    PetscReal      mu, lambda, coeff = 0;
    
    PetscFunctionBegin;
    lambda = E*nu/((1+nu)*(1-2*nu));
    mu = E/(2*(1+nu));
    for (l = 0; l < e->dim * e->nodes * e->dim * e->nodes; l++) {
        K_local[l] = 0;
    }
    for (l = 0,i = 0; i < e->nodes; i++) {
        for (ci = 0; ci < e->dim; ci++) {
            for (j = 0; j < e->nodes; j++) {
                for (cj = 0; cj < e->dim; cj++, l++) {
                    for (g = 0; g < e->ng; g++){
                    coeff = lambda*e->dphi[g][ci][i]*e->dphi[g][cj][j];
                    coeff += mu*e->dphi[g][cj][i]*e->dphi[g][ci][j];
                    if(ci == cj){
                        coeff += mu*(e->dphi[g][0][i]*e->dphi[g][0][j]+e->dphi[g][1][i]*e->dphi[g][1][j]+e->dphi[g][2][i]*e->dphi[g][2][j]);
                    }
                    K_local[l] += e->detJ[g] * coeff * e->weight[g];
                    }
                }
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDElasticity2DPlaneStrain_local"
extern PetscErrorCode FracDElasticity2DPlaneStrain_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e)
{
    
    PetscInt       i,j,ci,cj,l,g;
    PetscReal      mu, lambda, coeff = 0;
    
    PetscFunctionBegin;
    lambda = E*nu/((1+nu)*(1-2*nu));
    mu = E/(2*(1+nu));
    for (l = 0; l < e->dim * e->nodes * e->dim * e->nodes; l++) {
        K_local[l] = 0;
    }
    for (l = 0,i = 0; i < e->nodes; i++) {
        for (ci = 0; ci < e->dim; ci++) {
            for (j = 0; j < e->nodes; j++) {
                for (cj = 0; cj < e->dim; cj++, l++) {
                    for(g = 0; g < e->ng; g++){
                    coeff = lambda*e->dphi[g][ci][i]*e->dphi[g][cj][j];
                    coeff += mu*e->dphi[g][cj][i]*e->dphi[g][ci][j];
                    if(ci == cj){
                        coeff += mu*(e->dphi[g][0][i]*e->dphi[g][0][j]+e->dphi[g][1][i]*e->dphi[g][1][j]);
                    }
                    K_local[l] += e->detJ[g] * coeff * e->weight[g];
                    }
                }
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDElasticity2DPlaneStress_local"
extern PetscErrorCode FracDElasticity2DPlaneStress_local(PetscReal *K_local,PetscReal E,PetscReal nu,FracDFEElement *e)
{

    PetscInt       i,j,ci,cj,l,g;
    PetscReal      mu, lambda, coeff = 0;
    
    PetscFunctionBegin;
    lambda = E*nu/((1+nu)*(1-2*nu));
    mu = E/(2*(1+nu));
    for (l = 0; l < e->dim * e->nodes * e->dim * e->nodes; l++) {
        K_local[l] = 0;
    }
    for (l = 0,i = 0; i < e->nodes; i++) {
        for (ci = 0; ci < e->dim; ci++) {
            for (j = 0; j < e->nodes; j++) {
                for (cj = 0; cj < e->dim; cj++, l++) {
                    for(g = 0; g < e->ng; g++){
                    coeff = 2*lambda*mu*e->dphi[g][ci][i]*e->dphi[g][cj][j];
                    coeff += mu*(lambda+2*mu)*e->dphi[g][(ci+1)%e->dim][i]*e->dphi[g][(cj+1)%e->dim][j];
                    if(ci == cj){
                        coeff += (2*lambda*mu+4*mu*mu)*e->dphi[g][ci][i]*e->dphi[g][ci][j];
                    }
                    K_local[l] += 1/(lambda+2*mu) * e->detJ[g] * coeff * e->weight[g];
                    }
                }
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDSolveU"
extern PetscErrorCode FracDSolveU(AppCtx *bag)
{
    PetscErrorCode      ierr;
    SNESConvergedReason  reason;
    PetscInt            its;
    
    PetscFunctionBegin;
    ierr = SNESSolve(bag->snesU,PETSC_NULL,bag->fields.U);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(bag->snesU,&reason);CHKERRQ(ierr);
    if (reason < 0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"[ERROR] snesU diverged with reason %d\n",(int)reason);CHKERRQ(ierr);
    } else {
        ierr = SNESGetIterationNumber(bag->snesU,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"      snesU converged in %d iterations %d.\n",(int)its,(int)reason);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}
