#define FracDStandardFEElement
#include "petsc.h"
#include "FracDFiniteElement.h"

//////    http://isml.ecm.uwa.edu.au/ISML/Publication/pdfs/2014auglimillerJBEefficient.pdf
//See structural analysis with the finite element method by Eugenio Onate
//http://matveichev.blogspot.com/2013/12/building-hexagonal-meshes-with-gmsh.html


#undef __FUNCT__
#define __FUNCT__ "FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative"
extern PetscErrorCode FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(PetscReal *phi, PetscReal dphi[2][4],PetscReal *eta)
{
    PetscReal   phi_i[2],phi_j[2],dphi_i[2],dphi_j[2];
    
    PetscFunctionBegin;
    phi_i[0] = 0.5*(1-eta[0]);
    phi_i[1] = 0.5*(1+eta[0]);
    phi_j[0] = 0.5*(1-eta[1]);
    phi_j[1] = 0.5*(1+eta[1]);
    
    phi[0] = phi_i[0]*phi_j[0];
    phi[1] = phi_i[0]*phi_j[1];
    phi[2] = phi_i[1]*phi_j[1];
    phi[3] = phi_i[1]*phi_j[0];
    
    dphi_i[0] = dphi_j[0] = -0.5;
    dphi_i[1] = dphi_j[1] = 0.5;
    
    dphi[0][0] = dphi_i[0]*phi_j[0];
    dphi[0][1] = dphi_i[0]*phi_j[1];
    dphi[0][2] = dphi_i[1]*phi_j[1];
    dphi[0][3] = dphi_i[1]*phi_j[0];
    
    dphi[1][0] = phi_i[0]*dphi_j[0];
    dphi[1][1] = phi_i[0]*dphi_j[1];
    dphi[1][2] = phi_i[1]*dphi_j[1];
    dphi[1][3] = phi_i[1]*dphi_j[0];
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative"
extern PetscErrorCode FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(PetscReal *phi, PetscReal dphi[3][8],PetscReal *eta)
{
    PetscReal   phi_i[2],phi_j[2],phi_k[2],dphi_i[2],dphi_j[2],dphi_k[2];
    
    PetscFunctionBegin;
    phi_i[0] = 0.5*(1-eta[0]);
    phi_i[1] = 0.5*(1+eta[0]);
    phi_j[0] = 0.5*(1-eta[1]);
    phi_j[1] = 0.5*(1+eta[1]);
    phi_k[0] = 0.5*(1-eta[2]);
    phi_k[1] = 0.5*(1+eta[2]);
    
    phi[0] = phi_i[0]*phi_j[0]*phi_k[0];
    phi[1] = phi_i[0]*phi_j[1]*phi_k[0];
    phi[2] = phi_i[1]*phi_j[1]*phi_k[0];
    phi[3] = phi_i[1]*phi_j[0]*phi_k[0];
    phi[4] = phi_i[0]*phi_j[0]*phi_k[1];
    phi[5] = phi_i[1]*phi_j[0]*phi_k[1];
    phi[6] = phi_i[1]*phi_j[1]*phi_k[1];
    phi[7] = phi_i[0]*phi_j[1]*phi_k[1];
    
    dphi_i[0] = dphi_j[0]  = dphi_k[0] = -0.5;
    dphi_i[1] = dphi_j[1]  = dphi_k[1] = 0.5;
    
    dphi[0][0] = dphi_i[0]*phi_j[0]*phi_k[0];
    dphi[0][1] = dphi_i[0]*phi_j[1]*phi_k[0];
    dphi[0][2] = dphi_i[1]*phi_j[1]*phi_k[0];
    dphi[0][3] = dphi_i[1]*phi_j[0]*phi_k[0];
    dphi[0][4] = dphi_i[0]*phi_j[0]*phi_k[1];
    dphi[0][5] = dphi_i[1]*phi_j[0]*phi_k[1];
    dphi[0][6] = dphi_i[1]*phi_j[1]*phi_k[1];
    dphi[0][7] = dphi_i[0]*phi_j[1]*phi_k[1];
    
    dphi[1][0] = phi_i[0]*dphi_j[0]*phi_k[0];
    dphi[1][1] = phi_i[0]*dphi_j[1]*phi_k[0];
    dphi[1][2] = phi_i[1]*dphi_j[1]*phi_k[0];
    dphi[1][3] = phi_i[1]*dphi_j[0]*phi_k[0];
    dphi[1][4] = phi_i[0]*dphi_j[0]*phi_k[1];
    dphi[1][5] = phi_i[1]*dphi_j[0]*phi_k[1];
    dphi[1][6] = phi_i[1]*dphi_j[1]*phi_k[1];
    dphi[1][7] = phi_i[0]*dphi_j[1]*phi_k[1];
    
    dphi[2][0] = phi_i[0]*phi_j[0]*dphi_k[0];
    dphi[2][1] = phi_i[0]*phi_j[1]*dphi_k[0];
    dphi[2][2] = phi_i[1]*phi_j[1]*dphi_k[0];
    dphi[2][3] = phi_i[1]*phi_j[0]*dphi_k[0];
    dphi[2][4] = phi_i[0]*phi_j[0]*dphi_k[1];
    dphi[2][5] = phi_i[1]*phi_j[0]*dphi_k[1];
    dphi[2][6] = phi_i[1]*phi_j[1]*dphi_k[1];
    dphi[2][7] = phi_i[0]*phi_j[1]*dphi_k[1];
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DHexahedralLocalToGlobalCoordinate"
extern PetscErrorCode FracD3DHexahedralLocalToGlobalCoordinate(PetscReal *X, PetscReal *eta, PetscReal **elemcoords)
{
    PetscInt    i,l;
    PetscReal   phi[8],phi_i[2],phi_j[2],phi_k[2];
    
    PetscFunctionBegin;
    phi_i[0] = 0.5*(1-eta[0]);
    phi_i[1] = 0.5*(1+eta[0]);
    phi_j[0] = 0.5*(1-eta[1]);
    phi_j[1] = 0.5*(1+eta[1]);
    phi_k[0] = 0.5*(1-eta[2]);
    phi_k[1] = 0.5*(1+eta[2]);
    
    phi[0] = phi_i[0]*phi_j[0]*phi_k[0];
    phi[1] = phi_i[0]*phi_j[1]*phi_k[0];
    phi[2] = phi_i[1]*phi_j[1]*phi_k[0];
    phi[3] = phi_i[1]*phi_j[0]*phi_k[0];
    phi[4] = phi_i[0]*phi_j[0]*phi_k[1];
    phi[5] = phi_i[1]*phi_j[0]*phi_k[1];
    phi[6] = phi_i[1]*phi_j[1]*phi_k[1];
    phi[7] = phi_i[0]*phi_j[1]*phi_k[1];
    
    for (i = 0; i < 3; i++) {
        X[i] = 0.;
        for (l = 0; l < 8; l++) {
            X[i] += phi[l]*elemcoords[l][i];
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DQuadrilateralLocalToGlobalCoordinate"
extern PetscErrorCode FracD2DQuadrilateralLocalToGlobalCoordinate(PetscReal *X, PetscReal *eta, PetscReal **elemcoords)
{
    PetscInt        i,l;
    PetscReal       phi[4],dphi[2][4];
    PetscErrorCode  ierr;

    PetscFunctionBegin;
    ierr = FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(phi,dphi,eta);CHKERRQ(ierr);

    for (i = 0; i < 2; i++) {
        X[i] = 0.;
        for (l = 0; l < 4; l++) {
            X[i] += phi[l]*elemcoords[l][i];
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDComputeLocalFECoordinates"
extern PetscErrorCode FracDComputeLocalFECoordinates(PetscReal *eta_new, PetscReal *Xp, PetscReal *L,PetscReal **elemcoords, PetscInt dim, PetscErrorCode (*FracDComputeLocalToGlobalCoordinate)(PetscReal*, PetscReal*, PetscReal**))
{
    PetscInt        i,k = 0;
    PetscReal       Xk[dim],eta_old[dim],error = 1e+10;
    PetscErrorCode  ierr;
    
    PetscFunctionBegin;
        while(error > 1e-6){
            k++;
        ierr = FracDComputeLocalToGlobalCoordinate(Xk,eta_new,elemcoords);CHKERRQ(ierr);
        for(i = 0; i < dim; i++){
            eta_old[i] = eta_new[i];
            eta_new[i] = eta_old[i] + (Xp[i]-Xk[i])/L[i];
        }
        error = 0.;
        for(i = 0; i < dim; i++)  {
            error += PetscAbs(eta_new[i]-eta_old[i]);
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDPointFEElementCreate"
extern PetscErrorCode FracDPointFEElementCreate(PetscInt dim, FracDPointFEElement *e, FracDElementType elementType)
{
    PetscFunctionBegin;
    switch (dim) {
        case 1:
            e->dim       = dim;
            e->nodes     = 2;
            break;
        case 2:
            e->dim       = dim;
            if(elementType == TRIANGLE)         e->nodes     = 3;
            if(elementType == QUADRILATERAL)
            {
                e->nodes     = 4;
                e->FracDLocalToGlobalCoordinate = FracD2DQuadrilateralLocalToGlobalCoordinate;
            }
            if(elementType == TETRAHEDRAL)      e->nodes     = 3;
            if(elementType == HEXAHEDRAL)       e->nodes     = 4;
            break;
        case 3:
            e->dim       = dim;
            if(elementType == TETRAHEDRAL)      e->nodes     = 4;
            if(elementType == HEXAHEDRAL)
            {
                e->nodes     = 8;
                e->FracDLocalToGlobalCoordinate = FracD3DHexahedralLocalToGlobalCoordinate;
            }
            break;
        default:
            SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i %s\n",dim,__FUNCT__);
            break;
    }
    e->phi = (PetscReal *) malloc(e->nodes*sizeof(PetscReal));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DTetrahedralPointElementFE"
extern PetscErrorCode FracD3DTetrahedralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e)
{
    PetscInt            i,j,k,l;
    PetscReal           dphi[e->dim][e->nodes],J[e->dim][e->dim],detJ;
    PetscReal           **localcoords=NULL;
    
    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(e->nodes * sizeof(PetscReal *));
    for(i = 0; i < e->nodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(e->dim * sizeof(PetscReal));
    }
    dphi[0][0] = -1;dphi[0][1] = 1;dphi[0][2] = 0;dphi[0][3] = 0;
    dphi[1][0] = -1;dphi[1][1] = 0;dphi[1][2] = 1;dphi[1][3] = 0;
    dphi[2][0] = -1;dphi[2][1] = 0;dphi[2][2] = 0;dphi[2][3] = 1;
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < e->nodes; k++){
                J[i][j] += dphi[i][k]*elemcoords[k][j];
            }
        }
    }
    detJ = 0;
    for(i = 0; i < e->dim; i++){
        detJ += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
    }
    e->Volume = 1/6.*PetscAbs(detJ);
    for(l = 0; l < e->nodes; l++){
        for(i = 0; i < e->nodes; i++){
            for(j = 0; j < e->dim; j++){
                localcoords[i][j] = elemcoords[i][j];
            }
        }
        for(i = 0; i < e->dim; i++) localcoords[l][i] = coords[i];
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                J[i][j] = 0.;
                for(k = 0; k < e->nodes; k++){
                    J[i][j] += dphi[i][k]*localcoords[k][j];
                }
            }
        }
        e->phi[l] = 0;
        for(i = 0; i < e->dim; i++){
            e->phi[l] += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
        }
        e->phi[l] = 1/6.*PetscAbs(e->phi[l])/e->Volume;
    }
    for(i = 0; i < e->nodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DHexahedralPointElementFE"
extern PetscErrorCode FracD3DHexahedralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e)
{
    PetscInt            i,j,k;
    PetscReal           dphi[3][8],eta_array[3],L[3],centroid[e->dim];
    PetscReal           J[e->dim][e->dim];
    PetscErrorCode      ierr;
    
    PetscFunctionBegin;
    for(j = 0; j < e->dim; j++){
        centroid[j] = eta_array[j] = 0;
        for(i = 0; i < e->nodes; i++){
            centroid[j] += elemcoords[i][j];
        }
        centroid[j] = centroid[j]/e->nodes;
    }
    ierr = FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(e->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < e->nodes; k++){
                J[i][j] += dphi[i][k]*elemcoords[k][j];
            }
        }
    }
    for(i = 0; i < e->dim; i++) {
        L[i] = J[i][i];
        eta_array[i] = (coords[i]-centroid[i])/L[i];
    }
    ierr = FracDComputeLocalFECoordinates(eta_array,coords,L,elemcoords, e->dim, e->FracDLocalToGlobalCoordinate);CHKERRQ(ierr);
    ierr = FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(e->phi,dphi,eta_array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracD2DTrianglePointElementFE"
extern PetscErrorCode FracD2DTrianglePointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e)
{
    PetscInt            i,j,k,l;
    PetscReal           dphi[e->dim][e->nodes],J[e->dim][e->dim],detJ;
    PetscReal           **localcoords;
    
    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(e->nodes * sizeof(PetscReal *));
    for(i = 0; i < e->nodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(e->dim * sizeof(PetscReal));
    }
    dphi[0][0] = -1;dphi[0][1] = 1;dphi[0][2] = 0;
    dphi[1][0] = -1;dphi[1][1] = 0;dphi[1][2] = 1;
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < e->nodes; k++){
                J[i][j] += dphi[i][k]*elemcoords[k][j];
            }
        }
    }
    detJ = 0;
    detJ = J[1][1]*J[0][0]-J[0][1]*J[1][0];
    e->Volume = 1/2.*PetscAbs(detJ);
    for(l = 0; l < e->nodes; l++){
        for(i = 0; i < e->nodes; i++){
            for(j = 0; j < e->dim; j++){
                localcoords[i][j] = elemcoords[i][j];
            }
        }
        for(i = 0; i < e->dim; i++) localcoords[l][i] = coords[i];
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                J[i][j] = 0.;
                for(k = 0; k < e->nodes; k++){
                    J[i][j] += dphi[i][k]*localcoords[k][j];
                }
            }
        }
        e->phi[l] = 1/2.*(J[1][1]*J[0][0]-J[0][1]*J[1][0])/e->Volume;
        e->phi[l] = PetscAbs(e->phi[l]);
    }
    for(i = 0; i < e->nodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DQuadrilateralPointElementFE"
extern PetscErrorCode FracD2DQuadrilateralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e)
{
    PetscInt            i,j,k;
    PetscReal           dphi[2][4],eta_array[2],L[2],centroid[e->dim];
    PetscReal           J[e->dim][e->dim];
    PetscErrorCode      ierr;

    PetscFunctionBegin;
    for(j = 0; j < e->dim; j++){
        centroid[j] = eta_array[j] = 0;
        for(i = 0; i < e->nodes; i++){
            centroid[j] += elemcoords[i][j];
        }
        centroid[j] = centroid[j]/e->nodes;
    }
    ierr = FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(e->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < e->nodes; k++){
                J[i][j] += dphi[i][k]*elemcoords[k][j];
            }
        }
    }
    for(i = 0; i < e->dim; i++) {
        L[i] = J[i][i];
        eta_array[i] = (coords[i]-centroid[i])/L[i];
    }
    ierr = FracDComputeLocalFECoordinates(eta_array,coords,L,elemcoords, e->dim, e->FracDLocalToGlobalCoordinate);CHKERRQ(ierr);
    ierr = FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative( e->phi,dphi,eta_array);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD1DPointElementFE"
extern PetscErrorCode FracD1DPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e)
{
    PetscInt            i;
    PetscFunctionBegin;
    e->phi[0] = (coords[0]-elemcoords[1][0])/(elemcoords[1][0]-elemcoords[0][0]);
    e->phi[1] = (coords[0]-elemcoords[0][0])/(elemcoords[1][0]-elemcoords[0][0]);
    for(i = 0; i < e->nodes; i++) {
        e->phi[i] = PetscAbs(e->phi[i]);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDPointFEElementDestroy"
extern PetscErrorCode FracDPointFEElementDestroy(FracDPointFEElement *e)
{
    PetscFunctionBegin;
    free(e->phi);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DReferenceTetrahedralElementFEDerivative"
extern PetscErrorCode FracD3DReferenceTetrahedralElementFEDerivative(FracDFEElement *e)
{
    PetscInt i,j,k;
    
    PetscFunctionBegin;
    e->weight[0]     = 1./6.;
    e->int_point[0][0]     = 1/4.;
    e->int_point[0][1]     = 1/4.;
    e->int_point[0][2]     = 1/4.;
    e->int_point[0][3]     = 1/4.;
    e->phi[0][1]     = e->int_point[0][1];
    e->phi[0][2]     = e->int_point[0][2];
    e->phi[0][3]     = e->int_point[0][3];
    e->phi[0][0]     = 1-e->phi[0][1]-e->phi[0][2]-e->phi[0][3];
    e->dphi_r[0][0][0] = -1;    e->dphi_r[0][0][1] = 1; e->dphi_r[0][0][2] = 0; e->dphi_r[0][0][3] = 0;
    e->dphi_r[0][1][0] = -1;    e->dphi_r[0][1][1] = 0; e->dphi_r[0][1][2] = 1; e->dphi_r[0][1][3] = 0;
    e->dphi_r[0][2][0] = -1;    e->dphi_r[0][2][1] = 0; e->dphi_r[0][2][2] = 0; e->dphi_r[0][2][3] = 1;
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->J[0][i][j] = 0.;
            e->invJ[0][i][j] = 0.;
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            for(k = 0; k < e->nodes; k++){
                e->J[0][i][j] += e->dphi_r[0][i][k]*e->nodecoords[k][j];
            }
        }
    }
    e->detJ[0] = 0;
    for(i = 0; i < e->dim; i++){
        e->detJ[0] += (e->J[0][0][i]*(e->J[0][1][(i+1)%3]*e->J[0][2][(i+2)%3] - e->J[0][1][(i+2)%3]*e->J[0][2][(i+1)%3]));
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->invJ[0][j][i] = 1./e->detJ[0]*((e->J[0][(i+1)%3][(j+1)%3]*e->J[0][(i+2)%3][(j+2)%3])-(e->J[0][(i+1)%3][(j+2)%3]*e->J[0][(i+2)%3][(j+1)%3]));
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->nodes; j++){
            e->dphi[0][i][j] = 0;
            for(k = 0; k < e->dim; k++){
                e->dphi[0][i][j] += e->invJ[0][i][k]*e->dphi_r[0][k][j];
            }
        }
    }
    e->detJ[0] = PetscAbs(e->detJ[0]);
    e->Volume = 1/6.*e->detJ[0];
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DTetrahedralElementFE"
extern PetscErrorCode FracD3DTetrahedralElementFE(PetscReal **coords, FracDFEElement *e)
{
    PetscInt i,j,k;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        for(j = 0; j < e->dim; j++){
            e->nodecoords[i][j] = coords[i][j];
        }
    }
    e->weight[0]     = 1./6.;
    e->int_point[0][0]     = 1/4.;
    e->int_point[0][1]     = 1/4.;
    e->int_point[0][2]     = 1/4.;
    e->int_point[0][3]     = 1/4.;
    e->phi[0][1]     = e->int_point[0][1];
    e->phi[0][2]     = e->int_point[0][2];
    e->phi[0][3]     = e->int_point[0][3];
    e->phi[0][0]     = 1-e->phi[0][1]-e->phi[0][2]-e->phi[0][3];
    e->dphi_r[0][0][0] = -1;    e->dphi_r[0][0][1] = 1; e->dphi_r[0][0][2] = 0; e->dphi_r[0][0][3] = 0;
    e->dphi_r[0][1][0] = -1;    e->dphi_r[0][1][1] = 0; e->dphi_r[0][1][2] = 1; e->dphi_r[0][1][3] = 0;
    e->dphi_r[0][2][0] = -1;    e->dphi_r[0][2][1] = 0; e->dphi_r[0][2][2] = 0; e->dphi_r[0][2][3] = 1;
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->J[0][i][j] = 0.;
            e->invJ[0][i][j] = 0.;
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            for(k = 0; k < e->nodes; k++){
                e->J[0][i][j] += e->dphi_r[0][i][k]*e->nodecoords[k][j];
            }
        }
    }
    e->detJ[0] = 0;
    for(i = 0; i < e->dim; i++){
        e->detJ[0] += (e->J[0][0][i]*(e->J[0][1][(i+1)%3]*e->J[0][2][(i+2)%3] - e->J[0][1][(i+2)%3]*e->J[0][2][(i+1)%3]));
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->invJ[0][j][i] = 1./e->detJ[0]*((e->J[0][(i+1)%3][(j+1)%3]*e->J[0][(i+2)%3][(j+2)%3])-(e->J[0][(i+1)%3][(j+2)%3]*e->J[0][(i+2)%3][(j+1)%3]));
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->nodes; j++){
            e->dphi[0][i][j] = 0;
            for(k = 0; k < e->dim; k++){
                e->dphi[0][i][j] += e->invJ[0][i][k]*e->dphi_r[0][k][j];
            }
        }
    }
    e->detJ[0] = PetscAbs(e->detJ[0]);
    e->Volume = 1/6.*e->detJ[0];
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DReferenceHexahedralElementFEDerivative"
extern PetscErrorCode FracD3DReferenceHexahedralElementFEDerivative(FracDFEElement *e)
{
    PetscInt    i,j,k,l,g,gi,gj,gk;
    PetscReal   dphi_i[e->ng][2],dphi_j[e->ng][2],dphi_k[e->ng][2];
    PetscReal   phi_i[e->ng][2],phi_j[e->ng][2],phi_k[e->ng][2];
    PetscReal   eta[2],nu[2],zeta[2];
    PetscReal   one_over_sqrt_three,tmp1,tmp2,tmp3,tmp4;
    PetscReal   tmp5a,tmp5b,tmp5c,tmp5d;
    
    PetscFunctionBegin;
    one_over_sqrt_three = PetscSqrtScalar(1./3.);
    eta[0] = nu[0] = zeta[0] = -one_over_sqrt_three;
    eta[1] = nu[1] = zeta[1] = one_over_sqrt_three;
    for(g = 0; g < 2; g++){
        phi_i[g][0] = 0.5*(1-eta[g]);
        phi_i[g][1] = 0.5*(1+eta[g]);
        phi_j[g][0] = 0.5*(1-nu[g]);
        phi_j[g][1] = 0.5*(1+nu[g]);
        phi_k[g][0] = 0.5*(1-zeta[g]);
        phi_k[g][1] = 0.5*(1+zeta[g]);
        
        dphi_i[g][0] = dphi_j[g][0] = dphi_k[g][0] = -0.5;
        dphi_i[g][1] = dphi_j[g][1] = dphi_k[g][1] = 0.5;
    }
    for (g = 0, gi = 0; gi < 2; gi++) {
        for (gj = 0; gj < 2; gj++) {
            for (gk = 0; gk < 2; gk++, g++) {
                e->detJ[g] = 0;
                e->weight[g]    = 1.;
                for (l = 0, i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        for (k = 0; k < 2; k++, l++) {
                            e->phi[g][l] = phi_i[gi][i]*phi_j[gj][j]*phi_k[gk][k];
                        }
                    }
                }
            }
        }
    }
    for (g = 0, gi = 0; gi < 2; gi++) {
        for (gj = 0; gj < 2; gj++) {
            for (gk = 0; gk < 2; gk++, g++) {
                for (l = 0, i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        for (k = 0; k < 2; k++, l++) {
                            e->dphi_r[g][0][l] = dphi_i[gi][i]*phi_j[gj][j]*phi_k[gk][k];
                            e->dphi_r[g][1][l] = phi_i[gi][i]*dphi_j[gj][j]*phi_k[gk][k];
                            e->dphi_r[g][2][l] = phi_i[gi][i]*phi_j[gj][j]*dphi_k[gk][k];
                        }
                    }
                }
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        tmp1 = e->phi[g][3];
        tmp2 = e->dphi_r[g][0][3];
        tmp3 = e->dphi_r[g][1][3];
        tmp4 = e->dphi_r[g][2][3];
        e->phi[g][3] = e->phi[g][2];
        e->dphi_r[g][0][3]   = e->dphi_r[g][0][2];
        e->dphi_r[g][1][3]   = e->dphi_r[g][1][2];
        e->dphi_r[g][2][3]   = e->dphi_r[g][2][2];
        e->phi[g][2] = tmp1;
        e->dphi_r[g][0][2] = tmp2;
        e->dphi_r[g][1][2] = tmp3;
        e->dphi_r[g][2][2] = tmp4;
        
        tmp5a = e->phi[g][5];
        tmp5b = e->dphi_r[g][0][5];
        tmp5c = e->dphi_r[g][1][5];
        tmp5d = e->dphi_r[g][2][5];
        
        e->phi[g][5] = e->phi[g][6];
        e->dphi_r[g][0][5]   = e->dphi_r[g][0][6];
        e->dphi_r[g][1][5]   = e->dphi_r[g][1][6];
        e->dphi_r[g][2][5]   = e->dphi_r[g][2][6];
        
        e->phi[g][6] = e->phi[g][7];
        e->dphi_r[g][0][6]   = e->dphi_r[g][0][7];
        e->dphi_r[g][1][6]   = e->dphi_r[g][1][7];
        e->dphi_r[g][2][6]   = e->dphi_r[g][2][7];
        
        e->phi[g][7] = tmp5a;
        e->dphi_r[g][0][7]   = tmp5b;
        e->dphi_r[g][1][7]   = tmp5c;
        e->dphi_r[g][2][7]   = tmp5d;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DHexahedralElementFE"
extern PetscErrorCode FracD3DHexahedralElementFE(PetscReal **coords, FracDFEElement *e)
{
    PetscInt    i,j,k,g;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        for(j = 0; j < e->dim; j++){
            e->nodecoords[i][j] = coords[i][j];
        }
    }
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                e->J[g][i][j] = 0.;
                e->invJ[g][i][j] = 0.;
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                for(k = 0; k < e->nodes; k++){
                    e->J[g][i][j] += e->dphi_r[g][i][k]*e->nodecoords[k][j];
                }
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        e->detJ[g] = 0;
        for(i = 0; i < e->dim; i++){
            e->detJ[g] += (e->J[g][0][i]*(e->J[g][1][(i+1)%3]*e->J[g][2][(i+2)%3] - e->J[g][1][(i+2)%3]*e->J[g][2][(i+1)%3]));
        }
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                e->invJ[g][j][i] = 1./e->detJ[g]*((e->J[g][(i+1)%3][(j+1)%3]*e->J[g][(i+2)%3][(j+2)%3])-(e->J[g][(i+1)%3][(j+2)%3]*e->J[g][(i+2)%3][(j+1)%3]));
            }
        }
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->nodes; j++){
                e->dphi[g][i][j] = 0;
                for(k = 0; k < e->dim; k++){
                    e->dphi[g][i][j] += e->invJ[g][i][k]*e->dphi_r[g][k][j];
                }
            }
        }
        e->detJ[g] = PetscAbs(e->detJ[g]);
    }
    e->Volume = 0.;
    for(g = 0; g < e->ng; g++){
        e->Volume += PetscAbs(e->detJ[g]) * e->weight[g];
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DReferenceTriangleElementFEDerivative"
extern PetscErrorCode FracD2DReferenceTriangleElementFEDerivative(FracDFEElement *e)
{
    PetscFunctionBegin;
    e->weight[0]    = 1./2;
    e->int_point[0][0]     = 1/3.;
    e->int_point[0][1]     = 1/3.;
    e->int_point[0][2]     = 1/3.;
    
    e->phi[0][1]     = e->int_point[0][1];
    e->phi[0][2]     = e->int_point[0][2];
    e->phi[0][0]     = 1-e->phi[0][1]-e->phi[0][2];
    
    e->dphi_r[0][0][0] = -1;e->dphi_r[0][0][1] = 1;e->dphi_r[0][0][2] = 0;
    e->dphi_r[0][1][0] = -1;e->dphi_r[0][1][1] = 0;e->dphi_r[0][1][2] = 1;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DTriangleElementFE"
extern PetscErrorCode FracD2DTriangleElementFE(PetscReal **coords, FracDFEElement *e)
{
    PetscInt    i,j,k;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        for(j = 0; j < e->dim; j++){
            e->nodecoords[i][j] = coords[i][j];
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->J[0][i][j] = 0.;
            e->invJ[0][i][j] = 0.;
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            for(k = 0; k < e->nodes; k++){
                e->J[0][i][j] += e->dphi_r[0][i][k]*e->nodecoords[k][j];
            }
        }
    }
    e->detJ[0] = 0;
    e->detJ[0] = e->J[0][1][1]*e->J[0][0][0]-e->J[0][0][1]*e->J[0][1][0];
    e->invJ[0][0][0] = 1./e->detJ[0]*e->J[0][1][1];
    e->invJ[0][0][1] = -1./e->detJ[0]*e->J[0][0][1];
    e->invJ[0][1][0] = -1./e->detJ[0]*e->J[0][1][0];
    e->invJ[0][1][1] = 1./e->detJ[0]*e->J[0][0][0];
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->nodes; j++){
            e->dphi[0][i][j] = 0;
            for(k = 0; k < e->dim; k++){
                e->dphi[0][i][j] += e->invJ[0][i][k]*e->dphi_r[0][k][j];
            }
        }
    }
    e->detJ[0] = PetscAbs(e->detJ[0]);
    e->Volume = 1/2.*e->detJ[0];
    PetscFunctionReturn(0);
}














#undef __FUNCT__
#define __FUNCT__ "FracD2DReferenceQuadrilateralElementFEDerivative"
extern PetscErrorCode FracD2DReferenceQuadrilateralElementFEDerivative(FracDFEElement *e)
{
    PetscInt    i,j,l,g,gi,gj;
    PetscReal   tmp1,tmp2,tmp3;
    PetscReal   phi_i[e->ng][2],phi_j[e->ng][2],eta[2],nu[2],one_over_sqrt_three,dphi_i[e->ng][2],dphi_j[e->ng][2];
    
    PetscFunctionBegin;

    one_over_sqrt_three = PetscSqrtScalar(1./3.);
    eta[0] = nu[0] = -one_over_sqrt_three;
    eta[1] = nu[1] = one_over_sqrt_three;
    for(g = 0; g < 2; g++){
        phi_i[g][0] = 0.5*(1-eta[g]);
        phi_i[g][1] = 0.5*(1+eta[g]);
        phi_j[g][0] = 0.5*(1-nu[g]);
        phi_j[g][1] = 0.5*(1+nu[g]);
        
        dphi_i[g][0] = dphi_j[g][0] = -0.5;
        dphi_i[g][1] = dphi_j[g][1] = 0.5;
    }
    for (g = 0, gi = 0; gi < 2; gi++) {
        for (gj = 0; gj < 2; gj++, g++) {
            e->detJ[g] = 0;
            e->weight[g]    = 1.;
            for (l = 0, i = 0; i < 2; i++) {
                for (j = 0; j < 2; j++, l++) {
                    e->phi[g][l] = phi_i[gi][i]*phi_j[gj][j];
                }
            }
        }
    }
    for (g = 0, gi = 0; gi < 2; gi++) {
        for (gj = 0; gj < 2; gj++, g++) {
            for (l = 0, i = 0; i < 2; i++) {
                for (j = 0; j < 2; j++, l++) {
                    e->dphi_r[g][0][l] = dphi_i[gi][i]*phi_j[gj][j];
                    e->dphi_r[g][1][l] = phi_i[gi][i]*dphi_j[gj][j];
                }
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        tmp1 = e->phi[g][3];
        tmp2 = e->dphi_r[g][0][3];
        tmp3 = e->dphi_r[g][1][3];
        e->phi[g][3] = e->phi[g][2];
        e->dphi_r[g][0][3]   = e->dphi_r[g][0][2];
        e->dphi_r[g][1][3]   = e->dphi_r[g][1][2];
        e->phi[g][2] = tmp1;
        e->dphi_r[g][0][2] = tmp2;
        e->dphi_r[g][1][2] = tmp3;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DQuadrilateralElementFE"
extern PetscErrorCode FracD2DQuadrilateralElementFE(PetscReal **coords, FracDFEElement *e)
{
    PetscInt    i,j,k,g;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        for(j = 0; j < e->dim; j++){
            e->nodecoords[i][j] = coords[i][j];
        }
    }
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                e->J[g][i][j] = 0.;
                e->invJ[g][i][j] = 0.;
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->dim; j++){
                for(k = 0; k < e->nodes; k++){
                    e->J[g][i][j] += e->dphi_r[g][i][k]*e->nodecoords[k][j];
                }
            }
        }
    }
    for(g = 0; g < e->ng; g++){
        e->detJ[g] = e->J[g][1][1]*e->J[g][0][0]-e->J[g][0][1]*e->J[g][1][0];
        e->invJ[g][0][0] = 1./e->detJ[g]*e->J[g][1][1];
        e->invJ[g][0][1] = -1./e->detJ[g]*e->J[g][0][1];
        e->invJ[g][1][0] = -1./e->detJ[g]*e->J[g][1][0];
        e->invJ[g][1][1] = 1./e->detJ[g]*e->J[g][0][0];
        for(i = 0; i < e->dim; i++){
            for(j = 0; j < e->nodes; j++){
                e->dphi[g][i][j] = 0;
                for(k = 0; k < e->dim; k++){
                    e->dphi[g][i][j] += e->invJ[g][i][k]*e->dphi_r[g][k][j];
                }
            }
        }
        e->detJ[g] = PetscAbs(e->detJ[g]);
    }
    e->Volume = 0.;
    for(g = 0; g < e->ng; g++){
        e->Volume += PetscAbs(e->detJ[g]) * e->weight[g];
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD1DReferenceElementFEDerivative"
extern PetscErrorCode FracD1DReferenceElementFEDerivative(FracDFEElement *e)
{
    
    PetscFunctionBegin;
    e->weight[0]    = 1.;
    e->int_point[0][0]     = 1/2.;
    e->int_point[0][1]     = 1/2.;
    e->phi[0][1]     = e->int_point[0][1];
    e->phi[0][0]     = 1-e->phi[0][1];
    e->dphi_r[0][0][0] = -1;e->dphi_r[0][0][1] = 1;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD1DElementFE"
extern PetscErrorCode FracD1DElementFE(PetscReal **coords, FracDFEElement *e)
{
    PetscInt i,j,k;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        for(j = 0; j < e->dim; j++){
            e->nodecoords[i][j] = coords[i][j];
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            e->J[0][i][j] = 0.;
            e->invJ[0][i][j] = 0.;
        }
    }
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->dim; j++){
            for(k = 0; k < e->nodes; k++){
                e->J[0][i][j] += e->dphi_r[0][i][k]*e->nodecoords[k][j];
            }
        }
    }
    e->detJ[0] = 0;
    e->detJ[0] = e->J[0][0][0];
    e->invJ[0][0][0] = 1./e->detJ[0];
    for(i = 0; i < e->dim; i++){
        for(j = 0; j < e->nodes; j++){
            e->dphi[0][i][j] = 0;
            for(k = 0; k < e->dim; k++){
                e->dphi[0][i][j] += e->invJ[0][i][k]*e->dphi_r[0][k][j];
            }
        }
    }
    e->detJ[0] = PetscAbs(e->detJ[0]);
    e->Volume = e->detJ[0];
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFEElementCreate"
extern PetscErrorCode FracDFEElementCreate(PetscInt dim, FracDFEElement *e, FracDElementType elementType)
{
    PetscInt i,j;
    PetscFunctionBegin;
    switch (dim) {
        case 1:
            e->ng        = 1;
            e->dim       = dim;
            e->nodes     = 2;
            break;
        case 2:
            e->dim       = dim;
            if(elementType == TRIANGLE || elementType == TETRAHEDRAL)
            {
                e->ng        = 1;
                e->nodes     = 3;
            }
            if(elementType == QUADRILATERAL || elementType == HEXAHEDRAL)
            {
                e->ng        = 4;
                e->nodes     = 4;
            }
            break;
        case 3:
            e->dim       = dim;
            if(elementType == TETRAHEDRAL)
            {
                e->ng        = 1;
                e->nodes     = 4;
            }
            if(elementType == HEXAHEDRAL)
            {
                e->ng        = 8;
                e->nodes     = 8;
            }
            break;
        default:
            SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i %s\n",dim,__FUNCT__);
            break;
    }
    e->nodecoords = (PetscReal **)malloc(e->nodes * sizeof(PetscReal *));
    for(i = 0; i < e->nodes; i++)
    {
        e->nodecoords[i] = (PetscReal *)malloc(e->dim * sizeof(PetscReal));
    }
    e->dphi = (PetscReal ***)malloc(e->ng * sizeof(PetscReal **));
    e->dphi_r = (PetscReal ***)malloc(e->ng * sizeof(PetscReal **));
    e->J = (PetscReal ***)malloc(e->ng * sizeof(PetscReal **));
    e->invJ = (PetscReal ***)malloc(e->ng * sizeof(PetscReal **));
    for(i = 0; i < e->ng; i++)
    {
        e->dphi[i] = (PetscReal **)malloc(e->dim * sizeof(PetscReal *));
        e->dphi_r[i] = (PetscReal **)malloc(e->dim * sizeof(PetscReal *));
        e->J[i] = (PetscReal **)malloc(e->dim * sizeof(PetscReal *));
        e->invJ[i] = (PetscReal **)malloc(e->dim * sizeof(PetscReal *));
        for(j = 0; j < e->dim; j++)
        {
            e->dphi[i][j] = (PetscReal *)malloc(e->nodes * sizeof(PetscReal));
            e->dphi_r[i][j] = (PetscReal *)malloc(e->nodes * sizeof(PetscReal));
            e->J[i][j]    = (PetscReal *)malloc(e->dim * sizeof(PetscReal));
            e->invJ[i][j] = (PetscReal *)malloc(e->dim * sizeof(PetscReal));
        }
    }
    e->int_point = (PetscReal **) malloc(e->ng*sizeof(PetscReal *));
    e->phi = (PetscReal **) malloc(e->ng*sizeof(PetscReal *));
    for(i = 0; i < e->ng; i++)
    {
        e->int_point[i] = (PetscReal *) malloc(e->nodes*sizeof(PetscReal));
        e->phi[i] = (PetscReal *) malloc(e->nodes*sizeof(PetscReal));
    }
    e->weight = (PetscReal *) malloc(e->ng*sizeof(PetscReal));
    e->detJ = (PetscReal *) malloc(e->ng*sizeof(PetscReal));

    switch (dim) {
        case 1:
            FracD1DReferenceElementFEDerivative(e);
        break;
        case 2:
            if(elementType == TRIANGLE || elementType == TETRAHEDRAL)       FracD2DReferenceTriangleElementFEDerivative(e);
            if(elementType == QUADRILATERAL || elementType == HEXAHEDRAL)   FracD2DReferenceQuadrilateralElementFEDerivative(e);
        break;
        case 3:
            if(elementType == TETRAHEDRAL)  FracD3DReferenceTetrahedralElementFEDerivative(e);
            if(elementType == HEXAHEDRAL)   FracD3DReferenceHexahedralElementFEDerivative(e);
        break;
        default:
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i %s\n",dim,__FUNCT__);
        break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFEElementDestroy"
extern PetscErrorCode FracDFEElementDestroy(FracDFEElement *e)
{
    PetscInt i,j;
    
    PetscFunctionBegin;
    for(i = 0; i < e->nodes; i++){
        free(e->nodecoords[i]);
    }
    for(i = 0; i < e->ng; i++){
        for(j = 0; j < e->dim; j++){
            free(e->dphi[i][j]);
            free(e->dphi_r[i][j]);
            free(e->invJ[i][j]);
            free(e->J[i][j]);
        }
        free(e->dphi[i]);
        free(e->dphi_r[i]);
        free(e->invJ[i]);
        free(e->J[i]);
        free(e->int_point[i]);
        free(e->phi[i]);
    }
    free(e->dphi);
    free(e->dphi_r);
    free(e->invJ);
    free(e->J);
    free(e->nodecoords);
    free(e->int_point);
    free(e->phi);
    free(e->weight);
    free(e->detJ);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCVFEFaceCreate"
extern PetscErrorCode FracDCVFEFaceCreate(PetscInt dim, FracDCVFEFace *f,FracDElementType elementType)
{
    PetscInt    i;
    PetscFunctionBegin;
    
    switch (dim) {
        case 2:
            f->dim           = dim;
            f->nodes         = 2;
            if(elementType == TRIANGLE)
                f->elemnodes     = 3;
            if(elementType == QUADRILATERAL){
                f->elemnodes     = 4;
                f->FracDLocalToGlobalCoordinate = FracD2DQuadrilateralLocalToGlobalCoordinate;
            }
            break;
        case 3:
            f->dim           = dim;
            f->nodes         = 4;
            if(elementType == TETRAHEDRAL)
                f->elemnodes     = 4;
            if(elementType == HEXAHEDRAL){
                f->elemnodes     = 8;
                f->FracDLocalToGlobalCoordinate = FracD3DHexahedralLocalToGlobalCoordinate;
            }
            break;
        default:
            SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i %s\n",dim,__FUNCT__);
            break;
    }
    f->scale = 1./f->elemnodes;
    f->n = (PetscReal *) malloc(f->dim*sizeof(PetscReal));
    f->int_point = (PetscReal *) malloc(f->dim*sizeof(PetscReal));
    f->phi = (PetscReal *) malloc(f->elemnodes*sizeof(PetscReal));
    f->facecoords = (PetscReal **)malloc(f->nodes * sizeof(PetscReal *));
    for(i = 0; i < f->nodes; i++)
    {
        f->facecoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    f->elemcoords = (PetscReal **)malloc(f->elemnodes * sizeof(PetscReal *));
    for(i = 0; i < f->elemnodes; i++)
    {
        f->elemcoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    
    f->dphi = (PetscReal **)malloc(f->dim * sizeof(PetscReal *));
    for(i = 0; i < f->dim; i++)
    {
        f->dphi[i] = (PetscReal *)malloc(f->elemnodes * sizeof(PetscReal));
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DTriangleCVFEFace"
extern PetscErrorCode FracD2DTriangleCVFEFace(PetscReal **elemcoords, PetscReal **coords, FracDCVFEFace *f)
{
    PetscInt            i,j,k,l;
    PetscReal           dphi[2][3],J[2][2],invJ[2][2],detJ;
    PetscReal           **localcoords;
    
    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(f->elemnodes * sizeof(PetscReal *));
    for(i = 0; i < f->elemnodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    for(i = 0; i < f->elemnodes; i++){
        for(j = 0; j < f->dim; j++){
            f->elemcoords[i][j] = elemcoords[i][j];
        }
    }
    for(i = 0; i < f->nodes; i++){
        for(j = 0; j < f->dim; j++){
            f->facecoords[i][j] = coords[i][j];
        }
    }
    /* Centroid of CVFE face*/
    for(j = 0; j < f->dim; j++){
        f->int_point[j] = 0;
        for(i = 0; i < f->nodes; i++){
            f->int_point[j] += f->facecoords[i][j];
        }
        f->int_point[j] = f->int_point[j]/f->nodes;
    }
    /* face size and normal computation*/
    f->n[0] = (f->facecoords[1][1]-f->facecoords[0][1]);
    f->n[1] = -1*(f->facecoords[1][0]-f->facecoords[0][0]);
    f->faceArea = sqrt(pow(f->n[0],2)+pow(f->n[1],2));
    for(i = 0; i < f->dim; i++) f->n[i] = f->n[i]/f->faceArea;
    /* Derivative of shape function of element nodes */
    dphi[0][0] = -1;dphi[0][1] = 1;dphi[0][2] = 0;
    dphi[1][0] = -1;dphi[1][1] = 0;dphi[1][2] = 1;
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = 0;
    detJ = J[1][1]*J[0][0]-J[0][1]*J[1][0];
    f->elemVolume = 1/2.*PetscAbs(detJ);
    invJ[0][0] = 1./detJ*J[1][1];
    invJ[0][1] = -1./detJ*J[0][1];
    invJ[1][0] = -1./detJ*J[1][0];
    invJ[1][1] = 1./detJ*J[0][0];
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->elemnodes; j++){
            f->dphi[i][j] = 0;
            for(k = 0; k < f->dim; k++){
                f->dphi[i][j] += invJ[i][k]*dphi[k][j];
            }
        }
    }
    /* Element volume and Shape function of element nodes */
    for(l = 0; l < f->elemnodes; l++){
        for(i = 0; i < f->elemnodes; i++){
            for(j = 0; j < f->dim; j++){
                localcoords[i][j] = f->elemcoords[i][j];
            }
        }
        for(i = 0; i < f->dim; i++) localcoords[l][i] = f->int_point[i];
        for(i = 0; i < f->dim; i++){
            for(j = 0; j < f->dim; j++){
                J[i][j] = 0.;
                for(k = 0; k < f->elemnodes; k++){
                    J[i][j] += dphi[i][k]*localcoords[k][j];
                }
            }
        }
        f->phi[l] = 1/2.*(J[1][1]*J[0][0]-J[0][1]*J[1][0])/f->elemVolume;
        f->phi[l] = PetscAbs(f->phi[l]);
    }
    /* Derivative of shape function of element nodes */
    for(i = 0; i < f->elemnodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DQuadrilateralCVFEFace"
extern PetscErrorCode FracD2DQuadrilateralCVFEFace(PetscReal **elemcoords, PetscReal **coords, FracDCVFEFace *f)
{
    PetscInt            i,j,k;
    PetscReal           dphi[2][4],J[2][2],invJ[2][2],detJ;
    PetscReal           eta_array[2],L[2],centroid[f->dim];
    PetscReal           **localcoords;
    PetscErrorCode      ierr;

    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(f->elemnodes * sizeof(PetscReal *));
    for(i = 0; i < f->elemnodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    for(j = 0; j < f->dim; j++){
        centroid[j] = 0;
        eta_array[j] = 0.;
        for(i = 0; i < f->elemnodes; i++){
            f->elemcoords[i][j] = elemcoords[i][j];
            centroid[j] += elemcoords[i][j];
        }
        centroid[j] = centroid[j]/f->elemnodes;
    }
    for(i = 0; i < f->nodes; i++){
        for(j = 0; j < f->dim; j++){
            f->facecoords[i][j] = coords[i][j];
        }
    }
    for(j = 0; j < f->dim; j++){
        L[j] = 0.;
        f->int_point[j] = 0;
        for(i = 0; i < f->nodes; i++){
            f->int_point[j] += f->facecoords[i][j];
        }
        f->int_point[j] = f->int_point[j]/f->nodes;
    }
    /* face size and normal computation*/
    f->n[0] = (f->facecoords[1][1]-f->facecoords[0][1]);
    f->n[1] = -1*(f->facecoords[1][0]-f->facecoords[0][0]);
    f->faceArea = sqrt(pow(f->n[0],2)+pow(f->n[1],2));
    for(i = 0; i < f->dim; i++) f->n[i] = f->n[i]/f->faceArea;
    
    ierr = FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(f->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = J[1][1]*J[0][0]-J[0][1]*J[1][0];
    f->elemVolume = 4.*PetscAbs(detJ);
    //inputs: L,,Xp
    //    initial_value will be computed inside function
    for(i = 0; i < f->dim; i++) {
        L[i] = J[i][i];
        eta_array[i] = (f->int_point[i]-centroid[i])/L[i];
    }
    ierr = FracDComputeLocalFECoordinates(eta_array,f->int_point,L,f->elemcoords, f->dim, f->FracDLocalToGlobalCoordinate);CHKERRQ(ierr);
    ierr = FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(f->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = J[1][1]*J[0][0]-J[0][1]*J[1][0];
    invJ[0][0] = 1./detJ*J[1][1];
    invJ[0][1] = -1./detJ*J[0][1];
    invJ[1][0] = -1./detJ*J[1][0];
    invJ[1][1] = 1./detJ*J[0][0];
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->elemnodes; j++){
            f->dphi[i][j] = 0;
            for(k = 0; k < f->dim; k++){
                f->dphi[i][j] += invJ[i][k]*dphi[k][j];
            }
        }
    }
    for(i = 0; i < f->elemnodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DTetrahedralCVFEFace"
extern PetscErrorCode FracD3DTetrahedralCVFEFace(PetscReal **elemcoords, PetscReal **coords, FracDCVFEFace *f)
{
    PetscInt            i,j,k,l;
    PetscReal           dphi[f->dim][f->elemnodes],J[f->dim][f->dim],invJ[f->dim][f->dim],detJ;
    PetscReal           **localcoords=NULL;
    
    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(f->elemnodes * sizeof(PetscReal *));
    for(i = 0; i < f->elemnodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    for(i = 0; i < f->elemnodes; i++){
        for(j = 0; j < f->dim; j++){
            f->elemcoords[i][j] = elemcoords[i][j];
            f->facecoords[i][j] = coords[i][j];
        }
    }
    /* Centroid of CVFE face*/
    for(j = 0; j < f->dim; j++){
        f->int_point[j] = 0;
        for(i = 0; i < f->nodes; i++){
            f->int_point[j] += f->facecoords[i][j];
        }
        f->int_point[j] = f->int_point[j]/f->nodes;
    }
    /* Face size and normal computation*/
    f->n[0] = (f->facecoords[1][1]-f->facecoords[0][1])*(f->facecoords[2][2]-f->facecoords[0][2])-(f->facecoords[2][1]-f->facecoords[0][1])*(f->facecoords[1][2]-f->facecoords[0][2]);
    f->n[1] = (f->facecoords[1][2]-f->facecoords[0][2])*(f->facecoords[2][0]-f->facecoords[0][0])-(f->facecoords[1][0]-f->facecoords[0][0])*(f->facecoords[2][2]-f->facecoords[0][2]);
    f->n[2] = (f->facecoords[1][0]-f->facecoords[0][0])*(f->facecoords[2][1]-f->facecoords[0][1])-(f->facecoords[2][0]-f->facecoords[0][0])*(f->facecoords[1][1]-f->facecoords[0][1]);
    f->faceArea = sqrt(pow(f->n[0],2)+pow(f->n[1],2)+pow(f->n[2],2));
    for(i = 0; i < f->dim; i++) f->n[i] = f->n[i]/f->faceArea;
    /* Derivative of shape function of element nodes */
    dphi[0][0] = -1;dphi[0][1] = 1;dphi[0][2] = 0;dphi[0][3] = 0;
    dphi[1][0] = -1;dphi[1][1] = 0;dphi[1][2] = 1;dphi[1][3] = 0;
    dphi[2][0] = -1;dphi[2][1] = 0;dphi[2][2] = 0;dphi[2][3] = 1;
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = 0;
    for(i = 0; i < f->dim; i++){
        detJ += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
    }
    f->elemVolume = 1/6.*PetscAbs(detJ);
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            invJ[j][i] = 1./detJ*((J[(i+1)%3][(j+1)%3]*J[(i+2)%3][(j+2)%3])-(J[(i+1)%3][(j+2)%3]*J[(i+2)%3][(j+1)%3]));
        }
    }
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->nodes; j++){
            f->dphi[i][j] = 0;
            for(k = 0; k < f->dim; k++){
                f->dphi[i][j] += invJ[i][k]*dphi[k][j];
            }
        }
    }
    /* Element volume and shape function of element nodes */
    for(l = 0; l < f->elemnodes; l++){
        for(i = 0; i < f->elemnodes; i++){
            for(j = 0; j < f->dim; j++){
                localcoords[i][j] = f->elemcoords[i][j];
            }
        }
        for(i = 0; i < f->dim; i++) localcoords[l][i] = f->int_point[i];
        for(i = 0; i < f->dim; i++){
            for(j = 0; j < f->dim; j++){
                J[i][j] = 0.;
                for(k = 0; k < f->elemnodes; k++){
                    J[i][j] += dphi[i][k]*localcoords[k][j];
                }
            }
        }
        f->phi[l] = 0;
        for(i = 0; i < f->dim; i++){
            f->phi[l] += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
        }
        f->phi[l] = 1/6.*PetscAbs(f->phi[l])/f->elemVolume;
    }
    for(i = 0; i < f->elemnodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD3DHexahedralCVFEFace"
extern PetscErrorCode FracD3DHexahedralCVFEFace(PetscReal **elemcoords, PetscReal **coords, FracDCVFEFace *f)
{
    PetscInt            i,j,k;
    PetscReal           dphi[3][8],J[3][3],invJ[3][3],detJ;
    PetscReal           eta_array[3],L[3],centroid[f->dim];
    PetscReal           **localcoords;
    PetscErrorCode      ierr;
    
    PetscFunctionBegin;
    localcoords = (PetscReal **)malloc(f->elemnodes * sizeof(PetscReal *));
    for(i = 0; i < f->elemnodes; i++)
    {
        localcoords[i] = (PetscReal *)malloc(f->dim * sizeof(PetscReal));
    }
    for(j = 0; j < f->dim; j++){
        centroid[j] = 0;
        eta_array[j] = 0.;
        for(i = 0; i < f->elemnodes; i++){
            f->elemcoords[i][j] = elemcoords[i][j];
            centroid[j] += elemcoords[i][j];
        }
        centroid[j] = centroid[j]/f->elemnodes;
    }
//    printf(" \n Face Coord\n ");
    for(i = 0; i < f->nodes; i++){
        for(j = 0; j < f->dim; j++){
            f->facecoords[i][j] = coords[i][j];
//            printf(" %g ",f->facecoords[i][j]);
        }
//        printf(" \n ");
    }
    
//    printf(" \n Elem Coord\n ");
//    for(i = 0; i < f->elemnodes; i++){
//        for(j = 0; j < f->dim; j++){
//            printf(" %g ",f->elemcoords[i][j]);
//        }
//        printf(" \n ");
//    }
    
    for(j = 0; j < f->dim; j++){
        L[j] = 0.;
        f->int_point[j] = 0;
        for(i = 0; i < f->nodes; i++){
            f->int_point[j] += f->facecoords[i][j];
        }
        f->int_point[j] = f->int_point[j]/f->nodes;
    }
    /* Face size and normal computation*/
    f->n[0] = (f->facecoords[1][1]-f->facecoords[0][1])*(f->facecoords[2][2]-f->facecoords[0][2])-(f->facecoords[2][1]-f->facecoords[0][1])*(f->facecoords[1][2]-f->facecoords[0][2]);
    f->n[1] = (f->facecoords[1][2]-f->facecoords[0][2])*(f->facecoords[2][0]-f->facecoords[0][0])-(f->facecoords[1][0]-f->facecoords[0][0])*(f->facecoords[2][2]-f->facecoords[0][2]);
    f->n[2] = (f->facecoords[1][0]-f->facecoords[0][0])*(f->facecoords[2][1]-f->facecoords[0][1])-(f->facecoords[2][0]-f->facecoords[0][0])*(f->facecoords[1][1]-f->facecoords[0][1]);
    f->faceArea = sqrt(pow(f->n[0],2)+pow(f->n[1],2)+pow(f->n[2],2));
    for(i = 0; i < f->dim; i++) f->n[i] = f->n[i]/f->faceArea;
    
    
    
//    printf("\n Face = %g\n",f->faceArea);
    ierr = FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(f->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = 0.;
    for(i = 0; i < f->dim; i++){
        detJ += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
    }
    f->elemVolume = 8.*PetscAbs(detJ);
//    printf("\n elemVolume = %g\n",f->elemVolume);

    
//    printf(" 1st Phi \n ");
//    for(i = 0; i < f->elemnodes; i++){
//        printf(" %g %g %g %g \n",f->phi[i],dphi[0][i],dphi[1][i],dphi[2][i]);
//    }
//    printf(" \n %g %g %g",eta_array[0],eta_array[1],eta_array[2]);
    
    
    
    //inputs: L,,Xp
    for(i = 0; i < f->dim; i++) {
        L[i] = J[i][i];
        eta_array[i] = (f->int_point[i]-centroid[i])/L[i];
//        printf("\nINt_point:  %g %g %g %g %g",L[i],f->int_point[i],centroid[i],1./L[i],eta_array[i]);
    }
    
//    printf(" \n Old eta_array: %g %g %g",eta_array[0],eta_array[1],eta_array[2]);

    ierr = FracDComputeLocalFECoordinates(eta_array,f->int_point,L,f->elemcoords, f->dim, f->FracDLocalToGlobalCoordinate);CHKERRQ(ierr);
//    printf(" \n New eta_array: %g %g %g",eta_array[0],eta_array[1],eta_array[2]);

    ierr = FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(f->phi,dphi,eta_array);CHKERRQ(ierr);
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            J[i][j] = 0.;
            for(k = 0; k < f->elemnodes; k++){
                J[i][j] += dphi[i][k]*f->elemcoords[k][j];
            }
        }
    }
    detJ = 0.;
    for(i = 0; i < f->dim; i++){
        detJ += (J[0][i]*(J[1][(i+1)%3]*J[2][(i+2)%3]-J[1][(i+2)%3]*J[2][(i+1)%3]));
    }
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->dim; j++){
            invJ[j][i] = 1./detJ*((J[(i+1)%3][(j+1)%3]*J[(i+2)%3][(j+2)%3])-(J[(i+1)%3][(j+2)%3]*J[(i+2)%3][(j+1)%3]));
        }
    }
    for(i = 0; i < f->dim; i++){
        for(j = 0; j < f->elemnodes; j++){
            f->dphi[i][j] = 0;
            for(k = 0; k < f->dim; k++){
                f->dphi[i][j] += invJ[i][k]*dphi[k][j];
            }
        }
    }
    
//    printf(" 2nd Phi \n ");
//    for(i = 0; i < f->elemnodes; i++){
//        printf(" %g %g %g %g \n",f->phi[i],f->dphi[0][i],f->dphi[1][i],f->dphi[2][i]);
//    }
    
    for(i = 0; i < f->elemnodes; i++) {
        free(localcoords[i]);
    }
    free(localcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCVFEFaceDestroy"
extern PetscErrorCode FracDCVFEFaceDestroy(FracDCVFEFace *f)
{
    PetscInt i;
    
    PetscFunctionBegin;
    
    free(f->n);
    free(f->int_point);
    free(f->phi);
    for(i = 0; i < f->dim; i++){
        free(f->dphi[i]);
    }
    free(f->dphi);
    for(i = 0; i < f->nodes; i++){
        free(f->facecoords[i]);
    }
    free(f->facecoords);
    for(i = 0; i < f->elemnodes; i++){
        free(f->elemcoords[i]);
    }
    free(f->elemcoords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDMatrixApplyDirichletBC"
extern PetscErrorCode FracDMatrixApplyDirichletBC(DM dm, Mat K, FracDBC *BC, PetscScalar diagonalvalue)
{
    PetscErrorCode      ierr;
    PetscInt            i,j,k,l,ll,goffset,dof;
    PetscInt            vStart, vEnd, fStart, fEnd;
    IS                  pointIS;
    PetscInt            numpoints;
    const PetscInt      *pointID;
    PetscSection        localSection;
    PetscInt            *components, *row, numrows = 0, count = 0, loc = 0;
    PetscInt            pt, numclpts, *closurept = NULL;
    
    PetscFunctionBegin;
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (i = 0; i < BC->numRegions; i++){
        ierr = DMGetStratumIS(dm, BC->labelName, BC->regions[i], &pointIS);
        if(pointIS){
            ierr = ISGetLocalSize(pointIS, &numpoints);
            ierr = ISGetIndices(pointIS, &pointID);
            count = 0;
            for(j = 0; j < numpoints; j++){
                if( pointID[j] >= fStart && pointID[j] < fEnd){
                    ierr = DMPlexGetTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    for(l = 0; l < numclpts; l++){
                        pt = closurept[2*l];
                        if(pt >= vStart && pt < vEnd){
                            count += 1;
                        }
                    }
                    ierr = DMPlexRestoreTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                }
            }
            numrows += count * BC->numcompsperlabel[i];
            ierr = ISRestoreIndices(pointIS, &pointID);CHKERRQ(ierr);
        }
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    //    compute indices for BC's
    ierr = PetscMalloc1(numrows,&row);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(dm,&localSection);CHKERRQ(ierr);
    for (l = 0, i = 0; i < BC->numRegions; i++){
        ierr = DMGetStratumIS(dm, BC->labelName, BC->regions[i], &pointIS);
        if(pointIS){
            ierr = ISGetLocalSize(pointIS, &numpoints);
            ierr = ISGetIndices(pointIS, &pointID);
            components = &BC->components[loc];
            for(j = 0; j < numpoints; j++){
                if( pointID[j] >= fStart && pointID[j] < fEnd){
                    ierr = DMPlexGetTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    for(ll = 0; ll < numclpts; ll++){
                        pt = closurept[2*ll];
                        if(pt >= vStart && pt < vEnd){
                            ierr = PetscSectionGetOffset(localSection, pt, &goffset);CHKERRQ(ierr);
                            goffset = goffset < 0 ? -(goffset+1):goffset;
                            for(k = 0; k < BC->numcompsperlabel[i]; k++, l++){
                                dof = components[k];
                                row[l] = goffset + dof;
                            }
                        }
                    }
                    ierr = DMPlexRestoreTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                }
            }
            ierr = ISRestoreIndices(pointIS, &pointID);CHKERRQ(ierr);
        }
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
        loc += BC->numcompsperlabel[i];
    }
    ierr = MatZeroRows(K,numrows,row,diagonalvalue,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscFree(row);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDResidualApplyDirichletBC"
extern PetscErrorCode FracDResidualApplyDirichletBC(Vec residual, Vec V, FracDBC *BC)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            i,j,k,l,offset,dof;
    PetscInt            vStart, vEnd, fStart, fEnd;
    IS                  pointIS;
    PetscInt            numpoints;
    const PetscInt      *pointID;
    PetscSection        localSection;
    PetscScalar         *Residual_array;
    const PetscScalar   *V_array;
    PetscInt            *components, loc = 0;
    PetscReal           *bcvalue;
    PetscInt            numclpts, pt, *closurept = NULL;
    Vec                 localResidual, localV;
    
    PetscFunctionBegin;
    ierr = VecGetDM(residual,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,residual,INSERT_VALUES,localResidual);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,residual,INSERT_VALUES,localResidual);CHKERRQ(ierr);
    ierr = VecSet(residual,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localResidual,&Residual_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = VecGetArrayRead(localV,&V_array);CHKERRQ(ierr);
    
    for (i = 0; i < BC->numRegions; i++){
        ierr = DMGetStratumIS(dm, BC->labelName, BC->regions[i], &pointIS);
        if(pointIS){
            ierr = ISGetLocalSize(pointIS, &numpoints);
            ierr = ISGetIndices(pointIS, &pointID);
            components = &BC->components[loc];
            bcvalue = &BC->values[loc];
            for(j = 0; j < numpoints; j++){
                if( pointID[j] >= fStart && pointID[j] < fEnd){
                    ierr = DMPlexGetTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    for(l = 0; l < numclpts; l++){
                        pt = closurept[2*l];
                        if(pt >= vStart && pt < vEnd){
                            ierr = PetscSectionGetOffset(localSection, pt, &offset);CHKERRQ(ierr);
                            for(k = 0; k < BC->numcompsperlabel[i]; k++){
                                dof = components[k];
                                Residual_array[offset+dof] = V_array[offset+dof]-bcvalue[k];
                            }
                        }
                    }
                    ierr = DMPlexRestoreTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                }
            }
            ierr = ISRestoreIndices(pointIS, &pointID);
        }
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
        loc += BC->numcompsperlabel[i];
    }
    ierr = VecRestoreArray(localResidual,&Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,localResidual,INSERT_VALUES,residual);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,localResidual,INSERT_VALUES,residual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localResidual);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(localV,&V_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localV);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DProjectFaceCoordinateDimensions"
extern PetscErrorCode FracD2DProjectFaceCoordinateDimensions(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim)
{
    PetscInt            i;
    PetscReal           ab,ac,abdotac = 0, theta;
    PetscReal           abvec[3];
    PetscReal           acvec[3];
    
    PetscFunctionBegin;
    for(i = 0; i < 3; i++)
    {
        abvec[i] = coords1[1][i]-coords1[0][i];
        acvec[i] = coords1[2][i]-coords1[0][i];
    }
    ab = sqrt(pow(abvec[0],2)+pow(abvec[1],2)+pow(abvec[2],2));
    ac = sqrt(pow(acvec[0],2)+pow(acvec[1],2)+pow(acvec[2],2));
    for(i = 0; i < 3; i++)
        abdotac += abvec[i]*acvec[i];
    theta = acos (abdotac/(ab*ac));
    coords[0][0] = 0;               coords[0][1] = 0;
    coords[1][0] = ab;              coords[1][1] = 0;
    coords[2][0] = ac*cos(theta);   coords[2][1] = ac*sin(theta);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD2DProjectFaceCoordinateDimensionsHexahedral"
extern PetscErrorCode FracD2DProjectFaceCoordinateDimensionsHexahedral(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim)
{
    PetscInt            i;
    PetscReal           ab,ac,ad,abdotac=0,abdotad=0,theta_bc,theta_bd;
    PetscReal           abvec[3];
    PetscReal           acvec[3];
    PetscReal           advec[3];
    
    PetscFunctionBegin;
    for(i = 0; i < 3; i++)
    {
        abvec[i] = coords1[1][i]-coords1[0][i];
        acvec[i] = coords1[2][i]-coords1[0][i];
        advec[i] = coords1[3][i]-coords1[0][i];
    }
    ab = sqrt(pow(abvec[0],2)+pow(abvec[1],2)+pow(abvec[2],2));
    ac = sqrt(pow(acvec[0],2)+pow(acvec[1],2)+pow(acvec[2],2));
    ad = sqrt(pow(advec[0],2)+pow(advec[1],2)+pow(advec[2],2));
    for(i = 0; i < 3; i++){
        abdotac += abvec[i]*acvec[i];
        abdotad += abvec[i]*advec[i];
    }
    theta_bc = acos (abdotac/(ab*ac));
    theta_bd = acos (abdotad/(ab*ad));
    coords[0][0] = 0;                   coords[0][1] = 0;
    coords[1][0] = ab;                  coords[1][1] = 0;
    coords[2][0] = ac*cos(theta_bc);    coords[2][1] = ac*sin(theta_bc);
    coords[3][0] = ad*cos(theta_bd);    coords[3][1] = ad*sin(theta_bd);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracD1DProjectFaceCoordinateDimensions"
extern PetscErrorCode FracD1DProjectFaceCoordinateDimensions(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim)
{
    PetscFunctionBegin;
    coords[0][0] = 0;
    coords[1][0] = sqrt(pow(coords1[0][0]-coords1[1][0],2)+pow(coords1[0][1]-coords1[1][1],2));
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDResidualApplyNeumannBC_local"
extern PetscErrorCode FracDResidualApplyNeumannBC_local(PetscReal *residual_local,PetscReal *bcvalues, FracDFEElement *e, PetscInt n)
{
    PetscInt       i,c,l,g;
    
    PetscFunctionBegin;
    for (l = 0; l < n * e->nodes; l++)
        residual_local[l] = 0;
    
    for (i = 0, l = 0; i < e->nodes; i++) {
        for (c = 0; c < n; c++, l++) {
            for (g = 0; g < e->ng; g++){
                residual_local[l] += e->detJ[g] * bcvalues[c] * e->phi[g][i] * e->weight[g];
            }
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDResidualApplyFENeumannBC"
extern PetscErrorCode FracDResidualApplyFENeumannBC(Vec residual, FracDBC *BC, FracDFEElement *elD, PetscErrorCode (*FracDCreateDMinusOneFEElement)(PetscReal**, FracDFEElement*), PetscErrorCode (*ProjectFaceCoordinates)(PetscReal**,PetscReal**,PetscInt,PetscInt))
{
    PetscErrorCode      ierr;
    DM                  dm, cdm;
    PetscInt            i,j,k,ii,jj,kk,l,dof,pt,offset;
    PetscInt            vStart, vEnd, fStart, fEnd;
    IS                  pointIS;
    PetscInt            numpoints;
    const PetscInt      *pointID;
    PetscInt            loc = 0;
    PetscReal           *bcvalue;
    PetscInt            numclpts, *closurept = NULL;
    Vec                 coordinates;
    PetscScalar         *coord_array = NULL, *Residual_array = NULL, *residual_local;
    PetscReal           **coords, **coords1;
    PetscSection        cordSection, section;
    PetscInt            cordsize;
    PetscReal           *neumvalues;
    Vec                 res,localResidual;
    
    PetscFunctionBegin;
    coords = (PetscReal **)malloc(elD->nodes * sizeof(PetscReal *));
    coords1 = (PetscReal **)malloc(elD->nodes * sizeof(PetscReal *));
    for(i = 0; i < elD->nodes; i++) coords[i] = (PetscReal *)malloc(elD->dim * sizeof(PetscReal));
    for(i = 0; i < elD->nodes; i++) coords1[i] = (PetscReal *)malloc((elD->dim+1) * sizeof(PetscReal));
    ierr = VecGetDM(residual,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section,vStart,&dof);CHKERRQ(ierr);
    neumvalues = (PetscReal *)malloc(dof * sizeof(PetscReal));
    residual_local = (PetscReal *)malloc(dof*elD->nodes * sizeof(PetscReal));
    
    ierr = VecDuplicate(residual,&res);CHKERRQ(ierr);
    ierr = VecSet(res,0.);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localResidual,&Residual_array);CHKERRQ(ierr);
    
    for (i = 0; i < BC->numRegions; i++){
        ierr = DMGetStratumIS(dm, BC->labelName, BC->regions[i], &pointIS);
        if(pointIS){
            ierr = ISGetLocalSize(pointIS, &numpoints);
            ierr = ISGetIndices(pointIS, &pointID);
            bcvalue = &BC->values[loc];
            for(l = 0; l < dof; l++)    neumvalues[l] = bcvalue[l];
            for(j = 0; j < numpoints; j++){
                if( pointID[j] >= fStart && pointID[j] < fEnd){
                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pointID[j], &cordsize, &coord_array);CHKERRQ(ierr);
                    for(l = 0, ii = 0; ii < elD->nodes; ii++){
                        for(jj = 0; jj < elD->dim+1; jj++, l++){
                            coords1[ii][jj] = coord_array[l];
                        }
                    }
                    ierr = ProjectFaceCoordinates(coords,coords1,elD->nodes,elD->dim);CHKERRQ(ierr);
                    ierr = FracDCreateDMinusOneFEElement(coords,elD);CHKERRQ(ierr);
                    ierr = FracDResidualApplyNeumannBC_local(residual_local, neumvalues, elD, dof);CHKERRQ(ierr);
                    ierr = DMPlexGetTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    for(kk = 0; kk < numclpts; kk++){
                        pt = closurept[2*kk];
                        if(pt >= vStart && pt < vEnd){
                            ierr = PetscSectionGetOffset(section, pt, &offset);CHKERRQ(ierr);
                            for(k = 0; k < dof; k++){
                                Residual_array[offset+k] -= residual_local[k];
                            }
                        }
                    }
                    ierr = DMPlexRestoreTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pointID[j], &cordsize, &coord_array);CHKERRQ(ierr);
                }
            }
            ierr = ISRestoreIndices(pointIS, &pointID);
        }
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
        loc += dof;
    }
    
    ierr = VecRestoreArray(localResidual,&Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,localResidual,ADD_VALUES,res);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,localResidual,ADD_VALUES,res);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localResidual);CHKERRQ(ierr);
    
    ierr = VecAXPY(residual,1.,res);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    
    free(neumvalues);
    free(residual_local);
    for(i = 0; i < elD->nodes; i++){
        free(coords[i]);
        free(coords1[i]);
    }
    free(coords);
    free(coords1);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDResidualApplyCVFENeumannBC"
extern PetscErrorCode FracDResidualApplyCVFENeumannBC(Vec residual, PetscReal dt, FracDBC *BC, FracDFEElement *elD,PetscErrorCode (*FracDCreateDMinusOneFEElement)(PetscReal**, FracDFEElement*), PetscErrorCode (*ProjectFaceCoordinates)(PetscReal**,PetscReal**,PetscInt,PetscInt))
{
    PetscErrorCode      ierr;
    DM                  dm, cdm;
    PetscInt            i,j,k,ii,jj,kk,l,dof,pt,offset;
    PetscInt            vStart, vEnd, fStart, fEnd;
    IS                  pointIS;
    PetscInt            numpoints;
    const PetscInt      *pointID;
    PetscInt            loc = 0;
    PetscReal           *bcvalue;
    PetscInt            numclpts, *closurept = NULL;
    Vec                 coordinates;
    PetscScalar         *coord_array = NULL, *Residual_array = NULL;
    PetscReal           **coords, **coords1;
    PetscSection        cordSection, section;
    PetscInt            cordsize;
    PetscReal           *neumvalues,scale=1.;
    Vec                 res,localResidual;
    
    PetscFunctionBegin;
    scale = 1./elD->nodes;
    coords = (PetscReal **)malloc(elD->nodes * sizeof(PetscReal *));
    coords1 = (PetscReal **)malloc(elD->nodes * sizeof(PetscReal *));
    for(i = 0; i < elD->nodes; i++) coords[i] = (PetscReal *)malloc(elD->dim * sizeof(PetscReal));
    for(i = 0; i < elD->nodes; i++) coords1[i] = (PetscReal *)malloc((elD->dim+1) * sizeof(PetscReal));
    ierr = VecGetDM(residual,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&section);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetDof(section,vStart,&dof);CHKERRQ(ierr);
    neumvalues = (PetscReal *)malloc(dof * sizeof(PetscReal));
    
    ierr = VecDuplicate(residual,&res);CHKERRQ(ierr);
    ierr = VecSet(res,0.);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localResidual,&Residual_array);CHKERRQ(ierr);
    
    for (i = 0; i < BC->numRegions; i++){
        ierr = DMGetStratumIS(dm, BC->labelName, BC->regions[i], &pointIS);
        if(pointIS){
            ierr = ISGetLocalSize(pointIS, &numpoints);
            ierr = ISGetIndices(pointIS, &pointID);
            bcvalue = &BC->values[loc];
            for(l = 0; l < dof; l++)    neumvalues[l] = bcvalue[l];
            for(j = 0; j < numpoints; j++){
                if( pointID[j] >= fStart && pointID[j] < fEnd){
                    ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, pointID[j], &cordsize, &coord_array);CHKERRQ(ierr);
                    for(l = 0, ii = 0; ii < elD->nodes; ii++){
                        for(jj = 0; jj < elD->dim+1; jj++, l++){
                            coords1[ii][jj] = coord_array[l];
                        }
                    }
                    ierr = ProjectFaceCoordinates(coords,coords1,elD->nodes,elD->dim);CHKERRQ(ierr);
                    ierr = FracDCreateDMinusOneFEElement(coords,elD);CHKERRQ(ierr);
                    ierr = DMPlexGetTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    for(kk = 0; kk < numclpts; kk++){
                        pt = closurept[2*kk];
                        if(pt >= vStart && pt < vEnd){
                            ierr = PetscSectionGetOffset(section, pt, &offset);CHKERRQ(ierr);
                            for(k = 0; k < dof; k++){
                                Residual_array[offset+k] += scale * elD->Volume * neumvalues[k];
                            }
                        }
                    }
                    ierr = DMPlexRestoreTransitiveClosure(dm, pointID[j], PETSC_TRUE, &numclpts, &closurept);CHKERRQ(ierr);
                    ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, pointID[j], &cordsize, &coord_array);CHKERRQ(ierr);
                }
            }
            ierr = ISRestoreIndices(pointIS, &pointID);
        }
        ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
        loc += dof;
    }
    ierr = VecRestoreArray(localResidual,&Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,localResidual,ADD_VALUES,res);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,localResidual,ADD_VALUES,res);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localResidual);CHKERRQ(ierr);
    
    ierr = VecAXPY(residual,1.,res);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    
    free(neumvalues);
    for(i = 0; i < elD->nodes; i++){
        free(coords[i]);
        free(coords1[i]);
    }
    free(coords);
    free(coords1);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDResidualApplyConstraintOnGasSaturation"
extern PetscErrorCode FracDResidualApplyConstraintOnGasSaturation(Vec residual, Vec Sg, Vec V)
{
    PetscErrorCode      ierr;
    DM                  dm;
    PetscInt            v,offset, vStart, vEnd;
    PetscSection        localSection;
    PetscScalar         *Residual_array;
    const PetscScalar   *V_array, *Sg_array;
    Vec                 localResidual, localSg, localV;
    
    PetscFunctionBegin;
    ierr = VecGetDM(residual,&dm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localResidual);CHKERRQ(ierr);
    ierr = VecSet(localResidual,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,residual,INSERT_VALUES,localResidual);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,residual,INSERT_VALUES,localResidual);CHKERRQ(ierr);
    ierr = VecSet(residual,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localResidual,&Residual_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localV);CHKERRQ(ierr);
    ierr = VecSet(localV,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = VecGetArrayRead(localV,&V_array);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localSg);CHKERRQ(ierr);
    ierr = VecSet(localSg,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,Sg,INSERT_VALUES,localSg);CHKERRQ(ierr);
    ierr = VecGetArrayRead(localSg,&Sg_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        if(PetscAbs(1.0-V_array[offset]) > PETSC_SMALL)
        Residual_array[offset] = Sg_array[offset]-0.;
    }
    ierr = VecRestoreArray(localResidual,&Residual_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dm,localResidual,INSERT_VALUES,residual);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dm,localResidual,INSERT_VALUES,residual);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localResidual);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(localSg,&Sg_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localSg);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(localV,&V_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localV);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDMatrixApplyConstraintOnGasSaturation"
extern PetscErrorCode FracDMatrixApplyConstraintOnGasSaturation(DM dm, Mat K, Vec V, PetscScalar diagonalvalue)
{
    PetscErrorCode      ierr;
    PetscInt            l = 0,v,vStart, vEnd;
    PetscSection        glocalSection,localSection;
    PetscInt            *row, numrows = 0, goffset,offset;
    Vec                 localV;
    const PetscScalar   *V_array;
    
    PetscFunctionBegin;
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultGlobalSection(dm,&glocalSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dm,&localSection);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(dm,&localV);CHKERRQ(ierr);
    ierr = VecSet(localV,0.);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm,V,INSERT_VALUES,localV);CHKERRQ(ierr);
    ierr = VecGetArrayRead(localV,&V_array);CHKERRQ(ierr);
    
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        if(PetscAbs(1.0-V_array[offset]) > PETSC_SMALL){
            numrows += 1;
        }
    }
    ierr = PetscMalloc1(numrows,&row);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionGetOffset(glocalSection, v, &goffset);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(localSection, v, &offset);CHKERRQ(ierr);
        goffset = goffset < 0 ? -(goffset+1):goffset;
        if(PetscAbs(1.0-V_array[offset]) > PETSC_SMALL){
            row[l] = goffset;
            l++;
        }
    }
    ierr = VecRestoreArrayRead(localV,&V_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dm,&localV);CHKERRQ(ierr);
    ierr = MatZeroRows(K,numrows,row,diagonalvalue,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscFree(row);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
