#ifndef FRACDFINITEELEMENT_H
#define FRACDFINITEELEMENT_H
/*
 FracDFiniteEelement.h
 (c) 2016-2018 Chukwudi CHukwudozie chdozie@gmail.com
 */

static const char *FracDElementType_name[] = {
    "TRIANGLE",
    "QUADRILATERAL",
    "TETRAHEDRAL",
    "HEXAHEDRAL",
    "FracDElementType_name",
    "",
    0
};

typedef enum {
    TRIANGLE,
    QUADRILATERAL,
    TETRAHEDRAL,
    HEXAHEDRAL
} FracDElementType;

typedef struct {
    char                labelName[PETSC_MAX_PATH_LEN];
    PetscBool           hasLabel;
    PetscInt            numRegions;
    IS                  regionIS;
    PetscInt            *regions;
    PetscInt            *components;
    PetscInt            *numcompsperlabel;
    PetscReal           *values;
} FracDBC;

typedef struct {
    PetscInt     ng;                   /* Number of integration points */
    PetscInt     dim;                  /* Dimension */
    PetscInt     nodes;                /* Number of nodes */
    PetscReal    **nodecoords;         /* Global cordinates of vertices*/
    PetscReal    **int_point;           /* Centroid (in local cordinate) of the element used for integration*/
    PetscReal    *weight;               /* Integration point weight */
    PetscReal    **phi;                 /* 2D linear finite element basis function */
    PetscReal    ***dphi;               /* Derivative of basis function at integration point */
    PetscReal    ***dphi_r;               /* Derivative of basis function at integration point */
    PetscReal    ***J;                  /* Jacobian matrix of tranformation */
    PetscReal    ***invJ;               /* Jacobian of tranformation */
    PetscReal    *detJ;                 /* Determinant of the Jacobian = 2A */
    PetscReal    Volume;                 /* Area/size of triagular element */
} FracDFEElement;

typedef struct {
    PetscInt     dim;                  /* Dimension */
    PetscInt     nodes;                /* Number of nodes */
    PetscReal    *phi;                 /* 2D linear finite element basis function */
    PetscReal    Volume;                 /* Area/size of triagular element */
    PetscErrorCode          (*FracDLocalToGlobalCoordinate)(PetscReal*,PetscReal*,PetscReal**);
} FracDPointFEElement;

typedef struct {
    PetscInt     ng;                   /* Number of integration points */
    PetscInt     dim;                  /* Dimension */
    PetscInt     nodes;                /* Number of controlv volume face nodes */
    PetscInt     elemnodes;                /* Number of controlv volume face nodes */
    PetscReal    *n;                   /* Normal to control volume face */
    PetscReal    **facecoords;         /* Global cordinates of face vertices*/
    PetscReal    **elemcoords;         /* Global cordinates of face vertices*/
    PetscReal    *int_point;           /* Centroid (in global cordinate) of the element used for integration*/
    PetscReal    *phi;                 /* 2D linear finite element basis function */
    PetscReal    **dphi;               /* Derivative of basis function at integration point */
    PetscReal    faceArea;             /* Area of CV face */
    PetscReal    elemVolume;             /* Area of CV face */
    PetscReal    scale;             
    PetscErrorCode      (*FracDLocalToGlobalCoordinate)(PetscReal*,PetscReal*,PetscReal**);
} FracDCVFEFace;


#ifndef FracDStandardFEElement
extern PetscErrorCode FracD2DUpdateGMSHQuadrilateralPointShapeFunctionAndReferenceDerivative(PetscReal *phi, PetscReal dphi[2][4],PetscReal *eta);
extern PetscErrorCode FracD3DUpdateGMSHHexahedralPointShapeFunctionAndReferenceDerivative(PetscReal *phi, PetscReal dphi[2][4],PetscReal *eta);

extern PetscErrorCode FracD3DHexahedralLocalToGlobalCoordinate(PetscReal *X, PetscReal *eta, PetscReal **elemcoords);
extern PetscErrorCode FracD2DQuadrilateralLocalToGlobalCoordinate(PetscReal *X, PetscReal *eta, PetscReal **elemcoords);
extern PetscErrorCode FracDComputeLocalFECoordinates(PetscReal *eta_new, PetscReal *Xp, PetscReal *L,PetscReal **elemcoords, PetscInt dim, PetscErrorCode (*FracDCreateDMinusOneFEElement)(PetscReal*, PetscReal*, PetscReal*));

extern PetscErrorCode FracDPointFEElementCreate(PetscInt dim, FracDPointFEElement *e, FracDElementType element);
extern PetscErrorCode FracD3DTetrahedralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
extern PetscErrorCode FracD3DHexahedralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
extern PetscErrorCode FracD2DTrianglePointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
extern PetscErrorCode FracD2DQuadrilateralPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
extern PetscErrorCode FracD1DPointElementFE(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
extern PetscErrorCode FracDPointFEElementDestroy(FracDPointFEElement *e);
extern PetscErrorCode FracDFEElementCreate(PetscInt dim, FracDFEElement *e, FracDElementType elementType);
extern PetscErrorCode FracD3DFEElementCreate(FracDFEElement *e,FracDElementType elementType);
extern PetscErrorCode FracD2DFEElementCreate(FracDFEElement *e,FracDElementType elementType);
extern PetscErrorCode FracD1DFEElementCreate(FracDFEElement *e,FracDElementType elementType);
extern PetscErrorCode FracD3DReferenceTetrahedralElementFEDerivative(FracDFEElement *e);
extern PetscErrorCode FracD3DTetrahedralElementFE(PetscReal **coords, FracDFEElement *e);
extern PetscErrorCode FracD3DReferenceHexahedralElementFEDerivative(FracDFEElement *e);
extern PetscErrorCode FracD3DHexahedralElementFE(PetscReal **coords, FracDFEElement *e);
extern PetscErrorCode FracD2DReferenceTriangleElementFEDerivative(FracDFEElement *e);
extern PetscErrorCode FracD2DTriangleElementFE(PetscReal **coords, FracDFEElement *e);
extern PetscErrorCode FracD2DReferenceQuadrilateralElementFEDerivative(FracDFEElement *e);
extern PetscErrorCode FracD2DQuadrilateralElementFE(PetscReal **coords, FracDFEElement *e);
extern PetscErrorCode FracD1DReferenceElementFEDerivative(FracDFEElement *e);
extern PetscErrorCode FracD1DElementFE(PetscReal **coords, FracDFEElement *e);
extern PetscErrorCode FracDFEElementDestroy(FracDFEElement *e);


extern PetscErrorCode FracDCVFEFaceCreate(PetscInt dim, FracDCVFEFace *f,FracDElementType elementType);

extern PetscErrorCode FracD2DTriangleCVFEFace(PetscReal **elemcoords, PetscReal **facecoords, FracDCVFEFace *f);
extern PetscErrorCode FracD2DQuadrilateralCVFEFace(PetscReal **elemcoords, PetscReal **facecoords, FracDCVFEFace *f);






extern PetscErrorCode FracD3DTetrahedralCVFEFace(PetscReal **elemcoords, PetscReal **facecoords, FracDCVFEFace *f);
extern PetscErrorCode FracD3DHexahedralCVFEFace(PetscReal **elemcoords, PetscReal **facecoords, FracDCVFEFace *f);
extern PetscErrorCode FracDCVFEFaceDestroy(FracDCVFEFace *f);
extern PetscErrorCode FracDMatrixApplyDirichletBC(DM dm, Mat K, FracDBC *BC, PetscScalar diagonalvalue);
extern PetscErrorCode FracDResidualApplyDirichletBC(Vec residual, Vec V, FracDBC *BC);
extern PetscErrorCode FracD1DProjectFaceCoordinateDimensions(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim);
extern PetscErrorCode FracD2DProjectFaceCoordinateDimensions(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim);
extern PetscErrorCode FracD2DProjectFaceCoordinateDimensionsHexahedral(PetscReal **coords,PetscReal **coords1, PetscInt nodes, PetscInt ldim);
extern PetscErrorCode FracDResidualApplyNeumannBC_local(PetscReal *residual_local,PetscReal *bcvalues, FracDFEElement *e, PetscInt n);
extern PetscErrorCode FracDResidualApplyFENeumannBC(Vec residual, FracDBC *BC, FracDFEElement *elD, PetscErrorCode (*FracDCreateDMinusOneFEElement)(PetscReal**, FracDFEElement*),PetscErrorCode (*ProjectFaceCoordinates)(PetscReal**,PetscReal**,PetscInt,PetscInt));
extern PetscErrorCode FracDResidualApplyCVFENeumannBC(Vec residual,  PetscReal dt, FracDBC *BC, FracDFEElement *elD,PetscErrorCode (*FracDCreateDMinusOneFEElement)(PetscReal**, FracDFEElement*), PetscErrorCode (*ProjectFaceCoordinates)(PetscReal**,PetscReal**,PetscInt,PetscInt));
extern PetscErrorCode FracDResidualApplyConstraintOnGasSaturation(Vec residual, Vec Sg, Vec V);
extern PetscErrorCode FracDMatrixApplyConstraintOnGasSaturation(DM dm, Mat K, Vec V, PetscScalar diagonalvalue);


#endif
#endif /* FRACDFINITEELEMENT_H */
