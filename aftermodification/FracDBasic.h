#ifndef FRACDBASICS_H
#define FRACDBASICS_H
/*
 FracDBasic.h
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */
static const char banner[] = "\n\nFracD:\nNumerical implementation of multiphysics models.\n(c) 2016-2017  Chukwudi Chukwudozie chdozie@gmail.com\n\n";

typedef struct {
    Vec                  U;
    Vec                  T;
    Vec                  P;
    Vec                  Pw;
    Vec                  Po;
    Vec                  Pb;    /* Bubble point pressure */
    Vec                  oPb;    /* Bubble point pressure */
    Vec                  Rs;    /* Solution gas oil ratio */
    Vec                  dervRs;    /* Solution gas oil ratio */
    Vec                  oRs;    /* Solution gas oil ratio */
    Vec                  Pcow;
    Vec                  oPcow;
    Vec                  dervPcow;
    Vec                  Pg;
    Vec                  Pcog;
    Vec                  oPcog;
    Vec                  dervPcog;
    Vec                  Sw;
    Vec                  So;
    Vec                  Sg;
    Vec                  q;   /* fluid flow rate  */
    Vec                  V;   /* fluid velocity  */
    Vec                  oU;
    Vec                  oT;
    Vec                  oP;
    Vec                  oPw;
    Vec                  oPo;
    Vec                  oPg;
    Vec                  oSw;
    Vec                  oSo;
    Vec                  oSg;
    Vec                  oV;  /*  previous fluid velocity */
    Vec                  oq;  /*  fluid flow rate */
    Vec                  Fb;  /*  body force*/
    Vec                  QT;  /*  Uniform fluid source  */
    Vec                  QP;  /*  Uniform fluid source  */
    Vec                  FlowPacker;
    Vec                  Pbh;
    Vec                  oPbh;
    Vec                  Qwbh;
    Vec                  Qobh;
    Vec                  Qgbh;
    Vec                  QLbh;
    Vec                  QTbh;
    Vec                  SaturatedStateIndicator;
    Vec                  INDC1;
    Vec                  INDC2;
} FracDFields;

typedef struct {
    
    PetscReal            **Pcow_TableData;
    PetscReal            **Pcog_TableData;
    PetscInt             numwaterdatarow;                         /* Number of rows of data in input table */
    PetscInt             numgasdatarow;                         /* Number of rows of data in input table */
    char                 waterdatafilename[PETSC_MAX_PATH_LEN];
    char                 gasdatafilename[PETSC_MAX_PATH_LEN];
    PetscBool            PcowIsZero;
    PetscBool            PcogIsZero;
    PetscErrorCode      (*FracDUpDatePcow)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
    PetscErrorCode      (*FracDUpDatePcog)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
} FracDCapPress;

static const char *FracDRelPermModel_name[] = {
    "SINGLEPHASE",
    "STONE1",
    "STONE2",
    "FracDRelPermModel_name",
    "",
    0
};

typedef enum {
    SINGLEPHASE,
    STONE1,
    STONE2
} FracDRelPermModel;

typedef struct {
    FracDRelPermModel    model;
    PetscReal            **Krw_TableData;
    PetscReal            **Krow_TableData;
    PetscReal            **Krg_TableData;
    PetscReal            **Krog_TableData;
    PetscInt             numwaterdatarow;                         /* Number of rows of data in input table */
    PetscInt             numgasdatarow;                         /* Number of rows of data in input table */
    PetscReal            Swc;
    PetscReal            Sor;
    PetscReal            krc;
    PetscReal            stone_model_data[4];
    PetscReal            stone_model_data1[4];
    char                 waterdatafilename[PETSC_MAX_PATH_LEN];
    char                 gasdatafilename[PETSC_MAX_PATH_LEN];
    PetscBool            PcowInrelPermData;
    PetscBool            PcogInrelPermData;
    PetscBool            krowdataprovided;
    PetscBool            krogdataprovided;
    PetscErrorCode      (*FracDUpDateKrw)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
    PetscErrorCode      (*FracDUpDateKro)(PetscReal*, PetscReal, PetscReal, PetscReal, PetscReal**, PetscReal**, PetscReal**, PetscReal*, PetscInt, PetscInt);
    PetscErrorCode      (*FracDUpDateKrg)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
} FracDRelPerm;

static const char *FracDPVTModel_name[] = {
    "ANALYTICAL",
    "INTERPOLATION",
    "ANALYTICAL_AND_INTERPOLATION",
    "FracDPVTModel_name",
    "",
    0
};

typedef enum {
    ANALYTICAL,
    INTERPOLATION,
    ANALYTICAL_AND_INTERPOLATION
} FracDQTYModelType;

typedef struct {
    PetscReal            p_ref;       /* Reference pressure for fluid viscosity            */
    PetscReal            Cf;            /* Fluid compressibility: This should be related to rho_coeff       */
    FracDQTYModelType    FVFtype;
    PetscReal            B_ModelData[3];           /* 0: Value at reference point; 1: Coefficient; 2: reference point i.e reference pressure        */
    PetscReal            RateConversion;           /* This is important to convert gas rate to MMSccf/D using B and Rs        */
    PetscReal            **B_TableData;
    FracDQTYModelType    mutype;
    PetscReal            mu_ModelData[3];            /* 0: Value at reference point; 1: Coefficient; 2: reference point i.e reference pressure        */
    PetscReal            **mu_TableData;
    FracDQTYModelType    rhotype;         /* Use of empirircal or interpolated data for rho */
    PetscReal            rho_ModelData[3];           /* 0: Value at reference point; 1: Coefficient; 2: reference point i.e reference pressure        */
    PetscReal            **rho_TableData;            /* 0: independent variable; 1: dependnent variable 2: Coefficients/second derivatives        */
    PetscInt             numdatarow;                         /* Number of rows of data in input table */
    char                 datafilename[PETSC_MAX_PATH_LEN];
    PetscErrorCode       (*FracDUpDateFVF)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
    PetscErrorCode       (*FracDUpDateViscosity)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
    PetscErrorCode       (*FracDUpDateDensity)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
} FracDPVT;

typedef struct {
    PetscReal            ModelData[3];           /* 0: Value at reference point; 1: Coefficient; 2: reference point i.e reference pressure        */
    PetscReal            RateConversion;           /* This is important to convert gas rate to MMSccf/D using B and Rs        */
    PetscReal            **TableData;
    PetscReal            **TableDataInv;
    PetscInt             numdatarow;                         /* Number of rows of data in input table */
    char                 datafilename[PETSC_MAX_PATH_LEN];
    PetscErrorCode       (*FracDUpDateSolutionGasOilRatio)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
    PetscErrorCode       (*FracDUpDateBubblePoint)(PetscReal*, PetscReal, PetscReal, PetscReal**, PetscReal*, PetscInt);
} FracDPbRs;

typedef struct {
    Vec                  E,nu;          /* Young modulus and Poisson ratio */
    Vec                  relperm;       /* Relative Permeability */
    PetscReal            Coeffrelperm_w;   /* Water relative permeability coefficient            */
    Vec                  perm;          /* Permeability in m^2 muliply by 1e12 */
    Vec                  cond;          /* Thermal Conductivity in x-direction */
    Vec                  alpha;         /* Linear thermal expansion coef.  */
    Vec                  beta;          /* Biot's constant                 */
    Vec                  phi;           /* porosity                        */
    PetscReal            Cr;            /* Rock compressibility           */
    Vec                  Cp;            /* Specific heat capacity  of rock        */
    PetscReal            Cpw;           /* Specific heat capacity of fluid          */
    Vec                  rhos;          /* Rock density          */
    PetscReal            mu;            /* Fluid viscosity              */
    PetscReal            mu_w;          /* Water viscosity              */
    PetscReal            coeffmu_w;     /* Water viscosity coefficient            */
    PetscReal            mu_o;          /* Oil viscosity              */
    PetscReal            coeffmu_o;     /* Oil viscosity coefficient           */
    PetscReal            mu_g;          /* Gas viscosity              */
    PetscReal            coeffmu_g;     /* Gas viscosity coefficient           */
    PetscReal            rho;           /* Fluid density                */
    PetscReal            rho_w;         /* Water density                */
    PetscReal            coeffrho_w;    /* Water density coefficient            */
    PetscReal            rho_o;         /* Oil density                */
    PetscReal            coeffrho_o;    /* Oil density coefficient            */
    PetscReal            rho_g;         /* Gas density                */
    PetscReal            coeffrho_g;    /* Gas density coefficient            */
    PetscReal            Bw;            /* Water formation volume factor */
    PetscReal            Bo;            /* Oil formation volume factor */
    PetscReal            Bg;            /* Gas formation volume factor */
    PetscReal            *g;            /* Gravity */
    PetscReal            Cf;            /* Fluid compressibility            */
    PetscReal            Cwf;           /* Water compressibility            */
    PetscReal            Cof;           /* Oil compressibility            */
    Vec                  CellVolume;    /* Volume of each element          */
    Vec                  dualCellVolume;/* Volume of each element in dual mesh          */
    PetscReal            **BwCoeffInt;            /* Coefficients for water formation volume factor obtained from input table*/
    PetscReal            **BoCoeffInt;            /* Coefficients for oil formation volume factor obtained from input table */
    PetscReal            **BgCoeffInt;            /* Coefficients for gas formation volume factor obtained from input table */
    FracDPVT             WaterPVTData;
    FracDPVT             OilPVTData;
    FracDPVT             GasPVTData;
    FracDRelPerm         RelPermData;
    FracDCapPress        CapPressData;
    PetscReal            PhiData[3];
    FracDPbRs            SolutionGasOilData;
    char                 solutiongas_bubblepoint_filename[PETSC_MAX_PATH_LEN];
    PetscReal            BubblePointFixed;
    PetscReal            SolutionGasOilRatioFixed;
} FracDPpty;

static const char *FracDMeshrefine_name[] = {
    "UNIFORM",
    "SIZECONSTRAINED",
    "FracDMeshrefine_name",
    "",
    0
};

typedef enum {
    UNIFORM,
    SIZECONSTRAINED
} FracDMeshrefine;

static const char *FracD2DElasticity_name[] = {
    "PLANESTRESS",
    "PLANESTRAIN",
    "FracD2DElasticity_name",
    "",
    0
};

typedef enum {
    PLANESTRESS,
    PLANESTRAIN
} FracD2DElasticity;


static const char *FracDUnits_name[] = {
    "FIELDUNITS",
    "METRICUNITS",
    "FracD2DElasticity_name",
    "",
    0
};

typedef enum {
    FIELDUNITS,
    METRICUNITS
} FracDUnits;


static const char *FracDFluidSystem_name[] = {
    "SINGLEPHASELIQUID",
    "SINGLEPHASEGAS",
    "OILWATER",
    "OILWATERGAS",
    "FracDFluidSystem_name",
    "",
    0
};

typedef enum {
    SINGLEPHASELIQUID,
    SINGLEPHASEGAS,
    OILWATER,
    OILWATERGAS
} FracDFluidSystem;

typedef struct {
    PetscInt            *Count;   /* Water relative permeability coefficient            */
    PetscInt            numberWellsInProcessor;   /* Water relative permeability coefficient            */
    PetscInt            **WellInfo;            /* Rock compressibility           */
} FracDWellMeshData;

typedef struct {
    PetscInt            dim;
    PetscInt            nodes;
    DM                  plexScalNode;
    DM                  plexVecNode;
    DM                  plexScalCell;
    DM                  plexVecCell;
    DM                  MultiPhasePacker;
    DM                  WellRedun;
    FracDMeshrefine     meshrefinetype;
    PetscBool           meshrefine;
    PetscReal           refinementLimit;
    PetscBool           defaultmesh;
    PetscBool           simplexmesh;
    PetscBool           meshinterpolate;                        /* Generate intermediate mesh elements */
    char                meshfilename[PETSC_MAX_PATH_LEN];
    FracDFields         fields;
    FracDPpty           ppties;
    SNES                snesU;
    SNES                snesP;
    SNES                snesT;
    FracDBC             UBC;
    FracDBC             TBC;
    FracDBC             PBC;
    FracDBC             TractionBC;
    FracDBC             HeatFluxBC;
    FracDBC             FlowFluxBC;
    PetscInt            TotalFaceSets;
    PetscInt            *FaceSetIds;
    FracDFEElement      eD;
    FracDFEElement      elD;
    FracDFEElement      eDD;
    FracDElementType    elementType;
    FracDPointFEElement      epD;
    FracDCVFEFace       CVFEface;
    PetscErrorCode      (*FracDCreateDPointFEElement)(PetscReal **elemcoords, PetscReal *coords, FracDPointFEElement *e);
    PetscErrorCode      (*FracDCreateDFEElement)(PetscReal **coords, FracDFEElement *e);
    PetscErrorCode      (*FracDCreateDMinusOneFEElement)(PetscReal **coords, FracDFEElement *e);
    PetscErrorCode      (*FracDCreateDMinusOneFEElement1)(PetscReal **coords, FracDFEElement *e);
    PetscErrorCode      (*FracDCreateCVFEFace)(PetscReal **elemcoords, PetscReal **facecoords, FracDCVFEFace *f);
    PetscErrorCode      (*FracDProjectFaceCoordinateDimensions)(PetscReal**,PetscReal**, PetscInt, PetscInt);
    PetscErrorCode      (*FracDElasticityStiffnessMatrixLocal)(PetscReal*,PetscReal,PetscReal,FracDFEElement*);
    PetscErrorCode      (*FracDIsWellInElement)(PetscInt,PetscReal*,PetscReal**,PetscBool*);
    FracD2DElasticity   elasticity2DType;
    PetscReal           theta;
    PetscReal           timevalue;
    PetscReal           mintimevalue;
    PetscReal           maxtimevalue;
    PetscInt            maxtimestep;
    PetscReal           current_time;
    PetscInt            timestep;
    PetscInt            numWells;
    FracDWell           *well;
    PetscBool           verbose;
    FracDFluidSystem    fluid;
    PetscReal           P_ref;
    PetscReal           T_ref;
    PetscReal           S_ref;
    PetscReal           Sw_ref;
    PetscReal           So_ref;
    PetscReal           P_initial;
    PetscReal           T_initial;
    PetscReal           Sw_initial;
    PetscReal           Sg_initial;
    FracDUnits          Units;
    PetscReal           ConversionFactorBeta;
    PetscReal           ConversionFactorGamma;
    PetscReal           ConversionFactorAlpha;
    PetscReal           ConversionMMGasRate;
    PetscInt            BlockMatrixSize;
    FracDWellMeshData   WellinMeshData;
    PetscReal           SMALL_SATURATION;
    PetscReal           SMALL_PRESSURE;
} AppCtx;

extern PetscErrorCode FracDFindPointIn3DElement(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg);
extern PetscErrorCode FracDFindPointIn3DElement(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg);
extern PetscErrorCode FracDInitialize(AppCtx *bag);
extern PetscErrorCode FracDGetBagOptions(AppCtx *bag);
extern PetscErrorCode FracDCreateFluidPVTData(const char prefix[],FracDPVT *fluid);
extern PetscErrorCode FracDFluidPVTDataFunction(FracDPVT *fluid);
extern PetscErrorCode FracDProcessPVTData(AppCtx *bag);
extern PetscErrorCode FracDProcessRelPermCapillaryPressureData(FracDFluidSystem fluid, FracDRelPerm *relPerm, FracDCapPress *capPress);
extern PetscErrorCode FracDCreateMesh(AppCtx *bag);
extern PetscErrorCode FracDCreateFEShapeFunction(AppCtx *bag);
extern PetscErrorCode FracDSetMechanisMatrixType(AppCtx *bag);
extern PetscErrorCode FracDCreateBCLabels(AppCtx *bag, const char prefix[], FracDBC *BC);
extern PetscErrorCode FracDInitializeBoundaryConditions(AppCtx *bag);
extern PetscErrorCode FracDCreateDataSection(AppCtx *bag);
extern PetscErrorCode FracDCreatePackerDM(AppCtx *bag);
extern PetscErrorCode FracDCreateFields(AppCtx *bag,FracDFields *fields);
extern PetscErrorCode FracDInitializeWells(AppCtx *bag);
extern PetscErrorCode FracDFindPointIn3DHexahedral(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg);
extern PetscErrorCode FracDFindPointIn3DTetrahedral(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg);
extern PetscErrorCode FracDFindPointIn2DElement(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg);
extern PetscErrorCode FracDProcessWellBlockLocation(AppCtx *bag);
extern PetscErrorCode FracDCreateFlowMatix(AppCtx *bag, Mat *K, Mat *KPC, Vec X);
extern PetscErrorCode FracDInitializeSolvers(AppCtx *bag);
extern PetscErrorCode FracDGetResFluidProps(AppCtx *bag,FracDPpty *ppties);
extern PetscErrorCode FracDFinalize(AppCtx *bag);
extern PetscErrorCode FracDDestroyFields(AppCtx *bag,FracDFields *fields);
extern PetscErrorCode FracDDestroyBoundaryConditions(AppCtx *bag);
extern PetscErrorCode FracDInitializeUnitConversions(AppCtx *bag);
extern PetscErrorCode FracDFinalizePVTData(AppCtx *bag);
extern PetscErrorCode FracDFinalizeRelPermCapillaryPressureData(FracDFluidSystem fluid, FracDRelPerm *relPerm, FracDCapPress *capPress);
extern PetscErrorCode FracDFinalizeWells(AppCtx *bag);
extern PetscErrorCode FracDDestroySolvers(AppCtx *bag);
extern PetscErrorCode FracDDestroyResFluidProps(AppCtx *bag,FracDPpty *ppties);
extern PetscErrorCode FracDDestroyDMMesh(AppCtx *bag);
extern PetscErrorCode FracDDestroyFEShapeFunction(AppCtx *bag);
extern PetscErrorCode ThisismyVTKwriter(AppCtx *bag);
extern PetscErrorCode myWriteFunc(PetscObject vec,PetscViewer viewer);
extern PetscErrorCode FracDTimeStepPrepare(AppCtx *bag);
extern PetscErrorCode FracDTimeStepUpdate(AppCtx *bag);


#endif /* VFCOMMON_H */
