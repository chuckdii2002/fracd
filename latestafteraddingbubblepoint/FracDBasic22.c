/*
 FracD: Initialize the FracD code. 
 
 (c) 2016-2018 Chukwudi Chukwudozie chdozie@gmail.com
 */
//http://www.mcs.anl.gov/petsc/petsc-3.5/src/dm/impls/plex/plexindices.c
//    https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg29194.html
//    https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg28291.html
//https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg29173/ex_vtk_export.cc
//http://petsc-users.mcs.anl.narkive.com/Dq9TBbeo/nodes-coordinates-in-distributed-dmplex
//http://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex42.c.html
//http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/
//http://www.gnu.org/software/make/manual/make.html
//http://petsc-users.mcs.anl.narkive.com/h5ZaKuQS/related-to-dmplexvecsetclsoure
//http://129.123.73.40/~arlowry/Geodyn/Modeling/SeismicCycle/Codes_mac/pylith-2.0.3-darwin-10.6.8/src/pylith-2.0.3/libsrc/pylith/problems/Solver.cc
//https://www.mail-archive.com/petsc-users@mcs.anl.gov/msg19323.html
//http://petsc-users.mcs.anl.narkive.com/yAS75wnn/dmplex-example-with-manual-grid-layout
//https://searchcode.com/file/51581840/src/snes/examples/tutorials/ex52.c
//https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2016-June/029485.html
//http://129.123.73.40/~arlowry/Geodyn/Modeling/SeismicCycle/Codes_mac/pylith-2.0.3-darwin-10.6.8/src/pylith-2.0.3/unittests/libtests/faults/TestSlipFn.cc
//http://lists.mcs.anl.gov/pipermail/petsc-users/2014-February/020643.html
//http://lists.mcs.anl.gov/pipermail/petsc-users/2015-April/025045.html
//http://lists.mcs.anl.gov/pipermail/petsc-users/2013-September/018822.html
//http://web.mit.edu/tao-petsc_v3.5/.petsc-3.5.2.amd64_ubuntu1404/src/sys/classes/viewer/impls/vtk/vtkv.c.html
//https://xgitlab.cels.anl.gov/pguhur/petsc/commit/35ba5a26458ea69e18028af77d5eecb4c72e7ad8#diff-2
//http://lists.geodynamics.org/pipermail/cig-commits/2014-May/023539.html


//https://engineering.purdue.edu/ME608/webpage/project-reports/cvfem.pdf
//http://www.mathematik.uni-dortmund.de/~kuzmin/Transport.pdf
//http://math.stackexchange.com/questions/305642/how-to-find-surface-normal-of-a-triangle
//https://www.comsol.com/multiphysics/convection-diffusion-equation
//http://repository.up.ac.za/xmlui/bitstream/handle/2263/42877/kattoura_control_2014.pdf?sequence=1&isAllowed=y

//http://www.pefarrell.org/wp-content/uploads/2015/09/riseth2015.pdf
//http://www.mcs.anl.gov/petsc/petsc-current/src/sys/classes/random/examples/tutorials/ex2.c.html

/*
 To do:
 1. Check for duplication of dirichlet and neumann bc's e.g displacement and traction cannot be defined on same boundary labels
 REMEMBER: Each file (even those in the FracD directory ) is compiled individually before the tests are compiled. Therefore, each file should have all the necessary header files for complete compilation. Example, all files have FradWell.h.
 
 2. Error to tell if mesh has not been provided
 */


#include "petsc.h"
#include "FracDWell.h"
#include "FracDFiniteElement.h"
#if defined(PETSC_HAVE_CGNS)
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif
#include "FracDBasic.h"
#include "FracDMechanics.h"
#include "FracDHeatFlow.h"
//#include "FracDFluidFlow.h"
#include "FracDFlow.h"
#include "FracDComputations.h"

#undef __FUNCT__
#define __FUNCT__ "FracDGetBagOptions"
extern PetscErrorCode FracDGetBagOptions(AppCtx *bag)
{
    PetscErrorCode ierr;
    
    PetscFunctionBeginUser;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Meshing Problem Options", "");CHKERRQ(ierr);
    {
        bag->dim  = 2;
        ierr  = PetscOptionsInt("-dim","\n\tProblem dimension/The topological mesh dimension","",bag->dim,&bag->dim,PETSC_NULL);CHKERRQ(ierr);
        bag->verbose = PETSC_FALSE;
        ierr = PetscOptionsBool("-verbose", "Display most debug information about the computation", " ", bag->verbose, &bag->verbose, PETSC_NULL);CHKERRQ(ierr);
        
        bag->meshinterpolate = PETSC_TRUE;/*Interpolate creates edges from node and cell list. For now we will always interpolate since we need edges and faces in all our computations, especially for traction and flux integrations*/
        /*ierr = PetscOptionsBool("-meshinterpolate", "Generate intermediate mesh elements", " ", bag->meshinterpolate, &bag->meshinterpolate, PETSC_NULL);CHKERRQ(ierr);*/
        bag->meshrefine = PETSC_FALSE;
        ierr = PetscOptionsBool("-meshrefine", "Refine mesh elements", " ", bag->meshrefine, &bag->meshrefine, PETSC_NULL);CHKERRQ(ierr);
        bag->simplexmesh = PETSC_FALSE;
        ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", " ", bag->simplexmesh, &bag->simplexmesh, PETSC_NULL);CHKERRQ(ierr);
        bag->meshfilename[0] = '\0';
        ierr = PetscOptionsString("-meshfilename", "The mesh file", "Name of mesh file to be imported", bag->meshfilename, bag->meshfilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
        
        if (bag->dim  == 2)
            bag->elementType = TRIANGLE;
        else
            bag->elementType = TETRAHEDRAL;
        ierr = PetscOptionsEnum("-elementtype","\n\t\n\t finite element type","",FracDElementType_name,(PetscEnum)bag->elementType,(PetscEnum*)&bag->elementType,PETSC_NULL);CHKERRQ(ierr);
        
        if (bag->dim == 2 && (bag->elementType == TETRAHEDRAL || bag->elementType == HEXAHEDRAL)){
            SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Element type must be QUADRILATERAL or TRIANGLE, got %s in %s\n",FracDElementType_name[bag->elementType],__FUNCT__);
        }
        if (bag->dim == 3 && (bag->elementType == QUADRILATERAL || bag->elementType == TRIANGLE)){
            SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Element type must be TETRAHEDRAL or HEXAHEDRAL, got %s in %s\n",FracDElementType_name[bag->elementType],__FUNCT__);
        }
        
        bag->fluid = SINGLEPHASELIQUID;
        ierr = PetscOptionsEnum("-fluid","\n\t\n\t fluid system type","",FracDFluidSystem_name,(PetscEnum)bag->fluid,(PetscEnum*)&bag->fluid,PETSC_NULL);CHKERRQ(ierr);
        if(bag->fluid == SINGLEPHASELIQUID || bag->fluid == SINGLEPHASEGAS){
            bag->ppties.RelPermData.model = SINGLEPHASE;
        }
        if(bag->fluid == OILWATERGAS){
            bag->ppties.RelPermData.model = STONE1;
            ierr = PetscOptionsEnum("-relpermmodel","\n\t\n\t three phase relative permeability model","",FracDRelPermModel_name,(PetscEnum)bag->ppties.RelPermData.model,(PetscEnum*)&bag->ppties.RelPermData.model,PETSC_NULL);CHKERRQ(ierr);
        }
        if (bag->dim == 2)
        {
            bag->elasticity2DType = PLANESTRAIN;
            ierr = PetscOptionsEnum("-elasticity2dtype","\n\t\n\t mesh refinement type","",FracD2DElasticity_name,(PetscEnum)bag->elasticity2DType,(PetscEnum*)&bag->elasticity2DType,PETSC_NULL);CHKERRQ(ierr);
        }
        /*        Intialize boundary options    */
        bag->UBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-UBC", "Mesh has points for displacment BC", " ", bag->UBC.hasLabel, &bag->UBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        bag->TractionBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-TractionBC", "Mesh has points for traction BC", " ", bag->TractionBC.hasLabel, &bag->TractionBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        bag->PBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-PBC", "Mesh has points for pressure BC", " ", bag->PBC.hasLabel, &bag->PBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        bag->FlowFluxBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-FlowFluxBC", "Mesh has points for fluid flux BC", " ", bag->FlowFluxBC.hasLabel, &bag->FlowFluxBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        bag->TBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-TBC", "Mesh has points for temperature BC", " ", bag->TBC.hasLabel, &bag->TBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        bag->HeatFluxBC.hasLabel = PETSC_FALSE;
        ierr = PetscOptionsBool("-HeatfluxBC", "Mesh has points for heat flux BC", " ", bag->HeatFluxBC.hasLabel, &bag->HeatFluxBC.hasLabel, PETSC_NULL);CHKERRQ(ierr);
        
        bag->Units = METRICUNITS;
        ierr = PetscOptionsEnum("-units","\n\t\n\t simulation units","",FracDUnits_name,(PetscEnum)bag->Units,(PetscEnum*)&bag->Units,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "FracDCreateFluidPVTData"
extern PetscErrorCode FracDCreateFluidPVTData(const char prefix[],FracDPVT *fluid)
{
    PetscErrorCode ierr;
    FILE           *fluidfile;
    PetscInt       i,rank;
    PetscReal      p,B,mu,rho;
    
    PetscFunctionBegin;
    
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    fluid->numdatarow = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,prefix,"\n\n FracD: Fluid PVT Data:","");CHKERRQ(ierr);
    {
        ierr = PetscOptionsEnum("-FVFmodel","\n\t\n\t fluid fvf model type","",FracDPVTModel_name,(PetscEnum)fluid->FVFtype,(PetscEnum*)&fluid->FVFtype,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnum("-viscmodel","\n\t\n\t  fluid viscosity model type","",FracDPVTModel_name,(PetscEnum)fluid->mutype,(PetscEnum*)&fluid->mutype,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnum("-densmodel","\n\t\n\t  fluid density model type","",FracDPVTModel_name,(PetscEnum)fluid->rhotype,(PetscEnum*)&fluid->rhotype,PETSC_NULL);CHKERRQ(ierr);
        if(fluid->FVFtype == ANALYTICAL || fluid->FVFtype == ANALYTICAL_AND_INTERPOLATION){
            ierr = PetscOptionsReal("-refFVF","\n\t fluid FVF at reference pressure","",fluid->B_ModelData[0],&fluid->B_ModelData[0],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-compress","\n\t fluid compressibility","",fluid->B_ModelData[1],&fluid->B_ModelData[1],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-preffvf","\n\t Reference pressure for fluid FVF","",fluid->B_ModelData[2],&fluid->B_ModelData[2],PETSC_NULL);CHKERRQ(ierr);
            fluid->B_ModelData[1]  = -1.*fluid->B_ModelData[1];
        }
        if(fluid->mutype == ANALYTICAL || fluid->mutype == ANALYTICAL_AND_INTERPOLATION){
            ierr = PetscOptionsReal("-refvisc","\n\t fluid viscosity at reference pressure","",fluid->mu_ModelData[0],&fluid->mu_ModelData[0],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-visccoeff","\n\t fluid viscosity coefficient","",fluid->mu_ModelData[1],&fluid->mu_ModelData[1],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-prefvisc","\n\t Reference pressure for fluid viscosity","",fluid->mu_ModelData[2],&fluid->mu_ModelData[2],PETSC_NULL);CHKERRQ(ierr);
        }
        if(fluid->rhotype == ANALYTICAL || fluid->rhotype == ANALYTICAL_AND_INTERPOLATION){
            ierr = PetscOptionsReal("-refdens","\n\t fluid density at reference pressure","",fluid->rho_ModelData[0],&fluid->rho_ModelData[0],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-denscoeff","\n\t fluid density coefficient","",fluid->rho_ModelData[1],&fluid->rho_ModelData[1],PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-prefdens","\n\t Reference pressure for fluid density","",fluid->rho_ModelData[2],&fluid->rho_ModelData[2],PETSC_NULL);CHKERRQ(ierr);
        }
        if(fluid->FVFtype != ANALYTICAL || fluid->mutype != ANALYTICAL || fluid->rhotype != ANALYTICAL){
            fluid->datafilename[0] = '\0';
            ierr = PetscOptionsString("-PVTfilename", "fluid PVT data file name", "Name of fluid PVT file to be imported", fluid->datafilename, fluid->datafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
            if(!rank){
                ierr   = PetscFOpen(PETSC_COMM_SELF,fluid->datafilename,"r",&fluidfile);CHKERRQ(ierr);CHKERRQ(ierr);
                while(fscanf(fluidfile, "%le%le%le%le", (double*)&p, (double*)&B, (double*)&mu, (double*)&rho) == 4){
                    fluid->numdatarow++;
                }
                fclose(fluidfile);
            }
            MPI_Bcast(&fluid->numdatarow,1,MPIU_INT,0,PETSC_COMM_WORLD);
            fluid->B_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
            fluid->mu_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
            fluid->rho_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
            for(i = 0; i < 3; i++)
            {
                fluid->B_TableData[i] = (PetscReal *)malloc(fluid->numdatarow * sizeof(PetscReal));
                fluid->mu_TableData[i] = (PetscReal *)malloc(fluid->numdatarow * sizeof(PetscReal));
                fluid->rho_TableData[i] = (PetscReal *)malloc(fluid->numdatarow * sizeof(PetscReal));
            }
            for(i = 0; i < fluid->numdatarow; i++)
            {
                fluid->mu_TableData[2][i] = fluid->rho_TableData[2][i] = fluid->B_TableData[2][i] = 0.;
            }
            if(!rank){
                ierr   = PetscFOpen(PETSC_COMM_SELF,fluid->datafilename,"r",&fluidfile);CHKERRQ(ierr);
                for(i = 0; i < fluid->numdatarow; i++){
                    fscanf(fluidfile, "%le%le%le%le", (double*)&fluid->B_TableData[0][i], (double*)&fluid->B_TableData[1][i], (double*)&fluid->mu_TableData[1][i], (double*)&fluid->rho_TableData[1][i]);
                    fluid->mu_TableData[0][i] = fluid->rho_TableData[0][i] = fluid->B_TableData[0][i];
                }
                fclose(fluidfile);
            }
            for(i = 0; i < 3; i++){
                MPI_Bcast(&fluid->B_TableData[i][0],fluid->numdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
                MPI_Bcast(&fluid->mu_TableData[i][0],fluid->numdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
                MPI_Bcast(&fluid->rho_TableData[i][0],fluid->numdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            }
            fluid->B_ModelData[0] = fluid->B_ModelData[0]*fluid->RateConversion;
            if(fluid->FVFtype != ANALYTICAL){
                for(i = 0; i < fluid->numdatarow; i++){
                    fluid->B_TableData[1][i] = fluid->B_TableData[1][i]*fluid->RateConversion;
                }
                ierr = FracDComputeCubicSplineInterpolationCoefficients(fluid->B_TableData,fluid->numdatarow);CHKERRQ(ierr);
            }
            if(fluid->mutype != ANALYTICAL){
                ierr = FracDComputeCubicSplineInterpolationCoefficients(fluid->mu_TableData,fluid->numdatarow);CHKERRQ(ierr);
            }
            if(fluid->rhotype != ANALYTICAL){
                ierr = FracDComputeCubicSplineInterpolationCoefficients(fluid->rho_TableData,fluid->numdatarow);CHKERRQ(ierr);
            }
        }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFluidPVTDataFunction"
extern PetscErrorCode FracDFluidPVTDataFunction(FracDPVT *fluid)
{
    PetscFunctionBegin;
    switch (fluid->FVFtype) {
        case 0:
        fluid->FracDUpDateFVF = FracDInterpolateUsingAnalyticalModel;
        break;
        case 1:
        fluid->FracDUpDateFVF = FracDInterpolateUsingCubicSpline;
        break;
        case 2:
        fluid->FracDUpDateFVF = FracDInterpolateUsingAnalyticalAndCubicSpline;
        break;
    }
    switch (fluid->mutype) {
        case 0:
        fluid->FracDUpDateViscosity = FracDInterpolateUsingAnalyticalModel;
        break;
        case 1:
        fluid->FracDUpDateViscosity = FracDInterpolateUsingCubicSpline;
        break;
        case 2:
        fluid->FracDUpDateViscosity = FracDInterpolateUsingAnalyticalAndCubicSpline;
        break;
    }
    switch (fluid->rhotype) {
        case 0:
        fluid->FracDUpDateDensity = FracDInterpolateUsingAnalyticalModel;
        break;
        case 1:
        fluid->FracDUpDateDensity = FracDInterpolateUsingCubicSpline;
        break;
        case 2:
        fluid->FracDUpDateDensity = FracDInterpolateUsingAnalyticalAndCubicSpline;
        break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDProcessPVTData"
extern PetscErrorCode FracDProcessPVTData(AppCtx *bag)
{
    PetscErrorCode ierr;
    char           prefix[PETSC_MAX_PATH_LEN+1];
    PetscInt       i,rank;
    PetscReal      p,Rs;
    FILE           *rs_bubble_point_file;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    
    bag->ppties.WaterPVTData.FVFtype = ANALYTICAL;
    bag->ppties.WaterPVTData.B_ModelData[0] = 1.;
    bag->ppties.WaterPVTData.B_ModelData[1] = 0;
    bag->ppties.WaterPVTData.B_ModelData[2] = 0;
    bag->ppties.WaterPVTData.mutype = ANALYTICAL;
    bag->ppties.WaterPVTData.mu_ModelData[0] = 1;
    bag->ppties.WaterPVTData.mu_ModelData[1] = 0;
    bag->ppties.WaterPVTData.mu_ModelData[2] = 0;
    bag->ppties.WaterPVTData.rhotype = ANALYTICAL;
    bag->ppties.WaterPVTData.rho_ModelData[0] = 1;
    bag->ppties.WaterPVTData.rho_ModelData[1] = 0;
    bag->ppties.WaterPVTData.rho_ModelData[2] = 0;
    if(bag->fluid == SINGLEPHASELIQUID || bag->fluid == OILWATER || bag->fluid == OILWATERGAS){
        strcpy(prefix,"water_");
        ierr = FracDCreateFluidPVTData(prefix,&bag->ppties.WaterPVTData);CHKERRQ(ierr);
    }
    ierr = FracDFluidPVTDataFunction(&bag->ppties.WaterPVTData);CHKERRQ(ierr);
    
    bag->ppties.OilPVTData.FVFtype = ANALYTICAL;
    bag->ppties.OilPVTData.B_ModelData[0] = 1.;
    bag->ppties.OilPVTData.B_ModelData[1] = 0;
    bag->ppties.OilPVTData.B_ModelData[2] = 0;
    bag->ppties.OilPVTData.mutype = ANALYTICAL;
    bag->ppties.OilPVTData.mu_ModelData[0] = 1;
    bag->ppties.OilPVTData.mu_ModelData[1] = 0;
    bag->ppties.OilPVTData.mu_ModelData[2] = 0;
    bag->ppties.OilPVTData.rhotype = ANALYTICAL;
    bag->ppties.OilPVTData.rho_ModelData[0] = 1;
    bag->ppties.OilPVTData.rho_ModelData[1] = 0;
    bag->ppties.OilPVTData.rho_ModelData[2] = 0;
    if(bag->fluid == OILWATER || bag->fluid == OILWATERGAS){
        strcpy(prefix,"oil_");
        ierr = FracDCreateFluidPVTData(prefix,&bag->ppties.OilPVTData);CHKERRQ(ierr);
    }
    ierr = FracDFluidPVTDataFunction(&bag->ppties.OilPVTData);CHKERRQ(ierr);
    
    bag->ppties.GasPVTData.FVFtype = ANALYTICAL;
    bag->ppties.GasPVTData.B_ModelData[0] = 1.;
    bag->ppties.GasPVTData.B_ModelData[1] = 0;
    bag->ppties.GasPVTData.B_ModelData[2] = 0;
    bag->ppties.GasPVTData.mutype = ANALYTICAL;
    bag->ppties.GasPVTData.mu_ModelData[0] = 1;
    bag->ppties.GasPVTData.mu_ModelData[1] = 0;
    bag->ppties.GasPVTData.mu_ModelData[2] = 0;
    bag->ppties.GasPVTData.rhotype = ANALYTICAL;
    bag->ppties.GasPVTData.rho_ModelData[0] = 1;
    bag->ppties.GasPVTData.rho_ModelData[1] = 0;
    bag->ppties.GasPVTData.rho_ModelData[2] = 0;
    if(bag->fluid == SINGLEPHASEGAS || bag->fluid == OILWATERGAS){
        strcpy(prefix,"gas_");
        ierr = FracDCreateFluidPVTData(prefix,&bag->ppties.GasPVTData);CHKERRQ(ierr);
    }
    ierr = FracDFluidPVTDataFunction(&bag->ppties.GasPVTData);CHKERRQ(ierr);
    
    if (bag->fluid == OILWATERGAS){
        bag->ppties.SolutionGasOilData.numdatarow = 0;
        ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\n FracD: Bubble Point & Solution Gas-Oil Ratio PVT Data:","");CHKERRQ(ierr);
        {
            bag->ppties.SolutionGasOilData.datafilename[0] = '\0';
            ierr = PetscOptionsString("-RsBubblepointfilename", "solution gas-oil ratio file name", "Name of solution gas-oil ratio file to be imported", bag->ppties.SolutionGasOilData.datafilename, bag->ppties.SolutionGasOilData.datafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
        }
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
        
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,bag->ppties.SolutionGasOilData.datafilename,"r",&rs_bubble_point_file);CHKERRQ(ierr);CHKERRQ(ierr);
            while(fscanf(rs_bubble_point_file, "%le%le", (double*)&p, (double*)&Rs) == 2){
                bag->ppties.SolutionGasOilData.numdatarow++;
            }
            fclose(rs_bubble_point_file);
        }
        MPI_Bcast(&bag->ppties.SolutionGasOilData.numdatarow,1,MPIU_INT,0,PETSC_COMM_WORLD);
        bag->ppties.SolutionGasOilData.TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        bag->ppties.SolutionGasOilData.TableDataInv = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        for(i = 0; i < 3; i++){
            bag->ppties.SolutionGasOilData.TableData[i] = (PetscReal *)malloc(bag->ppties.SolutionGasOilData.numdatarow * sizeof(PetscReal));
            bag->ppties.SolutionGasOilData.TableDataInv[i] = (PetscReal *)malloc(bag->ppties.SolutionGasOilData.numdatarow * sizeof(PetscReal));
        }
        for(i = 0; i < bag->ppties.SolutionGasOilData.numdatarow; i++){
            bag->ppties.SolutionGasOilData.TableData[2][i] = 0.;
            bag->ppties.SolutionGasOilData.TableDataInv[2][i] = 0.;
        }
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,bag->ppties.SolutionGasOilData.datafilename,"r",&rs_bubble_point_file);CHKERRQ(ierr);
            for(i = 0; i < bag->ppties.SolutionGasOilData.numdatarow; i++){
                fscanf(rs_bubble_point_file, "%le%le", (double*)&bag->ppties.SolutionGasOilData.TableData[0][i], (double*)&bag->ppties.SolutionGasOilData.TableData[1][i]);
            }
            fclose(rs_bubble_point_file);
        }
        for(i = 0; i < 3; i++){
            MPI_Bcast(&bag->ppties.SolutionGasOilData.TableData[i][0],bag->ppties.SolutionGasOilData.numdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            MPI_Bcast(&bag->ppties.SolutionGasOilData.TableData[i][1],bag->ppties.SolutionGasOilData.numdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
        }
        for(i = 0; i < bag->ppties.SolutionGasOilData.numdatarow; i++){
            bag->ppties.SolutionGasOilData.TableData[1][i] = bag->ppties.SolutionGasOilData.TableData[1][i]/bag->ppties.SolutionGasOilData.RateConversion;
            bag->ppties.SolutionGasOilData.TableDataInv[0][i] = bag->ppties.SolutionGasOilData.TableData[1][i];
            bag->ppties.SolutionGasOilData.TableDataInv[1][i] = bag->ppties.SolutionGasOilData.TableData[0][i];
        }
        ierr = FracDComputeCubicSplineInterpolationCoefficients(bag->ppties.SolutionGasOilData.TableData,bag->ppties.SolutionGasOilData.numdatarow);CHKERRQ(ierr);
        ierr = FracDComputeCubicSplineInterpolationCoefficients(bag->ppties.SolutionGasOilData.TableDataInv,bag->ppties.SolutionGasOilData.numdatarow);CHKERRQ(ierr);
        
        bag->ppties.SolutionGasOilData.FracDUpDateSolutionGasOilRatio = FracDInterpolateUsingCubicSpline;
    }
    else{
        bag->ppties.SolutionGasOilData.ModelData[0] = bag->ppties.SolutionGasOilData.ModelData[1] = bag->ppties.SolutionGasOilData.ModelData[2]= 0.;
        bag->ppties.SolutionGasOilData.numdatarow = 0;
        bag->ppties.SolutionGasOilData.FracDUpDateSolutionGasOilRatio = FracDZeros;
    }
    bag->ppties.SolutionGasOilData.FracDUpDateBubblePoint = FracDInterpolateUsingAnalyticalAndCubicSpline;
    bag->SMALL_SATURATION    = 1e-2;
    bag->SMALL_PRESSURE      = 1e-6;
    if ( (bag->fluid == OILWATERGAS || bag->fluid == OILWATER) &&  bag->ppties.OilPVTData.FVFtype != ANALYTICAL){
        bag->SMALL_PRESSURE      = bag->ppties.OilPVTData.B_TableData[0][1]/100;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDProcessRelPermCapillaryPressureData"
extern PetscErrorCode FracDProcessRelPermCapillaryPressureData(FracDFluidSystem fluid, FracDRelPerm *relPerm, FracDCapPress *capPress)
{
    PetscErrorCode ierr;
    FILE           *waterrelpermfile,*gasrelpermfile;
    FILE           *watercappressfile=NULL,*gascappressfile=NULL;
    PetscInt       i,rank;
    PetscReal      S_w,kr_w,kr_ow,P_cow;
    PetscReal      S_g,kr_g,kr_og,P_cog;
    
    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    relPerm->numwaterdatarow = capPress->numwaterdatarow = 0;
    relPerm->numgasdatarow = capPress->numgasdatarow = 0;
    if ((fluid == OILWATER) || (fluid == OILWATERGAS)){
        ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\n FracD: Oil/Water PVT Data:","");CHKERRQ(ierr);
        {
            relPerm->PcowInrelPermData = PETSC_FALSE;
            ierr = PetscOptionsBool("-PcowInrelPermData", "If Oil/water capillary pressure data is in relative permeability file ", " ", relPerm->PcowInrelPermData, &relPerm->PcowInrelPermData, PETSC_NULL);CHKERRQ(ierr);
            relPerm->waterdatafilename[0] = '\0';
            ierr = PetscOptionsString("-oilwaterrelpermfilename", "Oil/Water relative permeability data file name", "Name of oil/water relative permeability file to be imported", relPerm->waterdatafilename, relPerm->waterdatafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
            if(!relPerm->PcowInrelPermData){
                capPress->waterdatafilename[0] = '\0';
                ierr = PetscOptionsString("-oilwatercappressfilename", "Oil/Water capillary pressure data file name", "Name of oil/water capillary pressure file to be imported", capPress->waterdatafilename, capPress->waterdatafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
            }
        }
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,relPerm->waterdatafilename,"r",&waterrelpermfile);CHKERRQ(ierr);CHKERRQ(ierr);
            if(relPerm->PcowInrelPermData){
                while(fscanf(waterrelpermfile, "%le%le%le%le", (double*)&S_w, (double*)&kr_w, (double*)&kr_ow, (double*)&P_cow) == 4){
                    relPerm->numwaterdatarow++;
                }
                capPress->numwaterdatarow = relPerm->numwaterdatarow;
            }
            else{
                while(fscanf(waterrelpermfile, "%le%le%le", (double*)&S_w, (double*)&kr_w, (double*)&kr_ow) == 3){
                    relPerm->numwaterdatarow++;
                }
                ierr   = PetscFOpen(PETSC_COMM_SELF,capPress->waterdatafilename,"r",&watercappressfile);CHKERRQ(ierr);
                while(fscanf(watercappressfile, "%le%le", (double*)&S_w, (double*)&P_cow) == 2){
                    capPress->numwaterdatarow++;
                }
                fclose(watercappressfile);
            }
            fclose(waterrelpermfile);
        }
        MPI_Bcast(&relPerm->numwaterdatarow ,1,MPIU_INT,0,PETSC_COMM_WORLD);
        MPI_Bcast(&capPress->numwaterdatarow ,1,MPIU_INT,0,PETSC_COMM_WORLD);
        relPerm->Krw_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        relPerm->Krow_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        for(i = 0; i < 3; i++)
        {
            relPerm->Krw_TableData[i] = (PetscReal *)malloc(relPerm->numwaterdatarow * sizeof(PetscReal));
            relPerm->Krow_TableData[i] = (PetscReal *)malloc(relPerm->numwaterdatarow * sizeof(PetscReal));
        }
        capPress->Pcow_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        for(i = 0; i < 3; i++)
        {
            capPress->Pcow_TableData[i] = (PetscReal *)malloc(capPress->numwaterdatarow * sizeof(PetscReal));
        }
        
        
        
        
        
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,relPerm->waterdatafilename,"r",&waterrelpermfile);CHKERRQ(ierr);
            if(relPerm->PcowInrelPermData){
                for(i = 0; i < relPerm->numwaterdatarow; i++){
                    fscanf(waterrelpermfile, "%le%le%le%le", (double*)&relPerm->Krw_TableData[0][i], (double*)&relPerm->Krw_TableData[1][i], (double*)&relPerm->Krow_TableData[1][i], (double*)&capPress->Pcow_TableData[1][i]);
                    capPress->Pcow_TableData[0][i] = relPerm->Krow_TableData[0][i] = relPerm->Krw_TableData[0][i];
                }
            }
            else{
                for(i = 0; i < relPerm->numwaterdatarow; i++){
                    fscanf(waterrelpermfile, "%le%le%le", (double*)&relPerm->Krw_TableData[0][i], (double*)&relPerm->Krw_TableData[1][i], (double*)&relPerm->Krow_TableData[1][i]);
                    relPerm->Krow_TableData[0][i] = relPerm->Krw_TableData[0][i];
                }
                ierr   = PetscFOpen(PETSC_COMM_SELF,capPress->waterdatafilename,"r",&watercappressfile);CHKERRQ(ierr);
                for(i = 0; i < capPress->numwaterdatarow; i++){
                    fscanf(watercappressfile, "%le%le", (double*)&capPress->Pcow_TableData[0][i], (double*)&capPress->Pcow_TableData[1][i]);
                }
                fclose(watercappressfile);
            }
            fclose(waterrelpermfile);
        }
        for(i = 0; i < 3; i++){
            MPI_Bcast(&relPerm->Krw_TableData[i][0],relPerm->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            MPI_Bcast(&relPerm->Krow_TableData[i][0],relPerm->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            MPI_Bcast(&capPress->Pcow_TableData[i][0],capPress->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
        }
        relPerm->Swc = relPerm->Krw_TableData[0][0];
        relPerm->Sor = 1.0-relPerm->Krw_TableData[0][relPerm->numwaterdatarow-2];
        ierr = FracDComputeCubicSplineInterpolationCoefficients(relPerm->Krw_TableData,relPerm->numwaterdatarow);CHKERRQ(ierr);
        ierr = FracDComputeCubicSplineInterpolationCoefficients(relPerm->Krow_TableData,relPerm->numwaterdatarow);CHKERRQ(ierr);
        ierr = FracDComputeCubicSplineInterpolationCoefficients(capPress->Pcow_TableData,capPress->numwaterdatarow);CHKERRQ(ierr);
    }
    
    
    
    
    
    
    if (fluid == OILWATERGAS){
        ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\n FracD: Gas/Oil relperm/capillary data:","");CHKERRQ(ierr);
        {
            relPerm->PcogInrelPermData = PETSC_FALSE;
            ierr = PetscOptionsBool("-PcogInrelPermData", "If Gas/Oil capillary pressure data is in relative permeability file ", " ", relPerm->PcogInrelPermData, &relPerm->PcogInrelPermData, PETSC_NULL);CHKERRQ(ierr);
            relPerm->gasdatafilename[0] = '\0';
            ierr = PetscOptionsString("-gasoilrelpermfilename", "Gas/Oil relative permeability data file name", "Name of gas/oil relative permeability file to be imported", relPerm->gasdatafilename, relPerm->gasdatafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
            if(!relPerm->PcogInrelPermData){
                capPress->gasdatafilename[0] = '\0';
                ierr = PetscOptionsString("-gasoilcappressfilename", "Gas/Oil capillary pressure data file name", "Name of gas/oil capillary pressure file to be imported", capPress->gasdatafilename, capPress->gasdatafilename, PETSC_MAX_PATH_LEN, PETSC_NULL);CHKERRQ(ierr);
            }
        }
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,relPerm->gasdatafilename,"r",&gasrelpermfile);CHKERRQ(ierr);CHKERRQ(ierr);
            if(relPerm->PcogInrelPermData){
                while(fscanf(gasrelpermfile, "%le%le%le%le", (double*)&S_g, (double*)&kr_g, (double*)&kr_og, (double*)&P_cog) == 4){
                    relPerm->numgasdatarow++;
                }
                capPress->numgasdatarow = relPerm->numgasdatarow;
            }
            else{
                while(fscanf(gasrelpermfile, "%le%le%le", (double*)&S_g, (double*)&kr_g, (double*)&kr_og) == 3){
                    relPerm->numgasdatarow++;
                }
                ierr   = PetscFOpen(PETSC_COMM_SELF,capPress->gasdatafilename,"r",&gascappressfile);CHKERRQ(ierr);
                while(fscanf(gascappressfile, "%le%le", (double*)&S_g, (double*)&P_cog) == 2){
                    capPress->numgasdatarow++;
                }
                fclose(gascappressfile);
            }
            fclose(gasrelpermfile);
        }
        MPI_Bcast(&relPerm->numgasdatarow ,1,MPIU_INT,0,PETSC_COMM_WORLD);
        MPI_Bcast(&capPress->numgasdatarow ,1,MPIU_INT,0,PETSC_COMM_WORLD);
        relPerm->Krg_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        relPerm->Krog_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        for(i = 0; i < 3; i++)
        {
            relPerm->Krg_TableData[i] = (PetscReal *)malloc(relPerm->numgasdatarow * sizeof(PetscReal));
            relPerm->Krog_TableData[i] = (PetscReal *)malloc(relPerm->numgasdatarow * sizeof(PetscReal));
        }
        capPress->Pcog_TableData = (PetscReal **)malloc(3 * sizeof(PetscReal *));
        for(i = 0; i < 3; i++)
        {
            capPress->Pcog_TableData[i] = (PetscReal *)malloc(capPress->numgasdatarow * sizeof(PetscReal));
        }
        if(!rank){
            ierr   = PetscFOpen(PETSC_COMM_SELF,relPerm->gasdatafilename,"r",&gasrelpermfile);CHKERRQ(ierr);
            if(relPerm->PcogInrelPermData){
                for(i = 0; i < relPerm->numgasdatarow; i++){
                    fscanf(gasrelpermfile, "%le%le%le%le", (double*)&relPerm->Krg_TableData[0][i], (double*)&relPerm->Krg_TableData[1][i], (double*)&relPerm->Krog_TableData[1][i], (double*)&capPress->Pcog_TableData[1][i]);
                    capPress->Pcog_TableData[0][i] = relPerm->Krog_TableData[0][i] = relPerm->Krg_TableData[0][i];
                }
            }
            else{
                for(i = 0; i < relPerm->numgasdatarow; i++){
                    fscanf(gasrelpermfile, "%le%le%le", (double*)&relPerm->Krg_TableData[0][i], (double*)&relPerm->Krg_TableData[1][i], (double*)&relPerm->Krog_TableData[1][i]);
                    relPerm->Krog_TableData[0][i] = relPerm->Krg_TableData[0][i];
                }
                ierr   = PetscFOpen(PETSC_COMM_SELF,capPress->gasdatafilename,"r",&gascappressfile);CHKERRQ(ierr);
                for(i = 0; i < capPress->numgasdatarow; i++){
                    fscanf(gascappressfile, "%le%le", (double*)&capPress->Pcog_TableData[0][i], (double*)&capPress->Pcog_TableData[1][i]);
                }
                fclose(gascappressfile);
            }
            fclose(gasrelpermfile);
        }
        for(i = 0; i < 3; i++){
            MPI_Bcast(&relPerm->Krg_TableData[i][0],relPerm->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            MPI_Bcast(&relPerm->Krog_TableData[i][0],relPerm->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
            MPI_Bcast(&capPress->Pcog_TableData[i][0],capPress->numwaterdatarow,MPIU_REAL,0,PETSC_COMM_WORLD);
        }
        ierr = FracDComputeCubicSplineInterpolationCoefficients(relPerm->Krg_TableData,relPerm->numgasdatarow);CHKERRQ(ierr);
        ierr = FracDComputeCubicSplineInterpolationCoefficients(relPerm->Krog_TableData,relPerm->numgasdatarow);CHKERRQ(ierr);
        ierr = FracDComputeCubicSplineInterpolationCoefficients(capPress->Pcog_TableData,capPress->numgasdatarow);CHKERRQ(ierr);
        
    }
    if(fluid == SINGLEPHASELIQUID)
    {
        relPerm->FracDUpDateKrw = FracDOnes;
        relPerm->FracDUpDateKro = FracDZeros1;
        relPerm->FracDUpDateKrg = FracDZeros;
    }
    if(fluid == SINGLEPHASEGAS)
    {
        relPerm->FracDUpDateKrw = FracDZeros;
        relPerm->FracDUpDateKrg = FracDOnes;
        relPerm->FracDUpDateKro = FracDZeros1;
    }
    if(fluid == OILWATER)
    {
        relPerm->FracDUpDateKrw = FracDInterpolateUsingCubicSpline;
        relPerm->FracDUpDateKro = FracDOilWaterKrow;
        relPerm->FracDUpDateKrg = FracDZeros;
    }
    if(fluid == OILWATERGAS)
    {
        relPerm->FracDUpDateKrw = FracDInterpolateUsingCubicSpline;
        relPerm->FracDUpDateKrg = FracDInterpolateUsingCubicSpline;
        if(relPerm->model == STONE1){
            relPerm->FracDUpDateKro = FracDStone1Model;
            relPerm->stone_model_data[0] = relPerm->Krw_TableData[0][0];
            relPerm->stone_model_data[1] = 1.-relPerm->Krw_TableData[0][relPerm->numwaterdatarow-1];
            relPerm->stone_model_data[2] = relPerm->Krow_TableData[1][0];
            for(i = 0; i < relPerm->numwaterdatarow; i++){
                if(relPerm->Krow_TableData[1][i] == 0.){
                    relPerm->stone_model_data[1] = 1.-relPerm->Krow_TableData[0][i];
                    break;
                }
            }
        }
        else{
            relPerm->FracDUpDateKro = FracDStone1Model;
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateMesh"
extern PetscErrorCode FracDCreateMesh(AppCtx *bag)
{
    PetscInt            cells[3] = {1, 1, 1};
    PetscInt            i,n;
    PetscErrorCode      ierr;
    size_t              len;
    DM                  dmRefined = NULL,dmDist = NULL,coordDM = NULL;
    PetscBool           flg,flg1;
    IS                  faceRegionIS;
    const PetscInt      *faceRegions;
    PetscInt            rank;
    
    PetscFunctionBeginUser;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    PetscStrlen(bag->meshfilename, &len);
    if (!len) {
        if(bag->simplexmesh)
        {
            //            ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, bag->dim, bag->meshinterpolate, &bag->plexScalNode);CHKERRQ(ierr);
            ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, bag->dim, PETSC_TRUE, NULL, NULL, NULL, NULL, bag->meshinterpolate, &bag->plexScalNode);CHKERRQ(ierr);
        }
        else{
            n = 3;
            ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Create Meshing Problem Options", "");CHKERRQ(ierr);
            {
                ierr = PetscOptionsIntArray("-grid_size","\n\tnumber of grid points (default 1), comma separated","",cells,&n,&flg);CHKERRQ(ierr);
            }
            ierr = PetscOptionsEnd();CHKERRQ(ierr);
            if (flg) {
                if (n != bag->dim) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Number of -grid_size options should be %i, got %i in %s",bag->dim, n,__FUNCT__);
            }
            //            ierr = DMPlexCreateHexBoxMesh(PETSC_COMM_WORLD, bag->dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &bag->plexScalNode);CHKERRQ(ierr);
        }
    }
    else{
        ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, bag->meshfilename, bag->meshinterpolate, &bag->plexScalNode);CHKERRQ(ierr);
    }
    if(bag->meshrefine){
        bag->meshrefinetype = UNIFORM;
        ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\n FracD: Option for Type of Mesh Refinement:","");CHKERRQ(ierr);
        ierr = PetscOptionsEnum("-meshrefinetype","\n\t\n\t mesh refinement type","",FracDMeshrefine_name,(PetscEnum)bag->meshrefinetype,(PetscEnum*)&bag->meshrefinetype,&flg1);CHKERRQ(ierr);
        ierr = PetscOptionsEnd();CHKERRQ(ierr);
        switch (bag->meshrefinetype) {
            case 0:
            ierr = DMPlexSetRefinementUniform(bag->plexScalNode, bag->meshrefine);CHKERRQ(ierr);
            break;
            case 1:
            bag->refinementLimit = 0;
            ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\nFracD: Option to set maximum volume constraint for mesh refinement:","");CHKERRQ(ierr);
            ierr = PetscOptionsReal("-maxmeshsize","\n\t maximum volume constraint for mesh refinement:","",bag->refinementLimit,&bag->refinementLimit,PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsEnd();CHKERRQ(ierr);
            ierr = DMPlexSetRefinementLimit(bag->plexScalNode, bag->refinementLimit);CHKERRQ(ierr);
            break;
        }
        ierr = DMRefine(bag->plexScalNode, PETSC_COMM_WORLD, &dmRefined);CHKERRQ(ierr);
        if (dmRefined) {
            ierr = DMDestroy(&bag->plexScalNode);CHKERRQ(ierr);
            bag->plexScalNode = dmRefined;
        }
    }
    if(!rank){
        ierr = DMGetLabelSize(bag->plexScalNode, "Face Sets", &bag->TotalFaceSets);CHKERRQ(ierr);
        ierr = DMGetLabelIdIS(bag->plexScalNode, "Face Sets", &faceRegionIS);CHKERRQ(ierr);
        ierr = ISGetIndices(faceRegionIS, &faceRegions);CHKERRQ(ierr);
    }
    MPI_Bcast(&bag->TotalFaceSets,1,MPIU_INT,0,PETSC_COMM_WORLD);
    ierr = PetscMalloc(bag->TotalFaceSets * sizeof(PetscInt),&bag->FaceSetIds);CHKERRQ(ierr);
    if(!rank){
        for(i = 0; i < bag->TotalFaceSets; i++){
            bag->FaceSetIds[i] = faceRegions[i];
        }
        ierr = ISRestoreIndices(faceRegionIS, &faceRegions);CHKERRQ(ierr);
        ierr = ISDestroy(&faceRegionIS);CHKERRQ(ierr);
    }
    MPI_Bcast(bag->FaceSetIds,bag->TotalFaceSets,MPIU_INT,0,PETSC_COMM_WORLD);
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(bag->plexScalNode, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist)
    {
        ierr = DMDestroy(&bag->plexScalNode);CHKERRQ(ierr);
        bag->plexScalNode = dmDist;
    }
    ierr = DMSetFromOptions(bag->plexScalNode);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) bag->plexScalNode, "Mesh");CHKERRQ(ierr);
    
    ierr = DMClone(bag->plexScalNode, &bag->plexVecNode);CHKERRQ(ierr);
    ierr = DMClone(bag->plexScalNode, &bag->plexScalCell);CHKERRQ(ierr);
    ierr = DMClone(bag->plexScalNode, &bag->plexVecCell);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexVecNode, coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexScalCell, coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexVecCell, coordDM);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateFEShapeFunction"
extern PetscErrorCode FracDCreateFEShapeFunction(AppCtx *bag)
{
    PetscErrorCode     ierr;
    
    PetscFunctionBegin;
    ierr = FracDCVFEFaceCreate(bag->dim, &bag->CVFEface,bag->elementType);CHKERRQ(ierr);
    ierr = FracDPointFEElementCreate(bag->dim, &bag->epD, bag->elementType);CHKERRQ(ierr);
    ierr = FracDFEElementCreate(bag->dim, &bag->eD, bag->elementType);CHKERRQ(ierr);
    ierr = FracDFEElementCreate(bag->dim-1, &bag->elD, bag->elementType);CHKERRQ(ierr);
    switch (bag->dim) {
        case 2:
        bag->FracDIsWellInElement = FracDFindPointIn2DElement;
        bag->FracDCreateDMinusOneFEElement = FracD1DElementFE;
        bag->FracDProjectFaceCoordinateDimensions = FracD1DProjectFaceCoordinateDimensions;
        if  (bag->elementType == TRIANGLE){
            bag->nodes = 3;
            bag->FracDCreateCVFEFace = FracD2DTriangleCVFEFace;
            bag->FracDCreateDPointFEElement = FracD2DTrianglePointElementFE;
            bag->FracDCreateDFEElement = FracD2DTriangleElementFE;
        }
        if  (bag->elementType == QUADRILATERAL){
            bag->nodes = 4;
            bag->FracDCreateCVFEFace = FracD2DQuadrilateralCVFEFace;
            bag->FracDCreateDPointFEElement = FracD2DQuadrilateralPointElementFE;
            bag->FracDCreateDFEElement = FracD2DQuadrilateralElementFE;
        }
        break;
        case 3:
        if  (bag->elementType == TETRAHEDRAL){
            bag->nodes = 4;
            bag->FracDIsWellInElement = FracDFindPointIn3DTetrahedral;
            bag->FracDCreateCVFEFace = FracD3DTetrahedralCVFEFace;
            bag->FracDCreateDPointFEElement = FracD3DTetrahedralPointElementFE;
            bag->FracDCreateDFEElement = FracD3DTetrahedralElementFE;
            bag->FracDCreateDMinusOneFEElement = FracD2DTriangleElementFE;
            bag->FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensions;
        }
        if  (bag->elementType == HEXAHEDRAL){
            bag->nodes = 8;
            bag->FracDIsWellInElement = FracDFindPointIn3DHexahedral;
            bag->FracDCreateCVFEFace = FracD3DHexahedralCVFEFace;
            bag->FracDCreateDPointFEElement = FracD3DHexahedralPointElementFE;
            bag->FracDCreateDFEElement = FracD3DHexahedralElementFE;
            bag->FracDCreateDMinusOneFEElement = FracD2DQuadrilateralElementFE;
            bag->FracDProjectFaceCoordinateDimensions = FracD2DProjectFaceCoordinateDimensionsHexahedral;
        }
        break;
        default:
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i in %s\n",bag->dim,__FUNCT__);
        break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDSetMechanisMatrixType"
extern PetscErrorCode FracDSetMechanisMatrixType(AppCtx *bag)
{
    PetscFunctionBegin;
    switch (bag->dim) {
        case 2:
        if (bag->elasticity2DType == PLANESTRESS)   bag->FracDElasticityStiffnessMatrixLocal = FracDElasticity2DPlaneStress_local;
        if (bag->elasticity2DType == PLANESTRAIN)   bag->FracDElasticityStiffnessMatrixLocal = FracDElasticity2DPlaneStrain_local;
        break;
        case 3:
        bag->FracDElasticityStiffnessMatrixLocal = FracDElasticity3D_local;
        break;
        default:
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Dimension should be 2 or 3, got %i in %s\n",bag->dim,__FUNCT__);
        break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateBCLabels"
extern PetscErrorCode FracDCreateBCLabels(AppCtx *bag, const char prefix[], FracDBC *BC)
{
    PetscErrorCode ierr;
    PetscInt            i,j,p,nopt,comps = 0;
    PetscInt            n,numfaceRegions,numPoints;
    PetscInt            *faceRegions;
    PetscInt            rank,size;
    PetscBool           hasindex=PETSC_FALSE;
    IS                  pointIS;
    const PetscInt      *points;
    PetscBool           flg;
    
    PetscFunctionBegin;
    
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
    numfaceRegions = bag->TotalFaceSets;
    faceRegions = bag->FaceSetIds;
    ierr =  DMCreateLabel(bag->plexScalNode, BC->labelName);CHKERRQ(ierr);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,prefix,"\n\n FracDBC Create Options: Creating BC labels:","");CHKERRQ(ierr);
    {
        BC->numRegions  = 1;
        ierr  = PetscOptionsInt("-numlabels","\n\t Number of BC regions/labels","",BC->numRegions,&BC->numRegions,PETSC_NULL);CHKERRQ(ierr);
        if (BC->numRegions > numfaceRegions)   SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Obtained %i BC labels for %s which is greater than %i, the total BC labels in mesh. See %s for %s  BC\n",BC->numRegions,BC->labelName,numfaceRegions,__FUNCT__);
        nopt = BC->numRegions;
        ierr            = PetscMalloc(nopt * sizeof(PetscInt),&BC->regions);CHKERRQ(ierr);
        ierr            = PetscMalloc(nopt * sizeof(PetscInt),&BC->numcompsperlabel);CHKERRQ(ierr);
        for(i = 0; i < nopt; i++)   {
            BC->regions[i] = -999999;
            BC->numcompsperlabel[i] = 1;
        }
        n = nopt;
        ierr = PetscOptionsIntArray("-labels", "\n\t BC labels ","",BC->regions,&n,PETSC_NULL);CHKERRQ(ierr);
        if (n != nopt)  SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i %s BC labels, got %i in %s\n",nopt,BC->labelName,n,__FUNCT__);
        for(i = 0; i < BC->numRegions; i++){
            for(j = 0; j < numfaceRegions; j++){
                if(BC->regions[i] == faceRegions[j])   hasindex=PETSC_TRUE;
            }
            if (!hasindex) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Inputted %s BC label %i in %s is not a mesh label\n",BC->labelName,BC->regions[i],__FUNCT__);
            ierr = DMGetStratumIS(bag->plexScalNode, "Face Sets", BC->regions[i], &pointIS);CHKERRQ(ierr);
            if(pointIS){
                ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
                ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
                for(p = 0; p < numPoints; ++p){
                    ierr = DMSetLabelValue(bag->plexScalNode, BC->labelName, points[p], BC->regions[i]);CHKERRQ(ierr);
                }
                ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
                ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
            }
        }
        flg = PETSC_FALSE;
        n = nopt;
        ierr = PetscOptionsIntArray("-componentsperlabel", "\n\t Number of BC components per label ","",BC->numcompsperlabel,&n,&flg);CHKERRQ(ierr);
        if (flg && n != nopt)  SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i number of %s BC components for each label, got %i in %s\n",nopt,BC->labelName,n,__FUNCT__);
        for (i = 0; i < nopt; i++){
            if (BC->numcompsperlabel[i] > bag->dim)
            SETERRQ5(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting at most %i %s BC components for i%th label, got %i in %s\n",bag->dim,BC->labelName,nopt,BC->numcompsperlabel[i],__FUNCT__);
            else
            comps += BC->numcompsperlabel[i];
        }
        ierr            = PetscMalloc(comps * sizeof(PetscInt),&BC->components);CHKERRQ(ierr);
        ierr            = PetscMalloc(comps * sizeof(PetscReal),&BC->values);CHKERRQ(ierr);
        for(i = 0; i < comps; i++)   {
            BC->components[i] = 0;
            BC->values[i] = 0;
        }
        flg = PETSC_FALSE;
        n = comps;
        ierr = PetscOptionsIntArray("-components", "\n\t BC components per label ","",BC->components,&n,&flg);CHKERRQ(ierr);
        if (flg && n != comps) SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting a total of %i %s components (i.e total number of components must equal sum of all components per label), got %i in %s\n",comps,BC->labelName,n,__FUNCT__);
        
        flg = PETSC_FALSE;
        n = comps;
        ierr = PetscOptionsRealArray("-values", "\n\n BC values at Boundaries ","",BC->values,&n,&flg);CHKERRQ(ierr);
        if (flg && n != comps)  SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of %s BC values, got %i in %s\n",comps,BC->labelName,n,__FUNCT__);
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitializeBoundaryConditions"
extern PetscErrorCode FracDInitializeBoundaryConditions(AppCtx *bag )
{
    PetscErrorCode      ierr;
    DM                  coordDM = NULL;
    char                prefix[PETSC_MAX_PATH_LEN+1];
    
    PetscFunctionBegin;
    if(bag->UBC.hasLabel){
        strcpy(prefix,"UBC_");
        strcpy (bag->UBC.labelName,"Displacement");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->UBC);CHKERRQ(ierr);
    }
    if(bag->PBC.hasLabel){
        strcpy(prefix,"PBC_");
        strcpy (bag->PBC.labelName,"Pressure");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->PBC);CHKERRQ(ierr);
    }
    if(bag->TBC.hasLabel){
        strcpy(prefix,"TBC_");
        strcpy (bag->TBC.labelName,"Temperature");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->TBC);CHKERRQ(ierr);
    }
    if(bag->TractionBC.hasLabel){
        strcpy(prefix,"TractionBC_");
        strcpy (bag->TractionBC.labelName,"Traction");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->TractionBC);CHKERRQ(ierr);
    }
    if(bag->FlowFluxBC.hasLabel){
        strcpy(prefix,"FlowFluxBC_");
        strcpy (bag->FlowFluxBC.labelName,"FlowFlux");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->FlowFluxBC);CHKERRQ(ierr);
    }
    if(bag->HeatFluxBC.hasLabel){
        strcpy(prefix,"HeatFluxBC_");
        strcpy (bag->HeatFluxBC.labelName,"FlowFlux");
        ierr = FracDCreateBCLabels(bag,prefix,&bag->HeatFluxBC);CHKERRQ(ierr);
    }
    
    ierr = DMDestroy(&bag->plexVecNode);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexScalCell);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexVecCell);CHKERRQ(ierr);
    
    ierr = DMClone(bag->plexScalNode, &bag->plexVecNode);CHKERRQ(ierr);
    ierr = DMClone(bag->plexScalNode, &bag->plexScalCell);CHKERRQ(ierr);
    ierr = DMClone(bag->plexScalNode, &bag->plexVecCell);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexScalNode, &coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexVecNode, coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexScalCell, coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(bag->plexVecCell, coordDM);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitializeUnitConversions"
extern PetscErrorCode FracDInitializeUnitConversions(AppCtx *bag)
{
    PetscFunctionBegin;
    bag->ConversionMMGasRate = 1.;
    if(bag->Units == FIELDUNITS){
        bag->ConversionFactorBeta = 0.001127;
        bag->ConversionFactorGamma = 0.21584e-3;
        bag->ConversionFactorAlpha = 5.614583;
        bag->ConversionMMGasRate = 1e+6;
        //        See 11363_02.pdf, page 21 (pdf page 15 of 35)
    }
    if(bag->Units == METRICUNITS){
        bag->ConversionFactorBeta = 1.;
        bag->ConversionFactorGamma = 1.;
        bag->ConversionFactorAlpha = 1.;
    }
    bag->ppties.WaterPVTData.RateConversion = bag->ppties.OilPVTData.RateConversion = 1.;
    bag->ppties.GasPVTData.RateConversion = bag->ppties.SolutionGasOilData.RateConversion = bag->ConversionMMGasRate;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitializeWells"
extern PetscErrorCode FracDInitializeWells(AppCtx *bag)
{
    PetscErrorCode      ierr;
    PetscInt            i;
    char                prefix[PETSC_MAX_PATH_LEN];
    
    PetscFunctionBegin;
    bag->numWells = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Creating Wells", "");CHKERRQ(ierr);
    {
        ierr          = PetscOptionsInt("-nw","\n\t Number of wells to insert","",bag->numWells,&bag->numWells,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    
    ierr          = PetscMalloc(bag->numWells*sizeof(FracDWell),&bag->well);CHKERRQ(ierr);
    for (i = 0; i < bag->numWells; i++) {
        ierr = PetscSNPrintf(prefix,sizeof(prefix),"w%d_",i);CHKERRQ(ierr);
        ierr = FracDWellCreate(&(bag->well[i]),bag->dim);CHKERRQ(ierr);
        ierr = FracDGetWell(prefix, &(bag->well[i]));CHKERRQ(ierr);
        if (bag->verbose) {
            ierr = FracDWellView(&(bag->well[i]),PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFindPointIn3DHexahedral"
extern PetscErrorCode FracDFindPointIn3DHexahedral(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg)
{
    PetscInt            i,j,k,plane[6][3];
    PetscReal           a,b,c,d,xp,yp,zp,x[3],y[3],z[3],centroid[3];
    PetscReal           value,value_c;
    PetscInt            scale_local[6], prod = 1.;
    
    //    https://stackoverflow.com/questions/16446147/tetrahedron-height-vertices
    
    PetscFunctionBegin;
    *flg = PETSC_FALSE;
    for(j = 0; j < 6; j++)  scale_local[j] = 0;
    for(i = 0; i < 3; i++){
        centroid[i] = 0.;
        for(j = 0; j < num_nodes; j++)  centroid[i] += coords[j][i];
        centroid[i] = centroid[i]/num_nodes;
    }
    xp = point[0];    yp = point[1];    zp = point[2];
    plane[0][0] = 0;plane[0][1] = 1;plane[0][2] = 2;
    plane[1][0] = 4;plane[1][1] = 5;plane[1][2] = 6;
    plane[2][0] = 0;plane[2][1] = 1;plane[2][2] = 7;
    plane[3][0] = 3;plane[3][1] = 2;plane[3][2] = 6;
    plane[4][0] = 0;plane[4][1] = 3;plane[4][2] = 5;
    plane[5][0] = 1;plane[5][1] = 2;plane[5][2] = 6;

    for(j = 0; j < 6; j++){
        scale_local[j] = 0.;
        for(i = 0; i < 3; i++){
            k = plane[j][i];
            x[i] = coords[k][0];
            y[i] = coords[k][1];
            z[i] = coords[k][2];
        }
        a = y[0] * (z[1]-z[2]) + y[1] * (z[2]-z[0]) + y[2] * (z[0]-z[1]);
        b = z[0] * (x[1]-x[2]) + z[1] * (x[2]-x[0]) + z[2] * (x[0]-x[1]);
        c = x[0] * (y[1]-y[2]) + x[1] * (y[2]-y[0]) + x[2] * (y[0]-y[1]);
        d = x[0] * (y[1]*z[2]-y[2]*z[1]) + x[1] * (y[2]*z[0]-y[0]*z[2]) + x[2] * (y[0]*z[1]-y[1]*z[0]);
        value = a*xp+b*yp+c*zp-d;
        value_c = a*centroid[0]+b*centroid[1]+c*centroid[2]-d;
        if((value_c < 0 && value <= 0) ||  (value_c > 0 && value >= 0)) scale_local[j] = 1.;
    }
    for(j = 0; j < 6; j++)  prod = prod*scale_local[j];
    if(prod)   *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFindPointIn3DTetrahedral"
extern PetscErrorCode FracDFindPointIn3DTetrahedral(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg)
{
    PetscInt            i,j,k,plane[4][3];
    PetscReal           a,b,c,d,xp,yp,zp,x[3],y[3],z[3],centroid[3];
    PetscReal           value,value_c;
    PetscInt            scale_local[4], prod = 1.;
    
    //    https://stackoverflow.com/questions/16446147/tetrahedron-height-vertices
    
    PetscFunctionBegin;
    *flg = PETSC_FALSE;
    for(j = 0; j < 4; j++)  scale_local[j] = 0;
    for(i = 0; i < 3; i++){
        centroid[i] = 0.;
        for(j = 0; j < num_nodes; j++)  centroid[i] += coords[j][i];
        centroid[i] = centroid[i]/num_nodes;
    }
    xp = point[0];    yp = point[1];    zp = point[2];
    plane[0][0] = 0;plane[0][1] = 1;plane[0][2] = 2;
    plane[1][0] = 0;plane[1][1] = 1;plane[1][2] = 3;
    plane[2][0] = 0;plane[2][1] = 2;plane[2][2] = 3;
    plane[3][0] = 1;plane[3][1] = 2;plane[3][2] = 3;
    for(j = 0; j < 4; j++){
        scale_local[j] = 0.;
        for(i = 0; i < 3; i++){
            k = plane[j][i];
            x[i] = coords[k][0];
            y[i] = coords[k][1];
            z[i] = coords[k][2];

        }
        a = y[0] * (z[1]-z[2]) + y[1] * (z[2]-z[0]) + y[2] * (z[0]-z[1]);
        b = z[0] * (x[1]-x[2]) + z[1] * (x[2]-x[0]) + z[2] * (x[0]-x[1]);
        c = x[0] * (y[1]-y[2]) + x[1] * (y[2]-y[0]) + x[2] * (y[0]-y[1]);
        d = x[0] * (y[1]*z[2]-y[2]*z[1]) + x[1] * (y[2]*z[0]-y[0]*z[2]) + x[2] * (y[0]*z[1]-y[1]*z[0]);
        value = a*xp+b*yp+c*zp-d;
        value_c = a*centroid[0]+b*centroid[1]+c*centroid[2]-d;
        if((value_c < 0 && value <= 0) ||  (value_c > 0 && value >= 0)) scale_local[j] = 1.;
    }
    for(j = 0; j < 4; j++)  prod = prod*scale_local[j];
    if(prod)   *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFindPointIn2DElement"
extern PetscErrorCode FracDFindPointIn2DElement(PetscInt num_nodes, PetscReal *point, PetscReal **coords, PetscBool *flg)
{
    PetscInt            i,j,c = 0;
    PetscReal           x,y,xp[num_nodes],yp[num_nodes];
    
    PetscFunctionBegin;
    *flg = PETSC_FALSE;
    x = point[0];
    y = point[1];
    for(i = 0; i < num_nodes; i++){
        xp[i] = coords[i][0];
        yp[i] = coords[i][1];
    }
    for (i = 0, j = num_nodes-1; i < num_nodes; j = i++) {
        if ((((yp[i] <= y) && (y < yp[j])) ||
             ((yp[j] <= y) && (y < yp[i]))) &&
            (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
        c = !c;
    }
    if(c)   *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDProcessWellBlockLocation"
extern PetscErrorCode FracDProcessWellBlockLocation(AppCtx *bag)
{
    PetscErrorCode      ierr;
    DM                  cdm;
    Vec                 coordinates;
    PetscScalar         *coord_array = NULL;
    PetscReal           **coords;
    PetscInt            cordsize, c, cStart, cEnd;
    PetscInt            i,j,l, w, *wellcount, tmp[bag->numWells];
    PetscSection        cordSection;
    PetscBool           flg;
    PetscInt            rank;

    PetscFunctionBegin;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    bag->WellinMeshData.numberWellsInProcessor=0;
    for(w = 0; w < bag->numWells; w++)  tmp[w] = 0;
    ierr = DMPlexGetHeightStratum(bag->plexVecCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(bag->plexVecCell,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexVecCell, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    coords = (PetscReal **)malloc(bag->eD.nodes * sizeof(PetscReal *));
    for(i = 0; i < bag->eD.nodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->eD.dim * sizeof(PetscReal));
    }
    for(w = 0; w < bag->numWells; w++){
            for(c = cStart; c < cEnd; ++c){
            ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, &cordsize, &coord_array);CHKERRQ(ierr);
            for(l = 0, i = 0; i < bag->eD.nodes; i++){
                for(j = 0; j < bag->eD.dim; j++, l++){
                    coords[i][j] = coord_array[l];
                }
            }
            ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, &cordsize, &coord_array);CHKERRQ(ierr);
            flg = PETSC_FALSE;
            ierr = bag->FracDIsWellInElement(bag->eD.nodes,bag->well[w].coordinates, coords, &flg);CHKERRQ(ierr);
            if(flg){
                bag->WellinMeshData.numberWellsInProcessor++;
                if(c == 0)  tmp[w] = -1;
                else        tmp[w] = c;
                break;
            }
        }
    }
    bag->WellinMeshData.Count = (PetscInt *)malloc(bag->numWells * sizeof(PetscInt));
    wellcount = (PetscInt *)malloc(bag->numWells * sizeof(PetscInt));
    for(w = 0; w < bag->numWells; w++)  wellcount[w] = 0;
    
    bag->WellinMeshData.WellInfo = (PetscInt **)malloc(bag->WellinMeshData.numberWellsInProcessor * sizeof(PetscInt *));
    for(w = 0; w < bag->WellinMeshData.numberWellsInProcessor; w++)
    {
        bag->WellinMeshData.WellInfo[w] = (PetscInt *)malloc(2 * sizeof(PetscInt));
    }
    for(w = 0; w < bag->WellinMeshData.numberWellsInProcessor; w++) bag->WellinMeshData.WellInfo[w][0] = bag->WellinMeshData.WellInfo[w][1] =-99;
/*
    1st is Well number
    2nd is well cell number
    bag->WellinMeshData.Count provides info about how many time well w appears in the processors. For example, location of well w may be at the boundary of 2 processors.
    This well will have a count of 2. Therefore, in the processing of the jacobian and residual, this well's contribution will have to be divided by 2.
*/
    c = 0;
    for(w = 0; w < bag->numWells; w++){
        if(tmp[w] != 0){
            wellcount[w] = 1;
            bag->WellinMeshData.WellInfo[c][0] = w;
            if(tmp[w] == -1)    bag->WellinMeshData.WellInfo[c][1] = 0;
            else    bag->WellinMeshData.WellInfo[c][1] = tmp[w];
            c++;
        }
    }
    ierr = MPI_Allreduce(wellcount,bag->WellinMeshData.Count,bag->numWells,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
    for(i = 0; i < bag->eD.nodes; i++){
        free(coords[i]);
    }
    free(coords);
    free(wellcount);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateFlowMatix"
extern PetscErrorCode FracDCreateFlowMatix(AppCtx *bag, Mat *K, Mat *KPC, Vec X)
{
    PetscErrorCode ierr;
    MatType        Kpp_type;
    IS             *is;
    PetscInt       i,j,*m,*M,Msize,nnz;
    Mat            *bK;
    DM             *dm;
    Vec            p=NULL,sw=NULL,sg=NULL,pbh=NULL;
    
    PetscFunctionBegin;
    nnz = 2;
    switch (bag->fluid) {
        case 0:
        Msize = 2;
        break;
        case 1:
        Msize = 2;
        break;
        case 2:
        Msize = 3;
        break;
        case 3:
        Msize = 4;
        break;
    }
    if(bag->numWells == 0){
        Msize--;
        nnz = 0;
    }
    bag->BlockMatrixSize = Msize;
    m = (PetscInt *)malloc(Msize * sizeof(PetscInt));
    M = (PetscInt *)malloc(Msize * sizeof(PetscInt));
    bK = (Mat *)malloc(Msize * Msize * sizeof(Mat));
    ierr = DMCompositeGetGlobalISs(bag->MultiPhasePacker,&is);CHKERRQ(ierr);
    switch (bag->fluid) {
        case 0:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,X,&p,&pbh);CHKERRQ(ierr);
            ierr = VecGetSize(p,&M[0]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(p,&m[0]);CHKERRQ(ierr);
            ierr = VecGetSize(pbh,&M[1]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(pbh,&m[1]);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,X,&p,&pbh);CHKERRQ(ierr);
        }
        break;
        case 1:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,X,&p,&pbh);CHKERRQ(ierr);
            ierr = VecGetSize(p,&M[0]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(p,&m[0]);CHKERRQ(ierr);
            ierr = VecGetSize(pbh,&M[1]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(pbh,&m[1]);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,X,&p,&pbh);CHKERRQ(ierr);
        }
        break;
        case 2:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,X,&p,&sw,&pbh);CHKERRQ(ierr);
            ierr = VecGetSize(p,&M[0]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(p,&m[0]);CHKERRQ(ierr);
            ierr = VecGetSize(sw,&M[1]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(sw,&m[1]);CHKERRQ(ierr);
            ierr = VecGetSize(pbh,&M[2]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(pbh,&m[2]);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,X,&p,&sw,&pbh);CHKERRQ(ierr);
        }
        break;
        case 3:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,X,&p,&sw,&sg,&pbh);CHKERRQ(ierr);
            ierr = VecGetSize(p,&M[0]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(p,&m[0]);CHKERRQ(ierr);
            ierr = VecGetSize(sw,&M[1]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(sw,&m[1]);CHKERRQ(ierr);
            ierr = VecGetSize(sg,&M[2]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(sg,&m[2]);CHKERRQ(ierr);
            ierr = VecGetSize(pbh,&M[3]);CHKERRQ(ierr);
            ierr = VecGetLocalSize(pbh,&m[3]);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,X,&p,&sw,&sg,&pbh);CHKERRQ(ierr);
        }
        break;
    }
    for(i = 0; i < Msize; i++){
        for(j = 0; j < Msize; j++){
            if(i == j){
                if(bag->numWells != 0 && i == Msize-1) dm = &bag->WellRedun;
                else  dm = &bag->plexScalNode;
                ierr = DMCreateMatrix(*dm,&bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatSetBlockSize(bK[j+i*Msize],1);CHKERRQ(ierr);
                ierr = MatSetOption(bK[j+i*Msize],MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
                ierr = MatSetUp(bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatZeroEntries(bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatAssemblyBegin(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                ierr = MatAssemblyEnd(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            }
            else if(i == Msize-1 || j == Msize-1){
                if(bag->numWells != 0){
                    ierr = MatGetType(bK[0],&Kpp_type);CHKERRQ(ierr);
                    ierr = MatCreate(PETSC_COMM_WORLD,&bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatSetSizes(bK[j+i*Msize],m[i],m[j],M[i],M[j]);CHKERRQ(ierr);
                    ierr = MatSetBlockSize(bK[j+i*Msize],1);CHKERRQ(ierr);
                    ierr = MatSetType(bK[j+i*Msize],Kpp_type);CHKERRQ(ierr);
                    ierr = MatMPIAIJSetPreallocation(bK[j+i*Msize],nnz,PETSC_NULL,nnz,PETSC_NULL);CHKERRQ(ierr);
                    //                  ierr = MatSetOption(bK[j+i*Msize],MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
                    ierr = MatSetOption(bK[j+i*Msize],MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
                    ierr = MatSetUp(bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatZeroEntries(bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatAssemblyBegin(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                    ierr = MatAssemblyEnd(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                }
                else{
                    dm = &bag->plexScalNode;
                    ierr = DMCreateMatrix(*dm,&bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatSetBlockSize(bK[j+i*Msize],1);CHKERRQ(ierr);
                    ierr = MatSetOption(bK[j+i*Msize],MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
                    ierr = MatSetUp(bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatZeroEntries(bK[j+i*Msize]);CHKERRQ(ierr);
                    ierr = MatAssemblyBegin(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                    ierr = MatAssemblyEnd(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                }
            }
            else{
                dm = &bag->plexScalNode;
                ierr = DMCreateMatrix(*dm,&bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatSetBlockSize(bK[j+i*Msize],1);CHKERRQ(ierr);
                ierr = MatSetOption(bK[j+i*Msize],MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
                ierr = MatSetUp(bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatZeroEntries(bK[j+i*Msize]);CHKERRQ(ierr);
                ierr = MatAssemblyBegin(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                ierr = MatAssemblyEnd(bK[j+i*Msize],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            }
        }
    }
    ierr = MatCreateNest(PETSC_COMM_WORLD,Msize,is,Msize,is,&bK[0],K);CHKERRQ(ierr);
    ierr = MatSetUp(*K);CHKERRQ(ierr);
    ierr = MatCreateNest(PETSC_COMM_WORLD,Msize,is,Msize,is,&bK[0],KPC);CHKERRQ(ierr);
    ierr = MatSetUp(*KPC);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*K,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*KPC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if((bag->fluid == SINGLEPHASELIQUID || bag->fluid == SINGLEPHASEGAS) && bag->numWells == 0){
        ierr = MatDestroy(K);CHKERRQ(ierr);
        ierr = MatDestroy(KPC);CHKERRQ(ierr);
        ierr = DMCreateMatrix(bag->plexScalNode,K);CHKERRQ(ierr);
        ierr = MatSetOption(*K,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
        ierr = DMCreateMatrix(bag->plexScalNode,KPC);CHKERRQ(ierr);
        ierr = MatSetOption(*KPC,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    }
    //    ierr = MatZeroEntries(*K);CHKERRQ(ierr);
    //    ierr = MatZeroEntries(*KPC);CHKERRQ(ierr);
    for (i = 0; i < Msize; i++) ierr = ISDestroy(&is[i]);CHKERRQ(ierr);
    ierr = PetscFree(is);CHKERRQ(ierr);
    for (i = 0; i < Msize*Msize; i++) ierr = MatDestroy(&bK[i]);CHKERRQ(ierr);
    free(bK);
    free(m);
    free(M);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitializeSolvers"
extern PetscErrorCode FracDInitializeSolvers(AppCtx *bag)
{
    PetscErrorCode ierr;
    KSP            kspU,kspP,kspT;
    PC             pcU,pcP,pcT;
    Mat            JacPCU,JacPCP,JacPCT;
    Mat            JacU,JacP,JacT;
    Vec            residualU,residualP,residualT;
    IS             *is;
    PetscInt       i,Msize;
    
    PetscFunctionBegin;
    
    ierr = SNESCreate(PETSC_COMM_WORLD, &bag->snesU);CHKERRQ(ierr);
    ierr = SNESSetDM(bag->snesU, bag->plexVecNode);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(bag->snesU,"U_");CHKERRQ(ierr);
    
    ierr = SNESSetType(bag->snesU,SNESNEWTONLS);CHKERRQ(ierr);
    ierr = SNESSetTolerances(bag->snesU,1.e-8,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(bag->snesU);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode,&residualU);CHKERRQ(ierr);
    ierr = DMCreateMatrix(bag->plexVecNode,&JacU);CHKERRQ(ierr);
    ierr = MatSetOption(JacU,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCreateMatrix(bag->plexVecNode,&JacPCU);CHKERRQ(ierr);
    ierr = MatSetOption(JacPCU,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    ierr = SNESSetFunction(bag->snesU,residualU,FracDUResidual,bag);CHKERRQ(ierr);
    ierr = SNESSetJacobian(bag->snesU,JacU,JacPCU,FracDUJacobian,bag);CHKERRQ(ierr);
    
    ierr = SNESGetKSP(bag->snesU,&kspU);CHKERRQ(ierr);
    ierr = KSPSetTolerances(kspU,1.e-8,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetType(kspU,KSPCG);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(kspU);CHKERRQ(ierr);
    ierr = KSPGetPC(kspU,&pcU);CHKERRQ(ierr);
    ierr = PCSetType(pcU,PCHYPRE);CHKERRQ(ierr);
    //    ierr = PetscOptionsInsertString("-u_pc_hypre_boomeramg_strong_threshold 0.7 -u_pc_hypre_type boomeramg");CHKERRQ(ierr);
    ierr = PCSetFromOptions(pcU);CHKERRQ(ierr);
    
    ierr = SNESCreate(PETSC_COMM_WORLD, &bag->snesT);CHKERRQ(ierr);
    ierr = SNESSetDM(bag->snesT, bag->plexScalNode);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(bag->snesT,"T_");CHKERRQ(ierr);
    
    ierr = SNESSetType(bag->snesT,  SNESNEWTONLS);CHKERRQ(ierr);
    ierr = SNESSetTolerances(bag->snesT,1.e-8,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(bag->snesT);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode,&residualT);CHKERRQ(ierr);
    ierr = DMCreateMatrix(bag->plexScalNode,&JacT);CHKERRQ(ierr);
    ierr = MatSetOption(JacT,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMCreateMatrix(bag->plexScalNode,&JacPCT);CHKERRQ(ierr);
    ierr = MatSetOption(JacPCT,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
    ierr = SNESSetFunction(bag->snesT,residualT,FracDTResidual,bag);CHKERRQ(ierr);
    ierr = SNESSetJacobian(bag->snesT,JacT,JacPCT,FracDTJacobian,bag);CHKERRQ(ierr);
    
    ierr = SNESGetKSP(bag->snesT,&kspT);CHKERRQ(ierr);
    ierr = KSPSetTolerances(kspT,1.e-8,1.e-10,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetType(kspT,KSPGMRES);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(kspT);CHKERRQ(ierr);
    ierr = KSPGetPC(kspT,&pcT);CHKERRQ(ierr);
    ierr = PCSetType(pcT,PCHYPRE);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pcT);CHKERRQ(ierr);
    
    
    
    
    
    
    
    
    
    ierr = SNESCreate(PETSC_COMM_WORLD, &bag->snesP);CHKERRQ(ierr);
    ierr = SNESSetDM(bag->snesP, bag->MultiPhasePacker);CHKERRQ(ierr);
    ierr = SNESSetOptionsPrefix(bag->snesP,"P_");CHKERRQ(ierr);
    
    ierr = SNESSetType(bag->snesP,  SNESNEWTONLS);CHKERRQ(ierr);
    ierr = SNESSetTolerances(bag->snesP,1.e-8,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(bag->snesP);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->MultiPhasePacker,&residualP);CHKERRQ(ierr);
    ierr = FracDCreateFlowMatix(bag,&JacP,&JacPCP,residualP);CHKERRQ(ierr);
    ierr = SNESSetFunction(bag->snesP,residualP,FracDPResidual,bag);CHKERRQ(ierr);
    ierr = SNESSetJacobian(bag->snesP,JacP,JacPCP,FracDPJacobian,bag);CHKERRQ(ierr);
    
    ierr = SNESGetKSP(bag->snesP,&kspP);CHKERRQ(ierr);
    ierr = KSPSetTolerances(kspP,1.e-8,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetType(kspP,KSPFGMRES);CHKERRQ(ierr);
    //    ierr = KSPSetType(kspP,KSPFCG);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(kspP);CHKERRQ(ierr);
    ierr = KSPGetPC(kspP,&pcP);CHKERRQ(ierr);
    ierr = PCSetType(pcP, PCFIELDSPLIT);CHKERRQ(ierr);
    ierr = DMCompositeGetGlobalISs(bag->MultiPhasePacker,&is);CHKERRQ(ierr);
    switch (bag->fluid) {
        case 0:
        Msize = 2;
        break;
        case 1:
        Msize = 2;
        break;
        case 2:
        Msize = 3;
        break;
        case 3:
        Msize = 4;
        break;
    }
    if(bag->numWells == 0)  Msize--;
    for(i = 0; i < Msize; i++)  ierr = PCFieldSplitSetIS(pcP,NULL,is[i]);CHKERRQ(ierr);
    ierr = PCFieldSplitSetType(pcP,PC_COMPOSITE_MULTIPLICATIVE);CHKERRQ(ierr);
    for (i = 0; i < Msize; i++)  ierr = ISDestroy(&is[i]);CHKERRQ(ierr);
    ierr = PetscFree(is);CHKERRQ(ierr);
    if((bag->fluid == SINGLEPHASELIQUID || bag->fluid == SINGLEPHASEGAS) && bag->numWells == 0) ierr = PCSetType(pcP,PCHYPRE);CHKERRQ(ierr);
    ierr = PCSetFromOptions(pcP);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDInitialize"
extern PetscErrorCode FracDInitialize(AppCtx *bag)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    //    bag->printhelp = PETSC_FALSE;
    ierr = PetscPrintf(PETSC_COMM_WORLD,banner);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      ##########################################################\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #                                                        #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #                          WARNING!!!                    #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #                                                        #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #   For production runs, use a petsc compiled with       #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #   optimization, the performance will be generally      #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #   two or three times faster.                           #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      #                                                        #\n");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"      ##########################################################\n\n\n");CHKERRQ(ierr);
#endif
    ierr = FracDGetBagOptions(bag);CHKERRQ(ierr);
    
    ierr = FracDCreateMesh(bag);CHKERRQ(ierr);
    ierr = FracDInitializeBoundaryConditions(bag);
    ierr = FracDCreateFEShapeFunction(bag);CHKERRQ(ierr);
    ierr = FracDSetMechanisMatrixType(bag);CHKERRQ(ierr);
    ierr = FracDCreateDataSection(bag);CHKERRQ(ierr);
    ierr = FracDInitializeWells(bag);CHKERRQ(ierr);         /* this function, FracDCreatePackerDM and FracDCreateFields must go hand in hand since the parker is created based on the number of wells. In addition, the associated fields for the packer are created in FracDCreateFields*/
    ierr = FracDProcessWellBlockLocation(bag);CHKERRQ(ierr);
    ierr = FracDCreatePackerDM(bag);CHKERRQ(ierr);
    ierr = FracDCreateFields(bag,&bag->fields);CHKERRQ(ierr);
    ierr = FracDGetResFluidProps(bag,&bag->ppties);CHKERRQ(ierr);
    ierr = FracDInitializeUnitConversions(bag);CHKERRQ(ierr);
    ierr = FracDProcessPVTData(bag);CHKERRQ(ierr);
    ierr = FracDProcessRelPermCapillaryPressureData(bag->fluid,&bag->ppties.RelPermData,&bag->ppties.CapPressData);CHKERRQ(ierr);
    ierr = FracDInitializeSolvers(bag);CHKERRQ(ierr);
    ierr = FracDTimeStepPrepare(bag);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Option table:\n");CHKERRQ(ierr);
    ierr = PetscOptionsView(PETSC_NULL,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFinalize"
extern PetscErrorCode FracDFinalize(AppCtx *bag)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = FracDDestroySolvers(bag);CHKERRQ(ierr);
    ierr = FracDFinalizeRelPermCapillaryPressureData(bag->fluid,&bag->ppties.RelPermData,&bag->ppties.CapPressData);CHKERRQ(ierr);
    ierr = FracDFinalizePVTData(bag);CHKERRQ(ierr);
    ierr = FracDDestroyResFluidProps(bag,&bag->ppties);CHKERRQ(ierr);
    ierr = FracDDestroyFields(bag,&bag->fields);CHKERRQ(ierr);
    ierr = FracDDestroyFEShapeFunction(bag);CHKERRQ(ierr);
    ierr = FracDDestroyBoundaryConditions(bag);
    ierr = FracDDestroyDMMesh(bag);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDestroyFEShapeFunction"
extern PetscErrorCode FracDDestroyFEShapeFunction(AppCtx *bag)
{
    PetscErrorCode     ierr;
    
    PetscFunctionBegin;
    ierr = FracDCVFEFaceDestroy(&bag->CVFEface);CHKERRQ(ierr);
    ierr = FracDPointFEElementDestroy(&bag->epD);CHKERRQ(ierr);
    ierr = FracDFEElementDestroy(&bag->eD);CHKERRQ(ierr);
    ierr = FracDFEElementDestroy(&bag->elD);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDestroyDMMesh"
extern PetscErrorCode FracDDestroyDMMesh(AppCtx *bag)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscFree(bag->FaceSetIds);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexScalNode);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexScalCell);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexVecCell);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->plexVecNode);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->MultiPhasePacker);CHKERRQ(ierr);
    ierr = DMDestroy(&bag->WellRedun);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateDataSection"
extern PetscErrorCode FracDCreateDataSection(AppCtx *bag)
{
    PetscErrorCode ierr;
    PetscSection   nodes,nodes1,cells, cells1;
    PetscInt pStart, pEnd, cStart, cEnd, c, vStart, vEnd, v;
    
    PetscFunctionBegin;
    ierr = DMPlexGetChart(bag->plexScalNode, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexScalCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexScalNode, 0, &vStart, &vEnd);CHKERRQ(ierr);
    
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)bag->plexScalNode), &nodes);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(nodes, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(nodes, pStart, pEnd);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionSetDof(nodes, v, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(nodes, v, 0, 1);CHKERRQ(ierr);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(nodes);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(bag->plexScalNode, nodes);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&nodes);CHKERRQ(ierr);
    
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)bag->plexVecNode), &nodes1);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(nodes1, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(nodes1, pStart, pEnd);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v){
        ierr = PetscSectionSetDof(nodes1, v, bag->dim);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(nodes1, v, 0, bag->dim);CHKERRQ(ierr);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(nodes1);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(bag->plexVecNode, nodes1);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&nodes1);CHKERRQ(ierr);
    
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)bag->plexScalCell), &cells);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(cells, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(cells, pStart, pEnd);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c){
        ierr = PetscSectionSetDof(cells, c, 1);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(cells, c, 0, 1);CHKERRQ(ierr);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(cells);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(bag->plexScalCell, cells);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&cells);CHKERRQ(ierr);
    
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)bag->plexVecCell), &cells1);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(cells1, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(cells1, pStart, pEnd);CHKERRQ(ierr);
    for(c = cStart; c < cEnd; ++c){
        ierr = PetscSectionSetDof(cells1, c, bag->dim);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldDof(cells1, c, 0, bag->dim);CHKERRQ(ierr);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(cells1);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(bag->plexVecCell, cells1);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&cells1);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreatePackerDM"
extern PetscErrorCode FracDCreatePackerDM(AppCtx *bag)
{
    PetscErrorCode      ierr;
    
    PetscFunctionBegin;
    ierr = DMRedundantCreate(PETSC_COMM_WORLD,0,bag->numWells,&bag->WellRedun);CHKERRQ(ierr);
    ierr = DMCompositeCreate(PETSC_COMM_WORLD,&bag->MultiPhasePacker);CHKERRQ(ierr);
    switch (bag->fluid) {
        case 0:
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->WellRedun);CHKERRQ(ierr);
        break;
        case 1:
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->WellRedun);CHKERRQ(ierr);
        break;
        case 2:
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->WellRedun);CHKERRQ(ierr);
        break;
        case 3:
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->plexScalNode);CHKERRQ(ierr);
        ierr = DMCompositeAddDM(bag->MultiPhasePacker,bag->WellRedun);CHKERRQ(ierr);
        break;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDCreateFields"
extern PetscErrorCode FracDCreateFields(AppCtx *bag,FracDFields *fields)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->U);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->U,"Displacement");CHKERRQ(ierr);
    ierr = VecSet(fields->U,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Pb);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pb,"Bubble_point_pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Pb,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPb);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPb,"Previous time bubble_point_pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPb,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Rs);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Rs,"Solution_gas_oil_ratio");CHKERRQ(ierr);
    ierr = VecSet(fields->Rs,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oRs);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oRs,"Old solution_gas_oil_ratio");CHKERRQ(ierr);
    ierr = VecSet(fields->oRs,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->dervRs);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->dervRs,"Derivative of solution_gas_oil_ratio");CHKERRQ(ierr);
    ierr = VecSet(fields->dervRs,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->P);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->P,"Pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->P,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Pw);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pw,"Water_Pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Pw,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPw);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPw,"Previous_time_step_water_pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPw,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Po);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Po,"Oil_Pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Po,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPo);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPo,"Previous time step oil pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPo,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Pg);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pg,"Gas pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Pg,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPg);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPg,"Previous time step gas pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPg,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Pcow);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pcow,"Oil/water capillary pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Pcow,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPcow);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPcow,"Previous time step oil/water capillary pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPcow,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->dervPcow);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->dervPcow,"Pcow derivative wrt Sw");CHKERRQ(ierr);
    ierr = VecSet(fields->dervPcow,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Pcog);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pcog,"Oil/gas capillary pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->Pcog,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oPcog);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPcog,"Previous time step oil/gas capillary pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oPcog,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->dervPcog);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->dervPcog,"Pcog derivative wrt Sw");CHKERRQ(ierr);
    ierr = VecSet(fields->dervPcog,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Sw);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Sw,"Water_saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->Sw,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oSw);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oSw,"Previous_time_step_water_saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->oSw,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->So);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->So,"Oil saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->So,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oSo);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oSo,"Previous_time_step_oil_saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->oSo,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->Sg);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Sg,"Gas_saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->Sg,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oSg);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oSg,"Previous_time_step_gas_saturation");CHKERRQ(ierr);
    ierr = VecSet(fields->oSg,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->T);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->T,"Temperature");CHKERRQ(ierr);
    ierr = VecSet(fields->T,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->V);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->V,"Fluid velocity");CHKERRQ(ierr);
    ierr = VecSet(fields->V,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->oU);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oU,"Previous time step displacement");CHKERRQ(ierr);
    ierr = VecSet(fields->oU,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oP);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oP,"Previous time step pressure");CHKERRQ(ierr);
    ierr = VecSet(fields->oP,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oT,"Previous time step temperature");CHKERRQ(ierr);
    ierr = VecSet(fields->oT,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->oV);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oV,"Previous time step fluid velocity");CHKERRQ(ierr);
    ierr = VecSet(fields->oV,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->Fb);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Fb,"Mechanics body force");CHKERRQ(ierr);
    ierr = VecSet(fields->Fb,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->oq);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oq,"Previous time step fluid flow rate");CHKERRQ(ierr);
    ierr = VecSet(fields->oq,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecNode, &fields->q);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->q,"Fluid flow rate");CHKERRQ(ierr);
    ierr = VecSet(fields->q,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->QT);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->QT,"Uniform heat source");CHKERRQ(ierr);
    ierr = VecSet(fields->QT,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->QP);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->QP,"Uniform fluid source");CHKERRQ(ierr);
    ierr = VecSet(fields->QP,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->MultiPhasePacker, &fields->FlowPacker);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->FlowPacker,"Packed flow solution");CHKERRQ(ierr);
    ierr = VecSet(fields->FlowPacker,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->Pbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Pbh,"Well pressure solutions");CHKERRQ(ierr);
    ierr = VecSet(fields->Pbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->oPbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oPbh,"Previous well pressure solutions");CHKERRQ(ierr);
    ierr = VecSet(fields->oPbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->Qwbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Qwbh,"Well water Rate");CHKERRQ(ierr);
    ierr = VecSet(fields->Qwbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->Qobh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Qobh,"Well oil Rate");CHKERRQ(ierr);
    ierr = VecSet(fields->Qobh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->Qgbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->Qgbh,"Well gas Rate");CHKERRQ(ierr);
    ierr = VecSet(fields->Qgbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->QLbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->QLbh,"Well total liquid Rate");CHKERRQ(ierr);
    ierr = VecSet(fields->QLbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->WellRedun, &fields->QTbh);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->QTbh,"Well total fluid Rate");CHKERRQ(ierr);
    ierr = VecSet(fields->QTbh,0.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->SaturatedStateIndicator);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->SaturatedStateIndicator,"Saturate_unsaturated_state_indicator (0 for saturated, 1 for unsaturated)");CHKERRQ(ierr);
    ierr = VecSet(fields->SaturatedStateIndicator,0.);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->INDC1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->INDC1,"Saturation_Switching_Indicator1");CHKERRQ(ierr);
    ierr = VecSet(fields->INDC1,1.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->INDC2);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->INDC2,"Saturation_Switching_Indicator2");CHKERRQ(ierr);
    ierr = VecSet(fields->INDC2,3.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->SgRs);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->SgRs,"Combined_Sg_Rs_variable");CHKERRQ(ierr);
    ierr = VecSet(fields->SgRs,3.0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &fields->oSgRs);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fields->oSgRs,"Old_Combined_Sg_Rs_variable");CHKERRQ(ierr);
    ierr = VecSet(fields->oSgRs,3.0);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDGetResFluidProps"
extern PetscErrorCode FracDGetResFluidProps(AppCtx *bag,FracDPpty *ppties)
{
    PetscErrorCode ierr;
    PetscReal      E,nu,alpha,beta,phi,Cp,rhos;
    PetscReal      *perm,*cond,*g, scale;
    PetscInt       i,j,l,n,nopt,cordsize,numValues,numValues1,numValues2;
    PetscInt       vStart, vEnd, v, cStart, cEnd, c;
    PetscSection   scalSection, matSection, volSection, cordSection;
    Vec            local_perm, local_cond, local_vol, local_dualvol;
    PetscScalar    *perm_array = NULL, *cond_array = NULL, *vol_array = NULL, *dualvol_array = NULL;
    Vec            coordinates;
    PetscScalar    *coord_array = NULL;
    PetscReal      **coords;
    PetscInt       numclpts, *closurept=NULL;
    DM             cdm;
    PetscBool      flg;
    
    PetscFunctionBegin;
    n = bag->dim;
    scale = bag->CVFEface.scale;
    ierr = PetscMalloc(n * sizeof(PetscReal),&perm);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscReal),&cond);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscReal),&g);CHKERRQ(ierr);
    ierr = PetscMalloc(n * sizeof(PetscReal),&ppties->g);CHKERRQ(ierr);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"\n\n FracD: reservoir, fluid and simulation parameters/properties:","");CHKERRQ(ierr);
    {
        bag->timevalue = 1.0;
        ierr = PetscOptionsReal("-timevalue","\n\t Time step size","",bag->timevalue,&bag->timevalue,PETSC_NULL);CHKERRQ(ierr);
        bag->maxtimestep = 1;
        ierr = PetscOptionsInt("-maxtimestep","\n\t Maximum number of timestep","",bag->maxtimestep,&bag->maxtimestep,PETSC_NULL);CHKERRQ(ierr);
        bag->theta = 1.0;
        ierr = PetscOptionsReal("-theta","\n\t Time parameter","",bag->theta,&bag->theta,PETSC_NULL);CHKERRQ(ierr);
        E  = 1;
        ierr = PetscOptionsReal("-E","\n\t Reservoir Young's modulus","",E,&E,PETSC_NULL);CHKERRQ(ierr);
        nu  = 0.2;
        ierr = PetscOptionsReal("-nu","\n\t Reservoir Poisson's ratio","",nu,&nu,PETSC_NULL);CHKERRQ(ierr);
        
        for (i = 0; i < n; i++) perm[i] = 1.;
        nopt = n;
        ierr = PetscOptionsRealArray("-perm","\n\tComma separated list of reservoir permeability","",perm,&nopt,&flg);CHKERRQ(ierr);
        if (nopt != n && flg) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of the reservoir permeability, got %i in %s\n",n,nopt,__FUNCT__);
        
        flg = PETSC_FALSE;
        for (i = 0; i < n; i++) cond[i] = 1.;
        nopt = n;
        ierr = PetscOptionsRealArray("-conductivity","\n\tComma separated list of reservoir conductivity","",cond,&nopt,&flg);CHKERRQ(ierr);
        if (nopt != n && flg) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of the reservoir thermal conductivity, got %i in %s\n",n,nopt,__FUNCT__);
        
        alpha  = 1;
        ierr = PetscOptionsReal("-alpha","\n\t Reservoir linear thermal expansion coefficient","",alpha,&alpha,PETSC_NULL);CHKERRQ(ierr);
        
        beta  = 1;
        ierr = PetscOptionsReal("-beta","\n\t Reservoir Biot's constant","",beta,&beta,PETSC_NULL);CHKERRQ(ierr);
        
        phi  = 0.2;
        ierr = PetscOptionsReal("-phi","\n\t Reservoir porosity","",phi,&phi,PETSC_NULL);CHKERRQ(ierr);
        
        bag->ppties.PhiData[1]  = 0.2;
        ierr = PetscOptionsReal("-Cr","\n\t Reservoir compressibility","",bag->ppties.PhiData[1],&bag->ppties.PhiData[1],PETSC_NULL);CHKERRQ(ierr);
        bag->ppties.PhiData[2] = -1*bag->ppties.PhiData[1];
        
        
        bag->ppties.PhiData[2]  = 0;
        ierr = PetscOptionsReal("-pref_phi","\n\t Reference pressure for porosity calculation","",bag->ppties.PhiData[2],&bag->ppties.PhiData[2],PETSC_NULL);CHKERRQ(ierr);
        
        Cp  = 1.;
        ierr = PetscOptionsReal("-Cp","\n\t Specific heat capacity of rock","",Cp,&Cp,PETSC_NULL);CHKERRQ(ierr);
        
        ppties->Cpw  = 1;
        ierr = PetscOptionsReal("-Cpw","\n\t Specific heat capacity of fluid","",ppties->Cpw,&ppties->Cpw,PETSC_NULL);CHKERRQ(ierr);
        
        rhos  = 1.;
        ierr = PetscOptionsReal("-rhos","\n\t Rock density","",rhos,&rhos,PETSC_NULL);CHKERRQ(ierr);
        
        bag->P_ref  = 0;
        ierr = PetscOptionsReal("-refpressure","\n\t Reference reservoir pressure","",bag->P_ref,&bag->P_ref,PETSC_NULL);CHKERRQ(ierr);
        
        bag->T_ref  = 0;
        ierr = PetscOptionsReal("-reftemp","\n\t Reference reservoir temperature","",bag->T_ref,&bag->T_ref,PETSC_NULL);CHKERRQ(ierr);
        
        bag->S_ref  = 0;
        ierr = PetscOptionsReal("-refSat","\n\t Reference water saturation","",bag->S_ref,&bag->S_ref,PETSC_NULL);CHKERRQ(ierr);
        
        bag->Sw_ref  = 0;
        ierr = PetscOptionsReal("-refSw","\n\t Reference water saturation","",bag->Sw_ref,&bag->Sw_ref,PETSC_NULL);CHKERRQ(ierr);
        
        bag->So_ref  = 0;
        ierr = PetscOptionsReal("-refSo","\n\t Reference oil saturation","",bag->So_ref,&bag->So_ref,PETSC_NULL);CHKERRQ(ierr);
        
        for (i = 0; i < n; i++) g[i] = 0.;
        nopt = n;
        ierr = PetscOptionsRealArray("-gravity","\n\tComma separated list of gravity","",g,&nopt,&flg);CHKERRQ(ierr);
        if (nopt != n && flg) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of gravity, got %i in %s\n",n,nopt,__FUNCT__);
        for (i = 0; i < n; i++) ppties->g[i] = g[i];
        
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(bag->plexVecCell,0,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(bag->plexVecCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalCell,&volSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexVecCell,&matSection);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(bag->plexScalNode,&scalSection);CHKERRQ(ierr);
    
    coords = (PetscReal **)malloc(bag->eD.nodes * sizeof(PetscReal *));
    for(i = 0; i < bag->eD.nodes; i++)
    {
        coords[i] = (PetscReal *)malloc(bag->eD.dim * sizeof(PetscReal));
    }
    ierr = DMGetCoordinatesLocal(bag->plexVecCell,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(bag->plexVecCell, &cdm);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(cdm, &cordSection);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->CellVolume);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->CellVolume,"Cell Volume");CHKERRQ(ierr);
    ierr = VecSet(ppties->CellVolume,0);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_vol);CHKERRQ(ierr);
    ierr = VecSet(local_vol,0);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecCell, &ppties->perm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->perm,"Permeability");CHKERRQ(ierr);
    ierr = VecSet(ppties->perm,1.);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    ierr = VecSet(local_perm,1);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexVecCell, &ppties->cond);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->cond,"Thermal conductivity");CHKERRQ(ierr);
    ierr = VecSet(ppties->cond,1.);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    ierr = VecSet(local_cond,1);CHKERRQ(ierr);
    
    ierr = DMPlexVecGetClosure(bag->plexVecCell, matSection, local_perm, cStart, &numValues, NULL);CHKERRQ(ierr);
    ierr = DMPlexVecGetClosure(bag->plexScalCell, volSection, local_vol, cStart, &numValues1, NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexVecCell, numValues, MPIU_SCALAR, &perm_array);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexVecCell, numValues, MPIU_SCALAR, &cond_array);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexScalCell, numValues1, MPIU_SCALAR, &vol_array);CHKERRQ(ierr);
    
    for(c = cStart; c < cEnd; ++c){
        ierr = DMPlexVecGetClosure(cdm, cordSection, coordinates, c, &cordsize, &coord_array);CHKERRQ(ierr);
        for(l = 0, i = 0; i < bag->eD.nodes; i++){
            for(j = 0; j < bag->eD.dim; j++, l++){
                coords[i][j] = coord_array[l];
            }
        }
        ierr = bag->FracDCreateDFEElement(coords, &bag->eD);CHKERRQ(ierr);
        vol_array[0] = bag->eD.Volume;
        for(i = 0; i < bag->dim; i++){
            perm_array[i] = perm[i];
            cond_array[i] = cond[i];
        }
        ierr = DMPlexVecSetClosure(bag->plexScalCell, volSection, local_vol, c, vol_array, INSERT_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecSetClosure(bag->plexVecCell, matSection, local_perm, c, perm_array, INSERT_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecSetClosure(bag->plexVecCell, matSection, local_cond, c, cond_array, INSERT_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(cdm, cordSection, coordinates, c, &cordsize, &coord_array);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(bag->plexVecCell, numValues, PETSC_SCALAR, &perm_array);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(bag->plexVecCell, numValues, PETSC_SCALAR, &cond_array);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(bag->plexScalCell, numValues1, PETSC_SCALAR, &vol_array);CHKERRQ(ierr);
    
    ierr = DMLocalToGlobalBegin(bag->plexScalCell,local_vol,INSERT_VALUES,ppties->CellVolume);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexScalCell,local_vol,INSERT_VALUES,ppties->CellVolume);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_vol);CHKERRQ(ierr);
    
    ierr = DMLocalToGlobalBegin(bag->plexVecCell,local_perm,INSERT_VALUES,ppties->perm);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexVecCell,local_perm,INSERT_VALUES,ppties->perm);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_perm);CHKERRQ(ierr);
    
    ierr = DMLocalToGlobalBegin(bag->plexVecCell,local_cond,INSERT_VALUES,ppties->cond);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexVecCell,local_cond,INSERT_VALUES,ppties->cond);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexVecCell,&local_cond);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalNode, &ppties->dualCellVolume);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->dualCellVolume,"Dual mesh cell volume");CHKERRQ(ierr);
    ierr = VecSet(ppties->dualCellVolume,0);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalNode,&local_dualvol);CHKERRQ(ierr);
    ierr = VecSet(local_dualvol,0);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->plexScalCell,&local_vol);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(bag->plexScalCell,ppties->CellVolume,INSERT_VALUES,local_vol);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(bag->plexScalCell,ppties->CellVolume,INSERT_VALUES,local_vol);CHKERRQ(ierr);
    
    ierr = DMPlexVecGetClosure(bag->plexScalNode, scalSection, local_dualvol, vStart, &numValues2, NULL);CHKERRQ(ierr);
    ierr = DMGetWorkArray(bag->plexScalNode, numValues2, MPIU_SCALAR, &dualvol_array);CHKERRQ(ierr);
    for(v = vStart; v < vEnd; ++v){
        dualvol_array[0] = 0;
        ierr = DMPlexGetTransitiveClosure(bag->plexScalNode, v, PETSC_FALSE, &numclpts, &closurept);CHKERRQ(ierr);
        for(i = 0; i < numclpts; i++){
            c = closurept[2*i];
            if(c >= cStart && c < cEnd){
                ierr = DMPlexVecGetClosure(bag->plexScalCell, volSection, local_vol, c, NULL, &vol_array);CHKERRQ(ierr);
                dualvol_array[0] += vol_array[0];
                ierr = DMPlexVecRestoreClosure(bag->plexScalCell, volSection, local_vol, c, NULL, &vol_array);CHKERRQ(ierr);
            }
        }
        dualvol_array[0] = scale*dualvol_array[0];
        ierr = DMPlexVecSetClosure(bag->plexScalNode, scalSection, local_dualvol, v, dualvol_array, ADD_ALL_VALUES);CHKERRQ(ierr);
        ierr = DMPlexRestoreTransitiveClosure(bag->plexScalNode, v, PETSC_FALSE, &numclpts, &closurept);CHKERRQ(ierr);
    }
    ierr = DMRestoreWorkArray(bag->plexScalNode, numValues2, PETSC_SCALAR, &dualvol_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalCell,&local_vol);CHKERRQ(ierr);
    
    ierr = DMLocalToGlobalBegin(bag->plexScalNode,local_dualvol,ADD_VALUES,ppties->dualCellVolume);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->plexScalNode,local_dualvol,ADD_VALUES,ppties->dualCellVolume);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->plexScalNode,&local_dualvol);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->E);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->E,"Young's modulus");CHKERRQ(ierr);
    ierr = VecSet(ppties->E,E);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->nu);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->nu,"Poisson's ratio");CHKERRQ(ierr);
    ierr = VecSet(ppties->nu,nu);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->alpha);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->alpha,"Linear thermal expansion coefficient");CHKERRQ(ierr);
    ierr = VecSet(ppties->alpha,alpha);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->beta);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->beta,"Biot's coefficient");CHKERRQ(ierr);
    ierr = VecSet(ppties->beta,beta);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->phi);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->phi,"Porosity");CHKERRQ(ierr);
    ierr = VecSet(ppties->phi,phi);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->Cp);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->Cp,"Rock specific heat capacity");CHKERRQ(ierr);
    ierr = VecSet(ppties->Cp,Cp);CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(bag->plexScalCell, &ppties->rhos);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) ppties->rhos,"Rock density");CHKERRQ(ierr);
    ierr = VecSet(ppties->rhos,rhos);CHKERRQ(ierr);
    
    ierr = PetscFree(perm);CHKERRQ(ierr);
    ierr = PetscFree(cond);CHKERRQ(ierr);
    ierr = PetscFree(g);CHKERRQ(ierr);
    for(i = 0; i < bag->eD.nodes; i++){
        free(coords[i]);
    }
    free(coords);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFinalizePVTData"
extern PetscErrorCode FracDFinalizePVTData(AppCtx *bag)
{
    PetscInt    i;
    PetscFunctionBegin;
    if(bag->fluid == SINGLEPHASELIQUID || bag->fluid == OILWATER || bag->fluid == OILWATERGAS){
        if(bag->ppties.WaterPVTData.FVFtype == INTERPOLATION || bag->ppties.WaterPVTData.mutype == INTERPOLATION || bag->ppties.WaterPVTData.rhotype == INTERPOLATION){
            for(i = 0; i < 3; i++){
                free(bag->ppties.WaterPVTData.B_TableData[i]);
                free(bag->ppties.WaterPVTData.mu_TableData[i]);
                free(bag->ppties.WaterPVTData.rho_TableData[i]);
            }
            free(bag->ppties.WaterPVTData.B_TableData);
            free(bag->ppties.WaterPVTData.mu_TableData);
            free(bag->ppties.WaterPVTData.rho_TableData);
        }
    }
    if(bag->fluid == OILWATER || bag->fluid == OILWATERGAS){
        if(bag->ppties.OilPVTData.FVFtype != ANALYTICAL || bag->ppties.OilPVTData.mutype != ANALYTICAL || bag->ppties.OilPVTData.rhotype != ANALYTICAL){
            for(i = 0; i < 3; i++){
                free(bag->ppties.OilPVTData.B_TableData[i]);
                free(bag->ppties.OilPVTData.mu_TableData[i]);
                free(bag->ppties.OilPVTData.rho_TableData[i]);
            }
            free(bag->ppties.OilPVTData.B_TableData);
            free(bag->ppties.OilPVTData.mu_TableData);
            free(bag->ppties.OilPVTData.rho_TableData);
        }
    }
    if(bag->fluid == SINGLEPHASEGAS || bag->fluid == OILWATERGAS){
        if(bag->ppties.GasPVTData.FVFtype == INTERPOLATION || bag->ppties.GasPVTData.mutype == INTERPOLATION || bag->ppties.GasPVTData.rhotype == INTERPOLATION){
            for(i = 0; i < 3; i++){
                free(bag->ppties.GasPVTData.B_TableData[i]);
                free(bag->ppties.GasPVTData.mu_TableData[i]);
                free(bag->ppties.GasPVTData.rho_TableData[i]);
            }
            free(bag->ppties.GasPVTData.B_TableData);
            free(bag->ppties.GasPVTData.mu_TableData);
            free(bag->ppties.GasPVTData.rho_TableData);
        }
    }
    if (bag->fluid == OILWATERGAS){
        for(i = 0; i < 3; i++){
            free(bag->ppties.SolutionGasOilData.TableData[i]);
            free(bag->ppties.SolutionGasOilData.TableDataInv[i]);
        }
        free(bag->ppties.SolutionGasOilData.TableData);
        free(bag->ppties.SolutionGasOilData.TableDataInv);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFinalizeRelPermCapillaryPressureData"
extern PetscErrorCode FracDFinalizeRelPermCapillaryPressureData(FracDFluidSystem fluid, FracDRelPerm *relPerm, FracDCapPress *capPress)
{
    PetscInt            i;
    PetscFunctionBegin;
    if ((fluid == OILWATER) || (fluid == OILWATERGAS))
    {
        for(i = 0; i < 3; i++){
            free(relPerm->Krw_TableData[i]);
            free(relPerm->Krow_TableData[i]);
            free(capPress->Pcow_TableData[i]);
        }
        free(relPerm->Krw_TableData);
        free(relPerm->Krow_TableData);
        free(capPress->Pcow_TableData);
    }
    if (fluid == OILWATERGAS)
    {
        for(i = 0; i < 3; i++){
            free(relPerm->Krg_TableData[i]);
            free(relPerm->Krog_TableData[i]);
            free(capPress->Pcog_TableData[i]);
        }
        free(relPerm->Krg_TableData);
        free(relPerm->Krog_TableData);
        free(capPress->Pcog_TableData);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDFinalizeWells"
extern PetscErrorCode FracDFinalizeWells(AppCtx *bag)
{
    PetscErrorCode      ierr;
    PetscInt            i;
    
    PetscFunctionBeginUser;
    for (i = 0; i < bag->numWells; i++) {
        ierr = FracDWellDestroy(&(bag->well[i]),bag->dim);CHKERRQ(ierr);
    }
    ierr = PetscFree(bag->well);CHKERRQ(ierr);
    
    for(i = 0; i < bag->WellinMeshData.numberWellsInProcessor; i++)
    {
        ierr = PetscFree(bag->WellinMeshData.WellInfo[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(bag->WellinMeshData.WellInfo);CHKERRQ(ierr);
    ierr = PetscFree(bag->WellinMeshData.Count);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDestroyResFluidProps"
extern PetscErrorCode FracDDestroyResFluidProps(AppCtx *bag,FracDPpty *ppties)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = VecDestroy(&ppties->CellVolume);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->dualCellVolume);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->E);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->nu);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->perm);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->cond);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->alpha);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->beta);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->phi);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->Cp);CHKERRQ(ierr);
    ierr = VecDestroy(&ppties->rhos);CHKERRQ(ierr);
    ierr = PetscFree(ppties->g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDestroyFields"
extern PetscErrorCode FracDDestroyFields(AppCtx *bag,FracDFields *fields)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = VecDestroy(&fields->U);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Pb);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPb);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Rs);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oRs);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->dervRs);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->P);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->T);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->V);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oV);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oU);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oP);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oT);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Fb);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->q);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oq);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->QT);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->QP);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Pw);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPw);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Po);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPo);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Pcow);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPcow);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->dervPcow);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Pg);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPg);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->dervPcog);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Pcog);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oPcog);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Sw);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oSw);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->So);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oSo);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->Sg);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oSg);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.Pbh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.oPbh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.Qwbh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.Qobh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.Qgbh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.QLbh);CHKERRQ(ierr);
    ierr = VecDestroy(&bag->fields.QTbh);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->FlowPacker);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->SaturatedStateIndicator);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->INDC1);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->INDC2);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->SgRs);CHKERRQ(ierr);
    ierr = VecDestroy(&fields->oSgRs);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDDestroySolvers"
extern PetscErrorCode FracDDestroySolvers(AppCtx *bag)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = SNESDestroy(&bag->snesU);CHKERRQ(ierr);
    ierr = SNESDestroy(&bag->snesP);CHKERRQ(ierr);
    ierr = SNESDestroy(&bag->snesT);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDDestroyBoundaryConditions"
extern PetscErrorCode FracDDestroyBoundaryConditions(AppCtx *bag)
{
    PetscErrorCode      ierr;
    
    PetscFunctionBegin;
    if (bag->UBC.hasLabel) {
        ierr = PetscFree(bag->UBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->UBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->UBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->UBC.regions);CHKERRQ(ierr);
    }
    if (bag->TractionBC.hasLabel) {
        ierr = PetscFree(bag->TractionBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->TractionBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->TractionBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->TractionBC.regions);CHKERRQ(ierr);
    }
    if (bag->PBC.hasLabel) {
        ierr = PetscFree(bag->PBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->PBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->PBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->PBC.regions);CHKERRQ(ierr);
    }
    if (bag->FlowFluxBC.hasLabel) {
        ierr = PetscFree(bag->FlowFluxBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->FlowFluxBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->FlowFluxBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->FlowFluxBC.regions);CHKERRQ(ierr);
    }
    if (bag->TBC.hasLabel) {
        ierr = PetscFree(bag->TBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->TBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->TBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->TBC.regions);CHKERRQ(ierr);
    }
    if (bag->HeatFluxBC.hasLabel) {
        ierr = PetscFree(bag->HeatFluxBC.values);CHKERRQ(ierr);
        ierr = PetscFree(bag->HeatFluxBC.components);CHKERRQ(ierr);
        ierr = PetscFree(bag->HeatFluxBC.numcompsperlabel);CHKERRQ(ierr);
        ierr = PetscFree(bag->HeatFluxBC.regions);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "myWriteFunc"
extern PetscErrorCode myWriteFunc(PetscObject vec,PetscViewer viewer)
{
    PetscErrorCode      ierr;
    Vec             V = (Vec) vec;
    PetscFunctionBegin;
    ierr = VecView(V,viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ThisismyVTKwriter"
extern PetscErrorCode ThisismyVTKwriter(AppCtx *bag)
{
    PetscViewer viewer;
    PetscErrorCode      ierr;
    char           filename[FILENAME_MAX];
    
    PetscFunctionBegin;
    ierr = PetscSNPrintf(filename,FILENAME_MAX,"output_nodal.%.5i.vtk",bag->timestep);CHKERRQ(ierr);
    
    printf("\n Checking %d \n",bag->timestep);
    
    //
    ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(bag->fields.P,viewer);CHKERRQ(ierr);
    VecView(bag->fields.Sw,viewer);CHKERRQ(ierr);
    VecView(bag->fields.Sg,viewer);CHKERRQ(ierr);
    VecView(bag->fields.Rs,viewer);CHKERRQ(ierr);
    VecView(bag->fields.Pb,viewer);CHKERRQ(ierr);
    ierr = VecView(bag->fields.T,viewer);CHKERRQ(ierr);
    //    ierr = VecView(bag->fields.U,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDTimeStepPrepare"
extern PetscErrorCode FracDTimeStepPrepare(AppCtx *bag)
{
    PetscErrorCode          ierr;
    PetscInt                i,rank;
    Vec                     X1,X2,X3,X4,localPbh;
    PetscScalar             *Pbh_array=NULL;
    
    PetscFunctionBegin;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    bag->ppties.SolutionGasOilData.BubblePointFixed = bag->ppties.SolutionGasOilData.SolutionGasOilRatioFixed = bag->Sg_initial = 0.;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Initial state of the reservoir", "");CHKERRQ(ierr);
    {
        if (bag->fluid == OILWATERGAS){
            ierr = PetscOptionsReal("-initialbubblepoint","\n\t Initial bubble point pressure:","",bag->ppties.SolutionGasOilData.BubblePointFixed,&bag->ppties.SolutionGasOilData.BubblePointFixed,PETSC_NULL);CHKERRQ(ierr);
            ierr = PetscOptionsReal("-initialSg","\n\t Initial gas saturation:","",bag->Sg_initial,&bag->Sg_initial,PETSC_NULL);CHKERRQ(ierr);
        }
        ierr = bag->ppties.SolutionGasOilData.FracDUpDateSolutionGasOilRatio(&bag->ppties.SolutionGasOilData.SolutionGasOilRatioFixed,bag->ppties.SolutionGasOilData.BubblePointFixed,PETSC_NULL,bag->ppties.SolutionGasOilData.TableData,bag->ppties.SolutionGasOilData.ModelData,bag->ppties.SolutionGasOilData.numdatarow);CHKERRQ(ierr);
        bag->P_initial  = 0.;
        ierr = PetscOptionsReal("-initialpressure","\n\t Initial pressure:","",bag->P_initial,&bag->P_initial,PETSC_NULL);CHKERRQ(ierr);
        bag->T_initial  = 0.;
        ierr = PetscOptionsReal("-initialtemperature","\n\t Initial pressure:","",bag->T_initial,&bag->T_initial,PETSC_NULL);CHKERRQ(ierr);
        bag->Sw_initial  = 1.;
        if(bag->fluid == OILWATER || bag->fluid == OILWATERGAS){
            ierr = PetscOptionsReal("-initialSw","\n\t Initial water saturation:","",bag->Sw_initial,&bag->Sw_initial,PETSC_NULL);CHKERRQ(ierr);
        }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    
    ierr = VecSet(bag->fields.oP,bag->P_initial);CHKERRQ(ierr);
    ierr = VecSet(bag->fields.oT,bag->T_initial);CHKERRQ(ierr);
    ierr = VecSet(bag->fields.oSw,bag->Sw_initial);CHKERRQ(ierr);
    ierr = VecSet(bag->fields.oSg,bag->Sg_initial);CHKERRQ(ierr);
    ierr = VecSet(bag->fields.oSo,(1-bag->Sw_initial-bag->Sg_initial));CHKERRQ(ierr);
    if(bag->P_initial > bag->ppties.SolutionGasOilData.BubblePointFixed){
        ierr = VecSet(bag->fields.SaturatedStateIndicator,0.);CHKERRQ(ierr);
    }
    else{
        ierr = VecSet(bag->fields.SaturatedStateIndicator,1.);CHKERRQ(ierr);
    }
    

    
//    ierr = FracDInitializeSwitchingVariables(bag->fields.INDC1,bag->fields.INDC2,bag->fields.Sg);CHKERRQ(ierr);
    
    
    ierr = VecSet(bag->fields.oPb,bag->ppties.SolutionGasOilData.BubblePointFixed);CHKERRQ(ierr);
    ierr = VecSet(bag->fields.oRs,bag->ppties.SolutionGasOilData.SolutionGasOilRatioFixed);CHKERRQ(ierr);

    ierr = VecCopy(bag->fields.oP,bag->fields.P);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oT,bag->fields.T);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oSw,bag->fields.Sw);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oSo,bag->fields.So);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oSg,bag->fields.Sg);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oPb,bag->fields.Pb);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.oRs,bag->fields.Rs);CHKERRQ(ierr);
    
    ierr = DMGetLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    ierr = VecSet(localPbh,0.);CHKERRQ(ierr);
    ierr = VecGetArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    if(!rank){
        for (i = 0; i < bag->numWells; i++)
        if(bag->well[i].condition == PRESSURE)  Pbh_array[i] = bag->well[i].Pbh;
    }
    ierr = VecRestoreArray(localPbh,&Pbh_array);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(bag->WellRedun,localPbh,ADD_VALUES,bag->fields.Pbh);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(bag->WellRedun,localPbh,ADD_VALUES,bag->fields.Pbh);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(bag->WellRedun,&localPbh);CHKERRQ(ierr);
    
    switch (bag->fluid) {
        case 0:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.P,X1);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Pbh,X4);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
        }
        break;
        case 1:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.P,X1);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Pbh,X4);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
            ierr = VecSet(bag->fields.oSw,0.);CHKERRQ(ierr);
            ierr = VecSet(bag->fields.oSo,0.);CHKERRQ(ierr);
            ierr = VecSet(bag->fields.oSg,1.);CHKERRQ(ierr);
        }
        break;
        case 2:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X4);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.P,X1);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Sw,X2);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Pbh,X4);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X4);CHKERRQ(ierr);
        }
        break;
        case 3:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X3,&X4);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.P,X1);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Sw,X2);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Sg,X3);CHKERRQ(ierr);
            ierr = VecCopy(bag->fields.Pbh,X4);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X3,&X4);CHKERRQ(ierr);
        }
        break;
    }
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDTimeStepUpdate"
extern PetscErrorCode FracDTimeStepUpdate(AppCtx *bag)
{
    PetscErrorCode          ierr;
    Vec                     X1,X2,X3,X4;
    PetscFunctionBegin;
    switch (bag->fluid) {
        case 0:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
            ierr = VecCopy(X1,bag->fields.P);CHKERRQ(ierr);
            ierr = VecCopy(X4,bag->fields.Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
        }
        break;
        case 1:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
            ierr = VecCopy(X1,bag->fields.P);CHKERRQ(ierr);
            ierr = VecCopy(X4,bag->fields.Pbh);CHKERRQ(ierr);
            ierr = DMCompositeRestoreAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X4);CHKERRQ(ierr);
        }
        break;
        case 2:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X4);CHKERRQ(ierr);
            ierr = VecCopy(X1,bag->fields.P);CHKERRQ(ierr);
            ierr = VecCopy(X2,bag->fields.Sw);CHKERRQ(ierr);
            ierr = VecCopy(X4,bag->fields.Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X4);CHKERRQ(ierr);
        }
        break;
        case 3:
        {
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X3,&X4);CHKERRQ(ierr);
            ierr = VecCopy(X1,bag->fields.P);CHKERRQ(ierr);
            ierr = VecCopy(X2,bag->fields.Sw);CHKERRQ(ierr);
            ierr = VecCopy(X3,bag->fields.Sg);CHKERRQ(ierr);
            ierr = VecCopy(X4,bag->fields.Pbh);CHKERRQ(ierr);
            ierr = DMCompositeGetAccess(bag->MultiPhasePacker,bag->fields.FlowPacker,&X1,&X2,&X3,&X4);CHKERRQ(ierr);
        }
        break;
    }
    ierr = VecCopy(bag->fields.U,bag->fields.oU);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.T,bag->fields.oT);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.P,bag->fields.oP);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.V,bag->fields.oV);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.q,bag->fields.oq);CHKERRQ(ierr);
    //    ierr = VecCopy(bag->fields.Pw,bag->fields.oPw);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Po,bag->fields.oPo);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Pg,bag->fields.oPg);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Sw,bag->fields.oSw);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.So,bag->fields.oSo);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Sg,bag->fields.oSg);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Pbh,bag->fields.oPbh);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Pb,bag->fields.oPb);CHKERRQ(ierr);
    ierr = VecCopy(bag->fields.Rs,bag->fields.oRs);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
