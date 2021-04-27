#define FracDWellDescription
#include "petsc.h"
#include "FracDWell.h"

static const char *WellConstraint_Name[] = {
    "PRESSURE",
    "WATERRATE",
    "OILRATE",
    "GASRATE",
    "LIQUIDRATE",
    "TOTALRATE",
    "WellConstraint_Name",
    "",
    0
};

static const char *WellType_Name[] = {
    "INJECTORWATER",
    "INJECTORGAS",
    "PRODUCER",
    "WellType_Name",
    "",
    0
};

#undef __FUNCT__
#define __FUNCT__ "FracDWellCreate"
extern PetscErrorCode FracDWellCreate(FracDWell *well, PetscInt dim)
{
    PetscErrorCode ierr;
    int            i;
    
    PetscFunctionBegin;
    well->dim = dim;
    ierr = PetscStrcpy(well->name,"well");CHKERRQ(ierr);
    well->top = (PetscReal *)malloc(well->dim * sizeof(PetscReal));
    well->bottom = (PetscReal *)malloc(well->dim * sizeof(PetscReal));
    well->coordinates = (PetscReal *)malloc(well->dim * sizeof(PetscReal));
    for (i = 0; i < well->dim; i++) {
        well->top[i] = 0.;
        well->bottom[i] = 0.;
        well->coordinates[i] = 0.;
    }
    well->Qws = 0.;
    well->Qos = 0.;
    well->Qgs = 0.;
    well->QL = 0.;
    well->QT = 0.;
    well->Pbh = 0.;
    well->h = 1.;
    well->rw = 0.;
    well->re = 0.;
    well->sk = 0.;
    well->condition = WATERRATE;
    well->type = PRODUCER;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDGetWell"
extern PetscErrorCode FracDGetWell(const char prefix[],FracDWell *well)
{
    PetscErrorCode ierr;
    PetscInt       nval;
    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,prefix,"\n\n FracD: well description:","");CHKERRQ(ierr);
    {
        ierr = PetscOptionsString("-name","\n\t well name","",well->name,well->name,sizeof(well->name),PETSC_NULL);CHKERRQ(ierr);
        nval = well->dim;
        ierr = PetscOptionsRealArray("-top","\n\t well top coordinates (comma separated).","",well->top,&nval,PETSC_NULL);CHKERRQ(ierr);
        if (nval != well->dim)  SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of the well top coordinates, got %i in %s\n",well->dim,nval,__FUNCT__);
        nval = well->dim;
        ierr = PetscOptionsRealArray("-bottom","\n\t well bottom coordinates  (comma separated).","",well->bottom,&nval,PETSC_NULL);CHKERRQ(ierr);
        if (nval != well->dim)  SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of the well bottom coordinates, got %i in %s\n",well->dim,nval,__FUNCT__);
        nval = well->dim;
        ierr = PetscOptionsRealArray("-coords","\n\t well coordinates  (comma separated).","",well->coordinates,&nval,PETSC_NULL);CHKERRQ(ierr);
        if (nval != well->dim)  SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Expecting %i components of the well coordinates, got %i in %s\n",well->dim,nval,__FUNCT__);
        ierr = PetscOptionsReal("-Qws","\n\t well water flow rate","",well->Qws,&well->Qws,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-Qos","\n\t well oil flow rate","",well->Qos,&well->Qos,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-Qgs","\n\t well gas flow rate","",well->Qgs,&well->Qgs,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-QL","\n\t total well liquid rate","",well->QL,&well->QL,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-QT","\n\t total well flow rate","",well->QT,&well->QT,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-Pbh","\n\t well bottomhole pressure","",well->Pbh,&well->Pbh,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-h","\n\t well height (well prodution interval)","",well->h,&well->h,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-rw","\n\t well radius","",well->rw,&well->rw,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsReal("-skin","\n\t well skin","",well->sk,&well->sk,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnum("-constraint","\n\t\n\t well constraint type","",WellConstraint_Name,(PetscEnum)well->condition,(PetscEnum*)&well->condition,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsEnum("-type","\n\t\n\t well type","",WellType_Name,(PetscEnum)well->type,(PetscEnum*)&well->type,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscOptionsInt("-wellindex","\n\t well-to-fracture index","",well->index,&well->index,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if(well->type == INJECTORWATER && (well->condition != WATERRATE && well->condition != PRESSURE)){
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Water Injection must have either WATERRATE or PRESSURE as well condition, got %s in %s\n",WellConstraint_Name[well->condition],__FUNCT__);
    }
    if(well->type == INJECTORGAS && (well->condition != GASRATE && well->condition != PRESSURE)){
        SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Gas Injection must have either GASRATE or PRESSURE as well condition, got %i in %s\n",well->condition,__FUNCT__);
    }
    if(well->Qws < 0.)  SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Water injection rate cannot be negative, got %g in %s\n",(double)well->Qws,__FUNCT__);
    if(well->Qos < 0.)  SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Oil injection rate cannot be negative, got %g in %s\n",(double)well->Qos,__FUNCT__);
    if(well->Qgs < 0.)  SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Gas injection rate cannot be negative, got %g in %s\n",(double)well->Qgs,__FUNCT__);
    if(well->QL < 0.)  SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Liquid injection rate cannot be negative, got %g in %s\n",(double)well->QL,__FUNCT__);
    if(well->QT < 0.)  SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_USER,"ERROR: Total fluid injection rate cannot be negative, got %g in %s\n",(double)well->QT,__FUNCT__);
    if(well->type == PRODUCER){
        well->Qws = -1 * well->Qws;
        well->Qos = -1 * well->Qos;
        well->Qgs = -1 * well->Qgs;
        well->QT = -1 * well->QT;
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDWellView"
extern PetscErrorCode FracDWellView(FracDWell *well,PetscViewer viewer)
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = PetscViewerASCIIPrintf(viewer,"Well object \"%s\":\n",well->name);CHKERRQ(ierr);
    if(well->dim == 3){
        ierr = PetscViewerASCIIPrintf(viewer,"top:    \t%e \t%e \t%e\n",well->top[0],well->top[1],well->top[2]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"bottom: \t%e \t%e \t%e\n",well->bottom[0],well->bottom[1],well->bottom[2]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"wellnodes: \t%e \t%e \t%e\n",well->coordinates[0],well->coordinates[1],well->coordinates[2]);CHKERRQ(ierr);
    }
    if(well->dim == 2){
        ierr = PetscViewerASCIIPrintf(viewer,"top:    \t%e \t%e \n",well->top[0],well->top[1]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"bottom: \t%e \t%e \n",well->bottom[0],well->bottom[1]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"wellnodes: \t%e \t%e \n",well->coordinates[0],well->coordinates[1]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"well water rate:     \t%e\n",well->Qws);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well oil rate:     \t%e\n",well->Qos);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well gas rate:     \t%e\n",well->Qgs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"total well rate:     \t%e\n",well->QT);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well bottomhole pressure:     \t%e\n",well->Pbh);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well height/production interval:     \t%e\n",well->h);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well radius:     \t%e\n",well->rw);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well block equivalent radius:     \t%e\n",well->re);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"well skin:     \t%e\n",well->sk);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Well Type:	\"%s\" well under \"%s\" condition\n",WellType_Name[well->type],WellConstraint_Name[well->condition]);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FracDWellSetName"
extern PetscErrorCode FracDWellSetName(FracDWell *well,const char name[])
{
    PetscErrorCode ierr;
    
    PetscFunctionBegin;
    ierr = PetscStrcpy(well->name,name);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FracDWellDestroy"
extern PetscErrorCode FracDWellDestroy(FracDWell *well, PetscInt dim)
{
    PetscFunctionBegin;
    free(well->top);
    free(well->bottom);
    free(well->coordinates);
    PetscFunctionReturn(0);
}
