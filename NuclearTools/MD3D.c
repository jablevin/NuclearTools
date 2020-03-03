#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

enum {ERR_NONE, ERR_BOND_SNAPPED, ERR_CHECKPT_READ, ERR_CHECKPT_WRITE,
    ERR_COPY_BUFF_FULL, ERR_EMPTY_EVPOOL, ERR_MSG_BUFF_FULL,
    ERR_OUTSIDE_REGION, ERR_SNAP_READ, ERR_SNAP_WRITE,
    ERR_SUBDIV_UNFIN, ERR_TOO_MANY_CELLS, ERR_TOO_MANY_COPIES,
    ERR_TOO_MANY_LAYERS, ERR_TOO_MANY_LEVELS, ERR_TOO_MANY_MOLS,
    ERR_TOO_MANY_MOVES, ERR_TOO_MANY_NEBRS, ERR_TOO_MANY_REPLICAS};

char *errorMsg[] = {"", "bond snapped", "read checkpoint data",
    "write checkpoint data", "copy buffer full", "empty event pool",
    "message buffer full", "outside region", "read snap data",
    "write snap data", "subdivision unfinished", "too many cells",
    "too many copied mols", "too many layers", "too many levels",
    "too many mols", "too many moved mols", "too many neighbors",
    "too many replicas"};

typedef double real;
typedef struct { real x, y, z; } VecR;
typedef struct { VecR r, rv, ra; } Mol;
typedef struct { int x, y, z; } VecI;
typedef struct { real val, sum, sum2; } Prop;

// Random Constants
#define MASK 2147483647
#define SCALE 0.4656612873e-9
#define IADD 453806245
#define IMUL 314159269
#define N_OFFSET 14
#define OFFSET_VALS \
    {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {-1,1,0}, {0,0,1}, {1,0,1}, {1,1,1}, \
     {0,1,1}, {-1,1,1}, {-1,0,1}, {-1,-1,1}, {0,-1,1}, {1,-1,1}}

// Dimensionally-Based Macros
#define NDIM 3
#define VVAdd(v1, v2) VAdd (v1, v1, v2)
#define VAdd(v1, v2, v3) (v1).x = (v2).x + (v3).x, (v1).y = (v2).y + (v3).y, (v1).z = (v2).z + (v3).z
#define VSAdd(v1, v2, s3, v3) (v1).x = (v2).x + (s3)*(v3).x, \
                              (v1).y = (v2).y + (s3)*(v3).y, \
                              (v1).z = (v2).z + (s3)*(v3).z
#define VVSAdd(v1, s2, v2) VSAdd (v1, v1, s2, v2)
#define VVSub(v1, v2) VSub (v1, v1, v2)
#define VSub(v1, v2, v3) (v1).x = (v2).x - (v3).x, (v1).y = (v2).y - (v3).y, (v1).z = (v2).z - (v3).z
#define VDot(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)
#define VLen(v) sqrt (VDot (v, v))
#define VLenSq(v) VDot (v, v)
#define VSet(v, sx, sy, sz) (v).x = sx, (v).y = sy, (v).z = sz
#define VSetAll(v, s) VSet (v, s, s, s)
#define VZero(v) VSetAll (v, 0)
#define VMul(v1, v2, v3) (v1).x = (v2).x * (v3).x, (v1).y = (v2).y * (v3).y, (v1).z = (v2).z * (v3).z
#define VDiv(v1, v2, v3) (v1).x = (v2).x / (v3).x, (v1).y = (v2).y / (v3).y, (v1).z = (v2).z / (v3).z
#define VCSum(v) ((v).x + (v).y + (v).z)
#define VSCopy(v2, s1, v1) (v2).x = (s1) * (v1).x, (v2).y = (s1) * (v1).y, (v2).z = (s1) * (v1).z
#define VProd(v) ((v).x * (v).y * (v).z)
#define VScale(v, s) (v).x *= s, (v).y *= s, (v).z *= s
#define VLinear(p, s) (((p).z * (s).y + (p).y) * (s).x + (p).x)
#define VWrapAll(v) {VWrap (v, x); VWrap (v, y); VWrap (v, z);}
#define VWrap(v, t) if (v.t >= 0.5 * region.t) v.t -= region.t; else if (v.t < -0.5 * region.t) v.t += region.t
#define VCellWrap(t) \
    if (m2v.t >= cells.t) { m2v.t = 0; shift.t = region.t; } \
    else if (m2v.t < 0) { m2v.t = cells.t - 1; shift.t = - region.t; }
#define VCellWrapAll() {VCellWrap (x); VCellWrap (y); VCellWrap (z);}

// Other Macros
#define Max(x, y) (((x) > (y)) ? (x) : (y))
#define Min(x, y) (((x) < (y)) ? (x) : (y))
#define Sqr(x) ((x) * (x))
#define Cube(x) ((x) * (x) * (x))
#define DO_MOL for (n = 0; n < nMol; n ++)
#define DO_CELL(j, m) for (j = cellList[m]; j >= 0; j = cellList[j])
#define AllocMem(a, n, t) a = (t *) malloc ((n) * sizeof (t))
#define PropZero(v) v.sum = 0., v.sum2 = 0.
#define PropAccum(v) v.sum += v.val, v.sum2 += Sqr (v.val)
#define PropAvg(v, n) v.sum /= n, v.sum2 = sqrt(Max(v.sum2/n - Sqr (v.sum), 0.))
#define PropEst(v) v.sum, v.sum2

// Variables
Mol *mol;
VecR region, vSum;
VecI initUcell, cells;
Prop kinEnergy, pressure, totEnergy;
real deltaT, density, rCut, temperature, timeNow, uSum, velMag, virSum, vvSum;
real *histVel, rangeVel, hFunction, dispHi, rNebrShell, cpu_time;
int moreCycles, nMol, stepAvg, stepCount, stepEquil, stepLimit, randSeed;
int countVel, limitVel, sizeHistVel, stepVel, randSeedP, *cellList;
int *nebrTab, nebrNow, nebrTabFac, nebrTabLen, nebrTabMax;
int method, stepAdjustTemp, printout;
FILE *file;
clock_t start, end;


// Function Declarations
void SingleStep(), SetParams(), SetupJob(), LeapfrogStep(), BuildNebrList();
void ApplyBoundaryCond(), EvalProps(), VRand(), ErrExit(), BuildNebrListCombo();
void AccumProps(), PrintSummary(), InitCoords(), PrintVelDist();
void EvalVelDist(), AllocArrays(), InitVels(), InitAccels(), InitRand();
void ComputeForcesAll(), ComputeForcesCell(), ComputeForcesNbr();
real RandR(), gasdev();
////////////////////////////////////////////////////////////////////////////////
// Begin Main Program
////////////////////////////////////////////////////////////////////////////////
static PyObject* MD3D(PyObject* self, PyObject* args){
    FILE* in_file;
    char *s;
    if (!PyArg_ParseTuple(args, "s", &s)) {
        return NULL;}
    if ((in_file = fopen(s,"r")) == NULL){
        printf("Error! opening file");
        exit(1);}
        char* pend;
        char array[50], outfile[50];
        fscanf(in_file, "%s", array);
        deltaT = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        density = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        temperature = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        rCut = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        initUcell.x = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        initUcell.y = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        initUcell.z = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        nebrTabFac = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        rNebrShell = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepAvg = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepEquil = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepLimit = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepAdjustTemp = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        limitVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        rangeVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        sizeHistVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        stepVel = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        randSeedP = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        method = strtof(array, &pend);
        fscanf(in_file, "%s", array);
        printout = strtof(array, &pend);
        fscanf(in_file, "%s", outfile);
    file = fopen(outfile,"w");
    SetParams ();
    SetupJob ();
    moreCycles = 1;
    while (moreCycles) {
        SingleStep ();
        if (stepCount >= stepLimit) moreCycles = 0;}
    return Py_None;}

void SetParams (){
    VSCopy (region, 1. / pow (density / 4., 1./3.), initUcell);
    if (method == 0 || method == 1){
        VSCopy (cells, 1. / rCut, region);}
    if (method == 3){
        VSCopy (cells, 1. / (rCut + rNebrShell), region);}
    nMol = 4 * VProd (initUcell);
    if (method == 2 || method == 3){
        nebrTabMax = nebrTabFac * nMol;}
    velMag = sqrt (NDIM * (1. - 1. / nMol) * temperature);}


void SetupJob (){
    countVel = 0;
    if (method == 2 || method == 3){
        nebrNow = 1;}
    AllocArrays ();
    stepCount = 0;
    InitCoords ();
    InitVels ();
    InitAccels ();
    AccumProps (0);}


void AllocArrays (){
    AllocMem (mol, nMol, Mol);
    AllocMem (histVel, sizeHistVel, real);
    AllocMem (cellList, VProd (cells) + nMol, int);
    AllocMem (nebrTab, 2 * nebrTabMax, int);}


void AdjustTemp () {
    real vFac;
    int n;
    vvSum = 0.;
    DO_MOL vvSum += VLenSq (mol[n].rv);
    vFac = velMag / sqrt (vvSum / nMol);
    DO_MOL VScale (mol[n].rv, vFac);}


void InitCoords (){
    VecR c, gap;
    int j, n, nx, ny, nz;
    VDiv (gap, region, initUcell);
    n = 0;
    for (nz = 0; nz < initUcell.z; nz ++) {
        for (ny = 0; ny < initUcell.y; ny ++) {
            for (nx = 0; nx < initUcell.x; nx ++) {
                VSet (c, nx + 0.25, ny + 0.25, nz + 0.25);
                VMul (c, c, gap);
                VVSAdd (c, -0.5, region);
                for (j = 0; j < 4; j ++) {
                    mol[n].r = c;
                    if (j != 3) {
                        if (j != 0) mol[n].r.x += 0.5 * gap.x;
                        if (j != 1) mol[n].r.y += 0.5 * gap.y;
                        if (j != 2) mol[n].r.z += 0.5 * gap.z;}
                    ++ n;}}}}}


real gasdev (){
    static int available = 0;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available){
        do {
            v1 = 2.0 * rand() / (RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / (RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1 || rsq == 0.0);
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = 1;
        return v2 * fac;
    } else {
        available = 0;
        return gset;}}


void InitVels (){
    for (int n=0; n<nMol; n++){
        mol[n].rv.x = gasdev();
        mol[n].rv.y = gasdev();
        mol[n].rv.z = gasdev();}
    double vCM[3] = {0, 0, 0};
    for (int n=0; n<nMol; n++){
        vCM[0] += mol[n].rv.x;
        vCM[1] += mol[n].rv.y;
        vCM[2] += mol[n].rv.z;}
    for (int i=0; i<3; i++){
        vCM[i] /= nMol;}
    for (int n=0; n<nMol; n++){
        mol[n].rv.x -= vCM[0];
        mol[n].rv.y -= vCM[1];
        mol[n].rv.z -= vCM[2];}
    double vSqSum = 0;
    for (int n=0; n<nMol; n++){
        vSqSum += Sqr(mol[n].rv.x) + Sqr(mol[n].rv.y) + Sqr(mol[n].rv.z);}
    double lambda = sqrt(3 * (nMol-1) * temperature / vSqSum);
    for (int n=0; n<nMol; n++){
        mol[n].rv.x *= lambda;
        mol[n].rv.y *= lambda;
        mol[n].rv.z *= lambda;}}


void InitAccels (){
    int n;
    DO_MOL VZero (mol[n].ra);}


void SingleStep (){
    ++ stepCount;
    timeNow = stepCount * deltaT;
    LeapfrogStep (1);
    ApplyBoundaryCond ();
    if (method == 0){
        ComputeForcesAll ();}
    if (method == 1){
        ComputeForcesCell ();}
    if (method == 2){
        if (nebrNow) {
            nebrNow = 0;
            dispHi = 0.;
            BuildNebrList ();}
        ComputeForcesNbr ();}
    if (method == 3){
        if (nebrNow) {
            nebrNow = 0;
            dispHi = 0.;
            BuildNebrListCombo ();}
        ComputeForcesNbr ();}
    LeapfrogStep (2);
    EvalProps ();
    if ((stepCount < stepEquil) && (stepCount % stepAdjustTemp == 0)) AdjustTemp ();
    AccumProps (1);
    if (stepCount % stepAvg == 0) {
        AccumProps (2);
        PrintSummary ();
        AccumProps (0);}
    if (stepCount >= 0 &&
        (stepCount - 10) % stepVel == 0) EvalVelDist ();}


void LeapfrogStep (int part){
    int n;
    if (part == 1) {
        DO_MOL {
            VVSAdd (mol[n].rv, 0.5 * deltaT, mol[n].ra);
            VVSAdd (mol[n].r, deltaT, mol[n].rv);}}
    else {
        DO_MOL VVSAdd (mol[n].rv, 0.5 * deltaT, mol[n].ra);}}


void ApplyBoundaryCond (){
    int n;
    DO_MOL VWrapAll (mol[n].r);}


void ComputeForcesAll (){
    VecR dr;
    real fcVal, rr, rrCut, rri, rri3;
    int j1, j2, n;
    rrCut = Sqr (rCut);
    DO_MOL VZero (mol[n].ra);
    uSum = 0.;
    virSum = 0.;
    for (j1 = 0; j1 < nMol - 1; j1 ++) {
        for (j2 = j1 + 1; j2 < nMol; j2 ++) {
            VSub (dr, mol[j1].r, mol[j2].r);
            VWrapAll (dr);
            rr = VLenSq (dr);
            if (rr < rrCut) {
                rri = 1. / rr;
                rri3 = Cube (rri);
                fcVal = 48. * rri3 * (rri3 - 0.5) * rri;
                VVSAdd (mol[j1].ra, fcVal, dr);
                VVSAdd (mol[j2].ra, - fcVal, dr);
                uSum += 4. * rri3 * (rri3 - 1.);
                virSum += fcVal * rr;}}}}


void ComputeForcesCell (){
    VecR dr, invWid, rs, shift;
    VecI cc, m1v, m2v, vOff[] = OFFSET_VALS;
    real fcVal, rr, rrCut, rri, rri3, uVal;
    int c, j1, j2, m1, m1x, m1y, m1z, m2, n, offset;
    rrCut = Sqr (rCut);
    VDiv (invWid, cells, region);
    for (n = nMol; n < nMol + VProd (cells); n ++) cellList[n] = -1;
    DO_MOL {
        VSAdd (rs, mol[n].r, 0.5, region);
        VMul (cc, rs, invWid);
        c = VLinear (cc, cells) + nMol;
        cellList[n] = cellList[c];
        cellList[c] = n;}
    DO_MOL VZero (mol[n].ra);
    uSum = 0.;
    virSum = 0.;
    for (m1z = 0; m1z < cells.z; m1z ++) {
        for (m1y = 0; m1y < cells.y; m1y ++) {
            for (m1x = 0; m1x < cells.x; m1x ++) {
                VSet (m1v, m1x, m1y, m1z);
                m1 = VLinear (m1v, cells) + nMol;
                for (offset = 0; offset < N_OFFSET; offset ++) {
                    VAdd (m2v, m1v, vOff[offset]);
                    VZero (shift);
                    VCellWrapAll ();
                    m2 = VLinear (m2v, cells) + nMol;
                    DO_CELL (j1, m1) {
                        DO_CELL (j2, m2) {
                            if (m1 != m2 || j2 < j1) {
                                VSub (dr, mol[j1].r, mol[j2].r);
                                VVSub (dr, shift);
                                rr = VLenSq (dr);
                                if (rr < rrCut) {
                                    rri = 1. / rr;
                                    rri3 = Cube (rri);
                                    fcVal = 48. * rri3 * (rri3 - 0.5) * rri;
                                    uVal = 4. * rri3 * (rri3 - 1.);
                                    VVSAdd (mol[j1].ra, fcVal, dr);
                                    VVSAdd (mol[j2].ra, - fcVal, dr);
                                    uSum += uVal;
                                    virSum += fcVal * rr;}}}}}}}}}


void ComputeForcesNbr (){
    VecR dr;
    real fcVal, rr, rrCut, rri, rri3, uVal;
    int j1, j2, n;
    rrCut = Sqr (rCut);
    DO_MOL VZero (mol[n].ra);
    uSum = 0.;
    virSum = 0.;
    for (n = 0; n < nebrTabLen; n ++) {
        j1 = nebrTab[2 * n];
        j2 = nebrTab[2 * n + 1];
        VSub (dr, mol[j1].r, mol[j2].r);
        VWrapAll (dr);
        rr = VLenSq (dr);
        if (rr < rrCut) {
            rri = 1. / rr;
            rri3 = Cube (rri);
            fcVal = 48. * rri3 * (rri3 - 0.5) * rri;
            uVal = 4. * rri3 * (rri3 - 1.);
            VVSAdd (mol[j1].ra, fcVal, dr);
            VVSAdd (mol[j2].ra, - fcVal, dr);
            uSum += uVal;
            virSum += fcVal * rr;}}}


void BuildNebrListCombo (){
    VecR dr, invWid, rs, shift;
    VecI cc, m1v, m2v, vOff[] = OFFSET_VALS;
    real rrNebr;
    int c, j1, j2, m1, m1x, m1y, m1z, m2, n, offset;
    rrNebr = Sqr (rCut + rNebrShell);
    VDiv (invWid, cells, region);
    for (n = nMol; n < nMol + VProd (cells); n ++) cellList[n] = -1;
    DO_MOL {
        VSAdd (rs, mol[n].r, 0.5, region);
        VMul (cc, rs, invWid);
        c = VLinear (cc, cells) + nMol;
        cellList[n] = cellList[c];
        cellList[c] = n;}
    nebrTabLen = 0;
    for (m1z = 0; m1z < cells.z; m1z ++) {
        for (m1y = 0; m1y < cells.y; m1y ++) {
            for (m1x = 0; m1x < cells.x; m1x ++) {
                VSet (m1v, m1x, m1y, m1z);
                m1 = VLinear (m1v, cells) + nMol;
                for (offset = 0; offset < N_OFFSET; offset ++) {
                    VAdd (m2v, m1v, vOff[offset]);
                    VZero (shift);
                    VCellWrapAll ();
                    m2 = VLinear (m2v, cells) + nMol;
                    DO_CELL (j1, m1) {
                    DO_CELL (j2, m2) {
                    if (m1 != m2 || j2 < j1) {
                        VSub (dr, mol[j1].r, mol[j2].r);
                        VVSub (dr, shift);
                        if (VLenSq (dr) < rrNebr) {
                            if (nebrTabLen >= nebrTabMax)
                                ErrExit (ERR_TOO_MANY_NEBRS);
                            nebrTab[2 * nebrTabLen] = j1;
                            nebrTab[2 * nebrTabLen + 1] = j2;
                            ++ nebrTabLen;}}}}}}}}}


void BuildNebrList (){
    VecR dr;
    real rrNebr, rr;
    int j1, j2;
    rrNebr = Sqr (rCut+rNebrShell);
    nebrTabLen = 0;
    for (j1 = 0; j1 < nMol-1; j1++){
        for (j2 = j1+1; j2 < nMol; j2++){
            VSub(dr, mol[j1].r, mol[j2].r);
            VWrapAll(dr);
            rr = VLenSq (dr);
            if (rr < rrNebr){
                if (nebrTabLen >= nebrTabMax){
                    ErrExit(ERR_TOO_MANY_NEBRS);}
                nebrTab[2*nebrTabLen] = j1;
                nebrTab[2*nebrTabLen+1] = j2;
                ++nebrTabLen;}}}}


void EvalProps (){
    real vv, vvMax;
    int n;
    VZero (vSum);
    vvSum = 0.;
    vvMax = 0.;
    DO_MOL {
        VVAdd (vSum, mol[n].rv);
        vv = VLenSq (mol[n].rv);
        vvSum += vv;
        if (method == 2 || method == 3){
            vvMax = Max (vvMax, vv);}}
    kinEnergy.val = 0.5 * vvSum / nMol;
    totEnergy.val = kinEnergy.val + uSum / nMol;
    pressure.val = density * (vvSum + virSum) / (nMol * NDIM);
    if (method == 2 || method == 3){
        dispHi += sqrt (vvMax) * deltaT;
        if (dispHi > 0.5 * rNebrShell) nebrNow = 1;}}


void AccumProps (int icode){
    if (icode == 0) {
        PropZero (totEnergy);
        PropZero (kinEnergy);
        PropZero (pressure);
    } else if (icode == 1) {
        PropAccum (totEnergy);
        PropAccum (kinEnergy);
        PropAccum (pressure);
    } else if (icode == 2) {
        PropAvg (totEnergy, stepAvg);
        PropAvg (kinEnergy, stepAvg);
        PropAvg (pressure, stepAvg);}}


void EvalVelDist (){
    real deltaV, histSum;
    int j, n;
    if (countVel == 0) {
        for (j = 0; j < sizeHistVel; j ++) histVel[j] = 0.;}
    deltaV = rangeVel / sizeHistVel;
    DO_MOL {
        j = VLen (mol[n].rv) / deltaV;
        ++ histVel[Min (j, sizeHistVel - 1)];}
    ++ countVel;
    if (countVel == limitVel) {
        histSum = 0.;
        for (j = 0; j < sizeHistVel; j ++) histSum += histVel[j];
        for (j = 0; j < sizeHistVel; j ++) histVel[j] /= histSum;
        PrintVelDist ();
        countVel = 0;
        hFunction = 0.;
        for (j = 0; j < sizeHistVel; j ++) {
            if (histVel[j] > 0.) hFunction += histVel[j] * log (histVel[j] /
                ((j + 0.5) * deltaV));}}}


void ErrExit (int code){
    printf ("Error: %s\n", errorMsg[code]);
    exit (0);}


void PrintVelDist (){
    real vBin;
    int n;
    if (printout == 1){
        printf ("vdist (%.3f)\n", timeNow);}
    for (n = 0; n < sizeHistVel; n ++) {
        vBin = (n + 0.5) * rangeVel / sizeHistVel;
        fprintf (file, "%8.3f %8.3f\n", vBin, histVel[n]);}
    fprintf (file, "hfun: %8.3f %8.3f\n", timeNow, hFunction);}


void PrintSummary (){
    fprintf (file, "Summary:");
    fprintf (file,
    "%5d %8.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
    stepCount, timeNow, VCSum (vSum) / nMol, PropEst (totEnergy),
    PropEst (kinEnergy), PropEst (pressure), uSum, vvSum);}


static PyMethodDef myMethods[] = {
    { "MD3D", MD3D, METH_VARARGS, "Solves Molecular Dynamics 3D" },
    { NULL, NULL, 0, NULL }};


static struct PyModuleDef MolecDy = {
    PyModuleDef_HEAD_INIT,
    "MolecDy",
    "Dynamics Simulation",
    -1,
    myMethods};


PyMODINIT_FUNC PyInit_MD3D(void){
    return PyModule_Create(&MolecDy);}
