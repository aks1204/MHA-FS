
# Notification

**The biggest open-source Python library for feature selection has been successfully completed and is now available on 
GitHub at** https://github.com/thieu1995/mafese.

**MAFESE** is an improved and completed version of this module. It not only includes all meta-heuristic algorithms, but 
it also categorizes them into different feature selection methods. These methods include traditional techniques 
such as filter-based, wrapper-based, unsupervised-based, and embedded-based methods. Among them, 
metaheuristic-based feature selection belongs to the filter-based method.


# Feature Selection using Meta-Heuristic Algorithms (MEALPY)


# Usage

## Dependencies 

1. Mealpy (https://github.com/thieu1995/mealpy)
2. Permetrics (https://github.com/thieu1995/permetrics)

## Setup environments

### Pip 
```code 
pip install numpy, pandas, scikit-learn, matplotlib
pip install mealpy==2.5.0
pip install permetrics==1.3.0
```

### Conda 
```code 
conda create -n ml python==3.7.5
conda activate ml
conda install -c conda-forge numpy, pandas, scikit-learn, matplotlib
pip install mealpy==2.5.0
pip install permetrics==1.3.0
```


## Examples

Run the file: src/models/mha_fs.py 


Notes: 

```code

1. The format of solution / position 

- Because we need to select the best features in dataset right?

So a solution is a 1-D vector, each dimension represent an index of column in dataset. 
    + If it has value 1, meaning this column is selected for the model
    + If it has value 0, meaning this column is not selected for the model

2. The purposes of function: amend_position 

- Because the algorithm create a real-value solution, right?

So we need this function to convert real-value back into integer value (0 and 1). 
    + That is why the lower bound is 0 for all dimensions (floor of 0 is 0)
    + The upper bound is 1.99 for all dimensions (floor of 1.99 is 1)

Also if all the dimensions have value 0, meaning no column is selected --> Can't run the model 
    + So we need to select a random column for the model

3. Fitness function 

- Use the Evaluator object to build a classification model with the selected model (KNN, or Random forrest or SVM) 
and calculate the metrics such as accuracy, precision, recall and f1. 

- Set the obj_weights to 1 if you interested in selected metric 

- In the current example, accuracy metric is selected, so the weight for accuracy is 1 and other metrics are 0. 
Beside, this problem will be maximization problem (we want the maximum accuracy - 1)

4. Use different algorithm

- Replace the selected model in the section: ## 2. Define algorithm and trial

5. Setting

- Everything needs to config is located at: src/config.py 

```


![NoPrintAll](img/no_print_all.png)


![PrintAll](img/print_all.png)




## Call all models from MEALPY 

https://mealpy.readthedocs.io/en/latest/pages/general/guide_to_use_model.html#import-all-models

```code 

from mealpy.bio_based import BBO, EOA, IWO, SBO, SMA, TPO, VCS, WHO
from mealpy.evolutionary_based import CRO, DE, EP, ES, FPA, GA, MA
from mealpy.human_based import BRO, BSO, CA, CHIO, FBIO, GSKA, ICA, LCO, QSA, SARO, SSDO, TLO
from mealpy.math_based import AOA, CGO, GBO, HC, SCA
from mealpy.music_based import HS
from mealpy.physics_based import ArchOA, ASO, EFO, EO, HGSO, MVO, NRO, SA, TWO, WDO
from mealpy.probabilistic_based import CEM
from mealpy.system_based import AEO, GCO, WCA
from mealpy.swarm_based import ABC, ACOR, ALO, AO, BA, BeesA, BES, BFO, BSA, COA, CSA, CSO, DO, EHO, FA, FFA, FOA, GOA, GWO, HGS
from mealpy.swarm_based import HHO, JA, MFO, MRFO, MSA, NMRA, PFA, PSO, SFO, SHO, SLO, SRSR, SSA, SSO, SSpiderA, SSpiderO, WOA


model = BBO.OriginalBBO(problem, epoch=10, pop_size=50)
# model = BBO.BaseBBO(problem, epoch=10, pop_size=50)
# model = EOA.BaseEOA(problem, epoch=10, pop_size=50)
# model = IWO.OriginalIWO(problem, epoch=10, pop_size=50)
# model = SBO.BaseSBO(problem, epoch=10, pop_size=50)
# model = SBO.OriginalSBO(problem, epoch=10, pop_size=50)
# model = SMA.OriginalSMA(problem, epoch=10, pop_size=50)
# model = SMA.BaseSMA(problem, epoch=10, pop_size=50)
# model = VCS.OriginalVCS(problem, epoch=100, pop_size=50)
# model = VCS.BaseVCS(problem, epoch=10, pop_size=50)
# model = WHO.BaseWHO(problem, epoch=10, pop_size=50)

# model = CRO.BaseCRO(problem, epoch=10, pop_size=50)
# model = CRO.OCRO(problem, epoch=10, pop_size=50)
# model = DE.BaseDE(problem, epoch=10, pop_size=50)
# model = DE.JADE(problem, epoch=10, pop_size=50)
# model = DE.SADE(problem, epoch=10, pop_size=50)
# model = DE.SHADE(problem, epoch=10, pop_size=50)
# model = DE.L_SHADE(problem, epoch=10, pop_size=50)
# model = DE.SAP_DE(problem, epoch=10, pop_size=50)
# model = EP.BaseEP(problem, epoch=10, pop_size=50)
# model = EP.LevyEP(problem, epoch=10, pop_size=50)
# model = ES.BaseES(problem, epoch=10, pop_size=50)
# model = ES.LevyES(problem, epoch=10, pop_size=50)
# model = FPA.BaseFPA(problem, epoch=10, pop_size=50)
# model = GA.BaseGA(problem, epoch=10, pop_size=50)
# model = MA.BaseMA(problem, epoch=10, pop_size=50)

# model = BRO.OriginalBRO(problem, epoch=10, pop_size=50)
# model = BRO.BaseBRO(problem, epoch=10, pop_size=50, pr=0.03)
# model = BSO.BaseBSO(problem, epoch=10, pop_size=50)
# model = BSO.ImprovedBSO(problem, epoch=10, pop_size=50)
# model = CA.OriginalCA(problem, epoch=10, pop_size=50)
# model = CHIO.BaseCHIO(problem, epoch=10, pop_size=50)
# model = CHIO.OriginalCHIO(problem, epoch=10, pop_size=50, pr=0.03)
# model = GSKA.BaseGSKA(problem, epoch=10, pop_size=50)
# model = GSKA.OriginalGSKA(problem, epoch=10, pop_size=50)
# model = FBIO.OriginalFBIO(problem, epoch=10, pop_size=50)
# model = FBIO.BaseFBIO(problem, epoch=10, pop_size=50)
# model = ICA.BaseICA(problem, epoch=10, pop_size=50)
# model = LCO.BaseLCO(problem, epoch=10, pop_size=50)
# model = LCO.ImprovedLCO(problem, epoch=10, pop_size=50, pr=0.03)
# model = LCO.OriginalLCO(problem, epoch=10, pop_size=50)
# model = QSA.BaseQSA(problem, epoch=10, pop_size=50)
# model = QSA.OppoQSA(problem, epoch=10, pop_size=50)
# model = QSA.LevyQSA(problem, epoch=10, pop_size=50)
# model = QSA.ImprovedQSA(problem, epoch=10, pop_size=50, pr=0.03)
# model = QSA.OriginalQSA(problem, epoch=10, pop_size=50)
# model = SARO.BaseSARO(problem, epoch=10, pop_size=50)
# model = SARO.OriginalSARO(problem, epoch=10, pop_size=50)
# model = SSDO.BaseSSDO(problem, epoch=10, pop_size=50)
# model = TLO.BaseTLO(problem, epoch=10, pop_size=50)
# model = TLO.OriginalTLO(problem, epoch=10, pop_size=50)
# model = TLO.ITLO(problem, epoch=10, pop_size=50)

# model = AOA.OriginalAOA(problem, epoch=10, pop_size=50)
# model = CGO.OriginalCGO(problem, epoch=100, pop_size=50)
# model = GBO.OriginalGBO(problem, epoch=10, pop_size=50, pr=0.03)
# model = HC.OriginalHC(problem, epoch=10, pop_size=50)
# model = HC.BaseHC(problem, epoch=10, pop_size=50)
# model = SCA.OriginalSCA(problem, epoch=10, pop_size=50)
# model = SCA.BaseSCA(problem, epoch=10, pop_size=50)

# model = HS.OriginalHS(problem, epoch=10, pop_size=50)
# model = HS.BaseHS(problem, epoch=10, pop_size=50)

# model = ArchOA.OriginalArchOA(problem, epoch=10, pop_size=50)
# model = ASO.BaseASO(problem, epoch=10, pop_size=50)
# model = EFO.OriginalEFO(problem, epoch=10, pop_size=50)
# model = EFO.BaseEFO(problem, epoch=10, pop_size=50)
# model = EO.BaseEO(problem, epoch=10, pop_size=50)
# model = EO.ModifiedEO(problem, epoch=10, pop_size=50)
# model = EO.AdaptiveEO(problem, epoch=10, pop_size=50)
# model = HGSO.BaseHGSO(problem, epoch=10, pop_size=50)
# model = MVO.OriginalMVO(problem, epoch=10, pop_size=50)
# model = MVO.BaseMVO(problem, epoch=10, pop_size=50)
# model = NRO.BaseNRO(problem, epoch=10, pop_size=50)
# model = SA.BaseSA(problem, epoch=10, pop_size=50)
# model = TWO.BaseTWO(problem, epoch=10, pop_size=50)
# model = TWO.OppoTWO(problem, epoch=10, pop_size=50)
# model = TWO.LevyTWO(problem, epoch=10, pop_size=50)
# model = TWO.EnhancedTWO(problem, epoch=10, pop_size=50)
# model = WDO.BaseWDO(problem, epoch=10, pop_size=50)

# model = CEM.BaseCEM(problem, epoch=10, pop_size=50)

# model = AEO.OriginalAEO(problem, epoch=10, pop_size=50)
# model = AEO.ModifiedAEO(problem, epoch=10, pop_size=50)
# model = AEO.AdaptiveAEO(problem, epoch=10, pop_size=50)
# model = AEO.EnhancedAEO(problem, epoch=10, pop_size=50)
# model = AEO.IAEO(problem, epoch=10, pop_size=50)
# model = GCO.OriginalGCO(problem, epoch=10, pop_size=50)
# model = GCO.BaseGCO(problem, epoch=10, pop_size=50)
# model = WCA.BaseWCA(problem, epoch=10, pop_size=50)

# model = ABC.BaseABC(problem, epoch=10, pop_size=50)
# model = ACOR.BaseACOR(problem, epoch=10, pop_size=50)
# model = ALO.OriginalALO(problem, epoch=10, pop_size=50)
# model = ALO.BaseALO(problem, epoch=10, pop_size=50)
# model = AO.OriginalAO(problem, epoch=10, pop_size=50)
# model = BA.BaseBA(problem, epoch=10, pop_size=50)
# model = BA.OriginalBA(problem, epoch=10, pop_size=50)
# model = BA.ModifiedBA(problem, epoch=10, pop_size=50)
# model = BeesA.BaseBeesA(problem, epoch=10, pop_size=50)
# model = BeesA.ProbBeesA(problem, epoch=10, pop_size=50)
# model = BES.BaseBES(problem, epoch=10, pop_size=50)
# model = BFO.OriginalBFO(problem, epoch=10, pop_size=50)
# model = BFO.ABFO(problem, epoch=10, pop_size=50)
# model = BSA.BaseBSA(problem, epoch=10, pop_size=50)
# model = COA.BaseCOA(problem, epoch=10, pop_size=50)
# model = CSA.BaseCSA(problem, epoch=10, pop_size=50)
# model = CSO.BaseCSO(problem, epoch=10, pop_size=50)
# model = DO.BaseDO(problem, epoch=10, pop_size=50)
# model = EHO.BaseEHO(problem, epoch=10, pop_size=50)
# model = FA.BaseFA(problem, epoch=10, pop_size=50)
# model = FFA.BaseFFA(problem, epoch=10, pop_size=50)
# model = FOA.OriginalFOA(problem, epoch=10, pop_size=50)
# model = FOA.BaseFOA(problem, epoch=10, pop_size=50)
# model = FOA.WhaleFOA(problem, epoch=10, pop_size=50)
# model = GOA.BaseGOA(problem, epoch=10, pop_size=50)
# model = GWO.BaseGWO(problem, epoch=10, pop_size=50)
# model = GWO.RW_GWO(problem, epoch=10, pop_size=50)
# model = HGS.OriginalHGS(problem, epoch=10, pop_size=50)
# model = HHO.BaseHHO(problem, epoch=10, pop_size=50)
# model = JA.OriginalJA(problem, epoch=10, pop_size=50)
# model = JA.BaseJA(problem, epoch=10, pop_size=50)
# model = JA.LevyJA(problem, epoch=10, pop_size=50)
# model = MFO.OriginalMFO(problem, epoch=10, pop_size=50)
# model = MFO.BaseMFO(problem, epoch=10, pop_size=50)
# model = MRFO.BaseMRFO(problem, epoch=10, pop_size=50)
# model = MSA.BaseMSA(problem, epoch=10, pop_size=50)
# model = NMRA.BaseNMRA(problem, epoch=10, pop_size=50)
# model = NMRA.ImprovedNMRA(problem, epoch=10, pop_size=50)
# model = PFA.BasePFA(problem, epoch=10, pop_size=50)
# model = PSO.BasePSO(problem, epoch=10, pop_size=50)
# model = PSO.C_PSO(problem, epoch=10, pop_size=50)
# model = PSO.CL_PSO(problem, epoch=10, pop_size=50)
# model = PSO.PPSO(problem, epoch=10, pop_size=50)
# model = PSO.HPSO_TVAC(problem, epoch=10, pop_size=50)
# model = SFO.BaseSFO(problem, epoch=10, pop_size=50)
# model = SFO.ImprovedSFO(problem, epoch=10, pop_size=50)
# model = SHO.BaseSHO(problem, epoch=10, pop_size=50)
# model = SLO.BaseSLO(problem, epoch=10, pop_size=50)
# model = SLO.ModifiedSLO(problem, epoch=10, pop_size=50)
# model = SLO.ISLO(problem, epoch=10, pop_size=50)
# model = SRSR.BaseSRSR(problem, epoch=10, pop_size=50)
# model = SSA.OriginalSSA(problem, epoch=10, pop_size=50)
# model = SSA.BaseSSA(problem, epoch=10, pop_size=50)
# model = SSO.BaseSSO(problem, epoch=10, pop_size=50)
# model = SSpiderA.BaseSSpiderA(problem, epoch=10, pop_size=50)
# model = SSpiderO.BaseSSpiderO(problem, epoch=10, pop_size=50)
# model = WOA.BaseWOA(problem, epoch=10, pop_size=50)
# model = WOA.HI_WOA(problem, epoch=10, pop_size=50)

```

## Citations

* If this code is useful for you, please give me some credits, [link](https://gist.github.com/thieu1995/2dcebc754bf0038d0c12b26ec9d591aa) to my first-author papers.

```code 
@software{thieu_nguyen_2020_3711949,
  author       = {Nguyen Van Thieu},
  title        = {A collection of the state-of-the-art MEta-heuristics ALgorithms in PYthon: Mealpy},
  month        = march,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3711948},
  url          = {https://doi.org/10.5281/zenodo.3711948}
}
```
