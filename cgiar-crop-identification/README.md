# See-it-Grow Zindi Challenge

This dataset contains a set of training and testing data for the
See-it-Grow Zindi challenge. Data is divided in a training part
(the `train` directory), for which labels are provided in `train.csv`
and a testing part (`test` directory) for which no labels are 
provided (and used for external evaluation / leaderboard).

The full database contains roughly ~60K images, however only the
annotated ones were withheld. In addition, those without a growth
stage label were also removed. Testing data is a ~25% hold-out on
this data-set across two years of crop growth for both small and
large rain seasons (SR, and LR respectivelly). The hold-out is grouped
by season and by growth_stage, damage type and extent. In total the 
training data provided consists of ~26K images. 

| Season   | Image Count   |
|----------|:-------------:|
| LR2020   | 2034          |
| LR2021   | 7979          |
| SR2020   | 6163          |
| SR2021   | 9930          |

Note that the training data will need to be divided in a true
training and validation set (according to the choices of the
candidate). Also consider that the data is zero-inflated with
many values showing no damage or low damage extent numbers,
or good growing conditions (G, see table below). Furthermore,
classes are unbalanced adjust your model training accordingly.

Evaluation will be done using F1 and Cohen's Kappa statistic,
depending on the task at hand.

## Ancillary data

For each growth stage the damage types and their `extent` 
are provided as percentage (%) in 10% increments.

### Growth Stages

| Growth Stage   | Definition       |
|----------------|:----------------:|
| F              | Flowering        |
| M              | Maturity         |
| S              | Sowing           |
| V              | Vegetative       |

### Damage types

| Damage          | Definition         |
|-----------------|:------------------:|
| DR              | Drought            |
| DS              | Disease            |
| FD              | Flood              |
| G               | Good (growth)      |
| ND              | Nutrient Deficient |
| PS              | Pest               |
| WD              | Weed               |
| WN              | Wind               |


