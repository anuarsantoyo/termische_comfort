# Thermal Comfort Model
Goal: Test several models that predict thermal comfort based on Feature vector F = [Outside Temperature, Avg. Temperature, RH, va, Tg, Gender, Age, Height, Weight, Duration, Icl]

# Approaches:
## Predictor: F -> [-3, 3] 
- Linear (implemented)
- NN (implemented)
- Fanger
- Hybrid: From Fanger calculate W using NN or linear model
- Logistic

## Clasifier: F -> {-1, 0, 1}
- NN (implemented)
- Random Forest
- Adaboost
- SVM (implemented)

## Cluster:
Cluster data in k groups and see if groups found correspond to label, if so describe groups found and use them to label
new data
- k means
- nnmf
- Gaussian mixture model

## Mean Votation Polyfit:
Use rolling discrete hypercube to calculate mean comfort (MC) within hypercube on discrete points throughout feature 
space V. Fit polynomial (or kernel approach) so that it predicts F -> MC 
 todo:
- ver paper relacionados a buses
- 