# Thermal Comfort Model
Goal: Test several models that predict thermal comfort based on Feature vector F = [Outside Temperature, Avg. Temperature, RH, va, Tg, Gender, Age, Height, Weight, Duration, Icl]

# Approaches:
## Predictor: F -> [-3, 3] 
- Linear
- NN
- Fanger
- Hybrid: From Fanger calculate W using NN or linear model
- Logistic

## Clasifier: F -> {-1, 0, 1}
- NN
- Random Forest
- Adaboost
- SVM

## Cluster: Cluster data in k groups and see if groups found correspond to labels, if so describe groups found and use them to label new data
- k means
- nnmf
- Gaussian mixture model

## Mean Votation Polyfit: Use rolling discrete hypercube to calculate mean comfort (MC) within hypercube on discrete points throughout feature space V. Fit polynome so that it predicts F -> MC 
