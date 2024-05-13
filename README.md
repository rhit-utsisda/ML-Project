# March Madness Predictions

## About
This repository contains a `Project` folder that contains various Machine Learning Algorithms used to make predictions on the NCAA Division I Men's Basketball March Madness Tournament. There are a total of 6 code files: <br>
1. `AllRegressions.ipynb` <br>
This file performs a grid search with cross-validation to tune hyperparameters and run regressions for Decision Trees, Random Forests, and Gradient Boosted Trees to predict wins for each tournament team in 2024.
2. `DtreewTrap.ipynb` <br>
This file uses a Decision Tree and a Random Forest Regressor with the Trapezoid of Excellence feature to predict the number of wins. The Trapezoid of Excellence feature was proposed in the following: [https://www.reddit.com/r/CollegeBasketball/comments/1bhx541/trapezoid_of_excellence_march_madness_2024/](url).
3. `LinRegWins.ipynb` <br>
This file contains a Linear Regression aimed toward predicting wins for each tournament team in 2024 along with some observations of the results.
4. `RForest_Ranks.ipynb` <br>
This file uses a random forest regressor to predict team wins in order to create a ranking system that ranks the performance of teams based on how close the model fits the target number of wins they acquired in the tournament.
5. `rfrgbt.ipynb` <br>
This file performs feature importance analysis using unengineered features (excludes advanced metrics) with the use of a Gradient Boosted Tree and a bootstrapped Random Forest Regressor.
6. `StatBasedFeatureEngSelectionLinearization.ipynb` <br>
The file performs feature selection for all 2nd degree terms for the `/Project/Datasets/Ken Battorvik.csv` dataset

There is also one dataset in the main `Project` directory, while the rest exist in the `/Project/Datasets` folder. The base datasets came from [https://www.kaggle.com/datasets/nishaanamin/march-madness-data](url).
