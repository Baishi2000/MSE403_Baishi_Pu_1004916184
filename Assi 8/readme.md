### NOTE: for feature selection, LassoCV was used. But LassoCV is unlikely to be the best choice as it performs regression. Therefore, I wrote a custom function (in CustomFunctions.py) which select **n** most important features by LogisticRegressionCV. Change to this function require that everything after feature selection be re-done so it is not actually used in the assignment.
### As for standardization of X, I standardized over the entire X only when: 1. searching hyper-parameter, and 2. training a full data model



**Instructions for running code:** Please download all files into the same folder in order to run the code successfully. As the featurizes dataframes are saved as excel, it should take less than 1 minute to run

  glass.data: training data file

  CustomFunctions.py: has all the custom functions needed

  Assi 8.ipynb: main document which gives the desired output

  dff: featureized df from the training data

  dff2: featurized df of the Co-Ti-Cr-Zr system for prediction
