%load data
load('AchronarakisData.mat');
rng default
NumTrees = 100;

%BAGGING
B = fitcensemble(X,classes,'Method','Bag','NumLearningCycles',NumTrees,'Kfold',10)
accuracy_cv = 1 - kfoldLoss(B)

%RANDOM FOREST (Min_Leaf=5)
RF = TreeBagger(NumTrees,X,classes,'Method','Classification','NumPredictorsToSample','all','MinLeafSize',5 ,'OOBPrediction','on');
out_of_bag_error = oobError(RF)