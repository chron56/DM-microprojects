%load data
%DATA SET #1
load('AchronarakisData.mat');

%DATA SET #2
%DS = readtable('Achronarakis_ClassData.txt');
%X = DS(:,[1:8]);
%classes = DS(:,[9]);

%KNN Classifier
%CVKNNMdl = crossval(KNNMdl);
%classAccuracyKNN = 1 - kfoldLoss(CVKNNMdl)

%NAIVE BAYES Classifier
%NBMdl = fitcnb(X,classes,'CrossVal','on'); %Default DistributionNames=Normal 
%classAccuracyNB = 1 - kfoldLoss(NBMdl)

%SVM Classifier
%LINEAR
%SVMLINEARMdl = fitcsvm(X,classes,'Standardize',true,'KernelFunction','linear');
%CVSVMLINEARMdl = crossval(SVMLINEARMdl);
%classAccuracySVMLINEAR = 1 - kfoldLoss(CVSVMLINEARMdl)

%RBF
%SVMRBFMdl = fitcsvm(X,classes,'Standardize',true,'KernelFunction','rbf','KernelScale',100);
%CVSVMRBFMdl = crossval(SVMRBFMdl);
%classAccuracySVMRBF = 1 - kfoldLoss(CVSVMRBFMdl)

%DECISION TREES Classifier
DTMdl = fitctree(X,classes,'MaxNumSplits',100,'MinLeafSize',100,'MinParentSize',10);
CVDTMdl = crossval(DTMdl);
classAccuracyDT = 1 - kfoldLoss(CVDTMdl);
view(DTMdl,'mode','graph');


