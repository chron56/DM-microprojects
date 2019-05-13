%load data
load('7clusters.mat');
NoC=8;

%K-MEANS
idkm = kmeans(X,NoC);
plot_max10_clusters(X,idkm)



%ESTIMATE THE NUMBER OF CLUSTERS 
evaNoC = evalclusters(X,'kmeans','silhouette','KList',[1:10])
