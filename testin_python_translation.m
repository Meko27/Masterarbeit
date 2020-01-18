% test localDist

x = [2 8 8;
 9 3 5;
 9 7 1;
 5 2 4;
 4 2 1;
 9 2 9;
 7 4 9;
 6 7 5;
 8 7 1;
 8 7 6];
k = 3;
dist = localDistMtrx(x(1:5,:),k);
L = weightedGraphLaplacian(dist);
ground_dist = [0 1 2;1 0 1;2 1 0];

%%
tic
metric = 'emd';
k = 3; iter = 10; n_outliers = 2;
Ad_detector = Ano_detector(x(1:5,:),k,iter,metric,ground_dist);
[distvect_ano,idx_outliers, Ad_obj] = Ad_detector.calc_outliers(n_outliers);
toc