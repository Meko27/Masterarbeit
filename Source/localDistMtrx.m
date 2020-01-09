function localDist = localDistMtrx(x,k,metric,ground_dist)
% localDistMtrx - Calculates local Distance Matrix based on k and metric
%
% Syntax: localDist = localDistMtrx(x,k,metric)
%
% Inputs:
%   x           - Data samples (Dimension n x d)
%   k           - Number of nearest neighbors  
%   metric      - Underlying Metric to compute the distances
%   ground_dist - Ground distance 
%
% Outputs:
%   localDist  - Local Distance Matrix
%
if nargin < 3
    metric = 'euclidean';
end
if nargin < 4
    ground_dist = 0;
end 

n = size(x,1);
ixx = repmat(1:n,k,1); % column indices
if strcmp(metric,'emd')
    dist = squareform(calc_emd(x,ground_dist));
else 
    dist = squareform(pdist(x,metric));
end 
[knnDistMat,knnMat] = mink(dist,k+1);
knnMat(1,:)=[];
knnDistMat(1,:)=[];
localDist = sparse(ixx,double(knnMat),knnDistMat);
padNumber = n - max(max(knnMat));
localDist = padarray(localDist,[0 padNumber],0,'post');
localDist = sparse(localDist); 
localDist = max(localDist,localDist'); 

    function dist = calc_emd(x,ground_dist)
        [n_samples,n_dim] = size(x);
        %threshold = sqrt(n_dim);
        % Calculate Costmatrix
        %idx_vect = 1:n_dim;
        %idx_vect(idx_vect>threshold) = 0; % Set entries larger thres to zero
        %cost_mtrx = squareform(pdist(idx_vect'));
        emd_vect = zeros(1,n_samples*(n_samples-1)/2);
        size_emd_vect = size(emd_vect,2);
        idx=0;
        % Iterate over lower triangle of Matrix as pdist()-built in function
        for i=1:n_samples-1
                start = i+1;
            for j=start:n_samples
                idx = idx+1;
                emd_vect(idx) = emd_mex(x(i,:),x(j,:),ground_dist);
                %emd_vect(idx) = emd_hat_gd_metric_mex(x(i,:)',x(j,:)',cost_mtrx);
                fprintf(1,'Now calculating %i of %i\n',idx,size_emd_vect);
            end 
        end
        dist = emd_vect;
    end

end 