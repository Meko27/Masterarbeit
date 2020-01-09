function [xEncoded,l_idx] = encode(x, xTrain, LArray, metric) 
% encode - Denoises (encodes) records based on the Diffusion model.
% Based on a training set and the Laplacian Matrices of all iterations of
% the backward Diffusion process records x are being denoised.
%
% Syntax: [xEncoded, l_idx]
%          = encode(x, xTrain, LArray, metric) 
%
% Inputs:
%  x              - 'Noisy' Records (Dimension nX x d)
%  xTrain         - Training set (Dimension nTrain x d)
%  LArray         - Array of Laplacian Matrices of all iterations of
%                   training set
%  metric         - String which specifies underlying metric for calculating 
%                   the distances (Default: euclidean)  
%
% Outputs:
%  xEncoded       - Denoised (encoded) records
%  l_idx          - Indices of nearest Sample in training set to records
%

if nargin < 4
    metric = 'euclidean';
end 
nTrain = size(xTrain,1);
nX = size(x,1);
iter = size(LArray,2);

%% Find nearest Neigbors of each record of x in xTrain

l = 1; % Number of nearest neighbours 
l_idx = zeros(1,nX); 
k_check = uint8(sqrt(nTrain)); % parameter for checking model on the record 

% compute distances
for i=1:nX
    distTotal = squareform(pdist([xTrain;x(i,:)],metric));
    dist = distTotal(:,end);
    [dist,idx] = mink(dist,l+1);
    if dist(1)==dist(2)
        [~,j]=max(idx);
        dist(j,:) = [];
        idx(j,:) = [];
    else 
    dist(1,:) = []; 
    idx(1,:) = []; 
    end 
    l_idx(i) = idx;
    % Check if Model is able to denoise Test Point
    if dist > max(mink(distTotal(:,l_idx(i)),k_check))
        fprintf('Error. Testpoint of Index %i cannot be denoised based on underlying model.\n',i)
    end 
end 

%% Iterative backward Diffusion process

xTrain(l_idx,:) = x; 

for i = 1:iter 
    D = spdiags(diag(LArray{i}),0,nTrain,nTrain);
    xTrain=(D+0.25*LArray{i})\(D*xTrain); % Solve inverse Pseudo Markov
    xEncoded{i} = xTrain; 
end 

%xEncoded = xTrain;

end 