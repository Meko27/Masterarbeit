function L = weightedGraphLaplacian(localDist)
% weightedGraphLaplacian - Calculates the weighted Graph Laplacian 
%
% Syntax: L = weightedGraphLaplacian(localDist)
%
% Inputs:
%   localDist   - Local Distance Matrix
%
% Outputs:
%   L           - Weighted Graph Laplacian
%

n = size(localDist,1);

% Degree Matrix
d = sum(localDist,2);
if(sum(d==0)>0)
    d(d==0)=1;
end
D = spdiags(1./(d), 0, n, n);

% Weighted Adjacency Matrix
A = 1/(n)*D*localDist*D;
clear d;
clear D;

% Degree Matrix of weighted A
d = sum(A,2);
if(sum(d==0)>0)
    d(d==0)=1/n;
end 
D = spdiags(d,0,n,n);

% Graph Laplacian 
L = D-A;  
end 