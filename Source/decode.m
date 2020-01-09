function xDecoded = decode(xEncoded,xTrainEncoded,LArray,l_idx)
% decode - Decodes denoised (encoded) Data based on the denoising model
% Based on a denoised training set and the Laplacian Matrices of all 
% iterations of the backw. Diff. process, the process is reveresed
% 
% Syntax: [xDecoded]
%          = decode(xEncoded, xTrainEncoded, LArray, l_idx) 
% 
% Inputs:
%  xEncoded         - Denoised (encoded) records
%  xTrainEncoded    - Encoded train data set
%  LArray           - Array of Laplacian Matrices of all iterations of
%                     training set
%  l_idx            - Indices of nearest Sample in training set to records
%   
% Outputs:
%  xDecoded         - Decoded records
%

n = size(xTrainEncoded,1);
iter = length(LArray);
xTrainEncoded(l_idx,:) = xEncoded;
for i = 1:iter 
    D = spdiags(diag(LArray{iter-i+1}),0,n,n);
    xTrainEncoded = D\(D+0.25*LArray{iter-i+1})*xTrainEncoded; % Pseudo forward Markov
end 
xDecoded = xTrainEncoded(l_idx,:);

end