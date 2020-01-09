%% function to calculate the grounddistance of HOG features for emd
function [ground_dist] = calculateGroundDist(img,NumBins,CellSize,BlockSize,BlockOverlap,threshold)

if nargin < 2
    NumBins = 9;
end
if nargin < 3
    CellSize = 8;
end
if nargin < 4
    BlockSize = [2 2];
end 
if nargin < 5
    BlockOverlap = BlockSize/2;
end 
if nargin <6
    threshold = 0; % default: no thresholding
end 
img_size = [size(img,1) size(img,2)];
BlocksPerImage = floor((img_size./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1);
n_hog = prod([BlocksPerImage, BlockSize, NumBins]);


%% Calculate Distance Matrix

% indices for spatial distance

cell_i_vect = zeros(n_hog/NumBins,1);
cell_j_vect = cell_i_vect;
idx = 1;
for b_i=1:BlocksPerImage(1)
    for b_j=1:BlocksPerImage(2)
        for cb_j=1:BlockSize(2)
            for cb_i=1:BlockSize(1)
                if b_j==1
                    cell_j = cb_j;
                else 
                    cell_j = cb_j + (b_j - 1);
                end
                if b_i==1
                    cell_i = cb_i;
                else
                    cell_i = cb_i + (b_i - 1);
                end 
                cell_i_vect(idx) = cell_i;
                cell_j_vect(idx) = cell_j;
                idx = idx+1;
            end
        end
    end
end 

% Calculate spatial distance matrix
cell_idx_pair_vect = [cell_i_vect cell_j_vect];
spatial_dist_mat = squareform(pdist(cell_idx_pair_vect,'cityblock'));
if threshold ~= 0
    spatial_dist_mat(spatial_dist_mat > threshold) = threshold;
end 
%% HOG distance matrix
hog_segment_dist = xlsread('/Users/meko/Documents/Masterarbeit/AnomalyDetection/GroundDist.xlsx','E5:M13');

hog_dist = repmat(hog_segment_dist,n_hog/NumBins);
spatial_dist_mat_augmented = repelem(spatial_dist_mat,NumBins,NumBins);

ground_dist = hog_dist + spatial_dist_mat_augmented;

end 

