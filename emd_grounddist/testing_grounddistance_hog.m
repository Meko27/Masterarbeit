%% Ground Distance for EMD on HOG

% Input image
n_row = 24;
n_col = 24;
I = uint8(randi(255,n_row,n_col));
%I = img1_resize;
%%
% Parameters
I_size = size(I);
I_size = I_size(1:2);
NumBins = 9;
CellSize = [8 8];
BlockSize = [2 2];
BlockOverlap = BlockSize/2;

% Hog size
BlocksPerImage = floor((I_size./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1);
n_hog = prod([BlocksPerImage, BlockSize, NumBins]);

[hog] = extractHOGFeatures(I,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap);
                    
%% Calculate Distance Matrix

% Get indices for spatial distance
n_cell = I_size./CellSize;
n_cell_row = n_cell(1); n_cell_col = n_cell(2);

current_el = [1,1];
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

%% HOG distance matrix
hog_segment_dist = xlsread('/Users/meko/Documents/Masterarbeit/AnomalyDetection/GroundDist.xlsx','E5:M13');

hog_dist = repmat(hog_segment_dist,n_hog/NumBins);
spatial_dist_mat_augmented = repelem(spatial_dist_mat,NumBins,NumBins);

ground_dist = hog_dist + spatial_dist_mat_augmented;
ground_dist2 = HOG_ground_distance(I,NumBins,CellSize,BlockSize,BlockOverlap,0,1);

