function spatial_dist_mat = spatial_dist(n_hog_bins,orientations,block_size,blocks_per_image)
% spatial_dist - calculates the spatial distances of an image's HOG 
%                    feature vector 
% Inputs:
%   n_hog_bins          - Number of HOG bins representing the image 
%   orientations        - Number of orientation bins used to calculate the HOG 
%   block_size          - Number of Cells in a block
%   blocks_per_image    - Number of total blocks 
% Outputs_
%   spatial_dist_mat    - Spatial Distance matrix of the HOG vector
%

cell_i_vect = zeros(n_hog_bins/orientations,1);
cell_j_vect = cell_i_vect;
idx = 1;
for b_i=1:blocks_per_image(1)
    for b_j=1:blocks_per_image(2)
        for cb_j=1:block_size(2)
            for cb_i=1:block_size(1)
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

cell_idx_pair_vect = [cell_i_vect cell_j_vect];
spatial_dist_mat = squareform(pdist(cell_idx_pair_vect,'cityblock'));
end