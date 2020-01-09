function [ground_dist] = HOG_ground_distance(img,orientations,cell_size,block_size,block_stride,signed,rotation_cost,move_cost,threshold)
% HOG_ground_distance - Calculates the grounddistance of a HOG feature
%                       descriptor
% Inputs:
%   img             -   Image of which the HOG descriptor was calculated
%   orientations    -   Number of orientation bins used to calculate the HOG 
%   cell_size       -   Cell size for calculating the HOG
%   block_size      -   Number of Cells in a block
%   block_stride    -   Stride of blocks. Controls how much adjacent    
%                       blocks overlap.
%   signed          -   Defines wether signed or unsigned angles were used
%                       to calculate the HOG
%   rotation_cost   -   Cost for rotating one bin by one unit 
%   move_cost       -   Cost for moving by one cell
%   threshold       -   Thresholds the maximum spatial distance
% Outputs:
%   ground_distance -   Ground Distance of the HOG feature descriptor
%

if nargin < 2
    orientations = 9;
end
if nargin < 3
    cell_size = 8;
end
if nargin < 4
    block_size = [2 2];
end 
if nargin < 5
    block_stride = block_size/2;
end 
if nargin < 6
    signed = 0;
end 
if nargin < 7
    rotation_cost = 1;
end 
if nargin < 8
    move_cost = 1; 
end
if nargin < 9
    threshold = 0;
end 

img_size = [size(img,1) size(img,2)];
blocks_per_image = floor((img_size./cell_size - block_size)./(block_size - block_stride) + 1);
n_hog_bins = prod([blocks_per_image, block_size, orientations]);

% spatial distance matrix
spatial_dist_mat = spatial_dist(n_hog_bins,orientations,block_size,blocks_per_image);
if threshold ~= 0
    spatial_dist_mat(spatial_dist_mat > threshold) = threshold;
end
spatial_dist_mat = repelem(spatial_dist_mat,orientations,orientations);

% Orientations distance matrix
orient_dist_cell = rotation_dist(orientations,signed);
orient_dist_mat = repmat(orient_dist_cell,n_hog_bins/orientations);

% Total ground distance matrix
ground_dist = rotation_cost * rotation_cost*orient_dist_mat + move_cost * move_cost*spatial_dist_mat;

end 

