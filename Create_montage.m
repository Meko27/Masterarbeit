%% Create image montage

%% Load data
path = '/Users/meko/Desktop/Hole_images';
path_normal = '/good';
path_ano = '/ano';
path_ano_unmarked = '/unmarked';

%% Loading images
[img_normal,~] = loadImg(strcat(path,path_normal)); 
[img_anomal,~] = loadImg(strcat(path,path_ano)); 
[img_anomal_unmarked,~] = loadImg(strcat(path,path_ano_unmarked)); 


%% Save images into Vector
img_all_cell = [img_normal img_anomal]';
img_all_cell_u = [img_normal img_anomal_unmarked]';
n_images = length(img_all_cell);
random_idx = randperm(n_images);
% Caltulate indexes of random anomalous images
for i=1:10
    random_outlier_idx(i) = find(random_idx==idx_outliers(i));
end 


for i=1:n_images
    %img_all(:,:,:,random_idx(i)) = img_all_cell{i};
    img_all_u(:,:,:,i) = img_all_cell_u{i};
end 

%% Display 10 anomalous images
for i=1:10
    img_ano_vect(:,:,:,i) = img_all_u(:,:,:,idx_outlies(i));
end
figure
montage(img_ano_vect,'BorderSize',[10 10])

%% Display images as montage
n_columns = 9;
n_rows = 4; 
%figure(1)
%thumbnail_size = [256 256];
%montage(img_all(:,:,:,1:9*4),'size',[n_rows n_columns],'ThumbnailSize',thumbnail_size,'BorderSize',[1 1]);
figure
montage(img_all_u(:,:,:,9*4*7+1:9*4*8),'size',[n_rows n_columns],'ThumbnailSize',thumbnail_size,'BorderSize',[1 1]);
%%
img_ano_all_cell = [img_anomal img_anomal_unmarked];
for i=1:17*2
    img_ano_all(:,:,:,i) = img_ano_all_cell{i};
    %img_ano_unmarked(:,:,:,i) = img_anomal_unmarked{i};
end


%% Auxiliary Function to load images
function [imgArray,fileSpec] = loadImg(myFolder)
%% Loads images from myFolder into struct array
% Input:
%   path          - Path of image folder

% Output:
%   imgArray      - Images saved as cell array
%   theFiles      - struct with File Information

%% Load Image

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.png'); 
fileSpec = dir(filePattern);
n = length(fileSpec);
imgArray = cell(1,n); 
fprintf(1, 'Now reading Image files\n');
for i=1:n
    baseFileName = fileSpec(i).name;
    fullFileName = fullfile(myFolder, baseFileName);
    imgArray{i} = imread(fullFileName);
    imshow(uint8(imgArray{i}));  % Display image. 
    drawnow; % Force display to update immediately.
end
end 