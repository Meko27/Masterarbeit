%% Load data
path = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet';
path_normal = '/train/good';
path_anomal = '/test/hole';

%% Loading images
[img_normal,~] = loadImg(strcat(path,path_normal)); 
[img_anomal,~] = loadImg(strcat(path,path_anomal)); 

%% transform and compression
scale_factor = 0.5;
num_pixel = size(img_normal{1},1)*scale_factor;
for i=1:length(img_normal)
    img_normal_scaled{i} = imresize(img_normal{i},scale_factor);
%     img_normal_g{i} = rgb2gray(img_normal{i});
%     img_normal_g{i} = imresize(img_normal_g{i},scale_factor);
%     img_normal_g_mat(i,:) = reshape(img_normal_g{i},1,num_pixel^2);
end
for i=1:length(img_anomal)
    img_anomal_scaled{i} = imresize(img_anomal{i},scale_factor);
%     img_anomal_g{i} = rgb2gray(img_anomal{i});
%     img_anomal_g{i} = imresize(img_anomal_g{i},scale_factor);
%     img_anomal_g_mat(i,:) = reshape(img_anomal_g{i},1,num_pixel^2);
end

%img_g_mat = double([img_normal_g_mat ; img_anomal_g_mat]);

%% Extract HOG features
NumBins = 9;
CellSize = [32 32];
BlockSize = [2 2];
BlockOverlap = BlockSize/2;

ground_dist = calculateGroundDist(img_normal_scaled{1},NumBins,CellSize,...
                                  BlockSize,BlockOverlap);

for i=1:length(img_normal)
    hog_array(i,:) = extractHOGFeatures(img_normal_scaled{i},...
                                        'NumBins',NumBins,...
                                        'CellSize',CellSize,...
                                        'BlockSize',BlockSize,...
                                        'BlockOverlap',BlockOverlap);
end 

for i=1:length(img_anomal)
    hog_array(length(img_normal)+i,:) = extractHOGFeatures(img_anomal_scaled{i},...
                                        'NumBins',NumBins,...
                                        'CellSize',CellSize,...
                                        'BlockSize',BlockSize,...
                                        'BlockOverlap',BlockOverlap);
end
hog_array = double(hog_array);

%% transform and compression
% scale_factor = 0.5;
% num_pixel = size(img_normal{1},1)*scale_factor;
% for i=1:length(img_normal)
%     img_normal_g{i} = rgb2gray(img_normal{i});
%     img_normal_g{i} = imresize(img_normal_g{i},scale_factor);
%     img_normal_g_mat(i,:) = reshape(img_normal_g{i},1,num_pixel^2);
% end
% for i=1:length(img_anomal)
%     img_anomal_g{i} = rgb2gray(img_anomal{i});
%     img_anomal_g{i} = imresize(img_anomal_g{i},scale_factor);
%     img_anomal_g_mat(i,:) = reshape(img_anomal_g{i},1,num_pixel^2);
% end
% 
% img_g_mat = double([img_normal_g_mat ; img_anomal_g_mat]);

%% Construct anomaly construction element
tic
metric = 'emd';
k = 7; iter = 10; n_outliers = 17;
Ad_detector = Ano_detector(hog_array,k,iter,metric,ground_dist);
[distvect_ano,idx_outliers, Ad_obj] = Ad_detector.calc_outliers(n_outliers);
toc

        
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