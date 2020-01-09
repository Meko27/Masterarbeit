% TTest Ground Distance

% Load two sample images
img1 = imread('/Users/meko/Documents/MATLAB/Anomaly_Detection/data/test_ground_dist/img1.png');
img2 = imread('/Users/meko/Documents/MATLAB/Anomaly_Detection/data/test_ground_dist/img2.png');
img3 = imread('/Users/meko/Documents/MATLAB/Anomaly_Detection/data/test_ground_dist/img3.png');
img4 = imread('/Users/meko/Documents/MATLAB/Anomaly_Detection/data/test_ground_dist/img4.png');
path = '/Users/meko/Documents/MATLAB/Anomaly_Detection/data/carpet';
path_normal = '/train/good';
path_anomal = '/test/hole';
img11 = imread(strcat(path,path_normal,'/000.png'));
img22 = imread(strcat(path,path_anomal,'/004.png'));
%% Resize images
resize_factor = 0.5;
img1_resize = imresize(img1,resize_factor);
img2_resize = imresize(img2,resize_factor);
img3_resize = imresize(img3,resize_factor);
img4_resize = imresize(img4,resize_factor);
img11_resize = imresize(img11,resize_factor);
img22_resize = imresize(img22,resize_factor);
%% extract hog features
NumBins = 9;
CellSize = [8 8];
BlockSize = [2 2];
BlockOverlap = BlockSize/2;
%%
[feat1,vis1] = extractHOGFeatures(img1_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap);
[feat2,vis2] = extractHOGFeatures(img2_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap);
feat3 = extractHOGFeatures(img3_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap);                    
feat4 = extractHOGFeatures(img4_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap); 
%%
[feat11,vis11] = extractHOGFeatures(img11_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap); 
[feat22,vis22] = extractHOGFeatures(img22_resize,'NumBins',NumBins, 'CellSize',CellSize,...
                        'BlockSize',BlockSize, 'BlockOverlap',BlockOverlap); 

%% Calculate Ground dist
ground_dist = calculateGroundDist(img1_resize,NumBins,CellSize,BlockSize,BlockOverlap);

%% Calculate emd
t11=tic
emd12 = emd_mex(double(feat1),double(feat2),ground_dist);
emd13 = emd_mex(double(feat1),double(feat3),ground_dist);
emd14 = emd_mex(double(feat1),double(feat4),ground_dist);
emd23 = emd_mex(double(feat2),double(feat3),ground_dist);
t1=toc(t11)
t22=tic
ssim12 = ssim(img1_resize,img2_resize);
ssim13 = ssim(img1_resize,img3_resize);
ssim14 = ssim(img1_resize,img4_resize);
ssim23 = ssim(img2_resize,img3_resize);
ssim34 = ssim(img3_resize,img4_resize);
t2=toc(t22)
%%
emd24 = emd_mex(double(feat2),double(feat4),ground_dist);
emd34 = emd_mex(double(feat3),double(feat4),ground_dist);

%% Plot images
figure
subplot(2,2,1)
imshow(img1)
hold on
title('img1')
subplot(2,2,2)
imshow(img2)
title('img2')
subplot(2,2,3)
imshow(img3)
title('img3')
subplot(2,2,4)
imshow(img4)
title('img4')