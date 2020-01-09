classdef InvDiffDenoise
    % InvDiffDenoise   Encoding datasets based on an inverse Diffusion Process
    % By training a model based on an inverse (or backward) Diffusion 
    % Process, noisy records can be denoised (encoded) and decoded.
    % Denoise Properties:
    %    trainData          - Data Set for training the Denoising Model
    %    l_idx              - Index of l-NN of record in trainData
    %    LArray             - Diffusion Model 
    %    encodedTraindata   - Encoded train dataset 
    %
    % Denoise Methods:
    %    encode              - Encodes (denoises) record based on pretrained model
    %    decode              - Decodes record based on pretrained model
    %    train               - Builds the Diffusion Model based on train dataset
    %    getEncodedTraindata - Accesses property encodedTraindata 
    %
    
    properties (Access = private, Hidden = true)
        trainData
        l_idx
        LArray
        encodedTraindata
        k
        iter
        metric
    end 
    methods
        function dObj = InvDiffDenoise(trainData,k,iter,metric)
            if nargin<4
                metric = 'euclidean';
            end
            dObj.metric = metric;
            if nargin<3
                iter = 15;
            end 
            dObj.iter = iter; 
            if nargin<2
                k = 10;
            end 
            dObj.k = k; 
            dObj.LArray = trainModel(trainData,k,iter,metric);
            dObj.trainData = trainData;
        end 
        function [encodedData,dObj] = encode(dObj,data)
            [dObj.encodedTraindata,dObj.l_idx] = encode(data, dObj.trainData, dObj.LArray);
            for i=1:length(dObj.encodedTraindata)
                encTraindata = dObj.encodedTraindata{i};
                encodedData{i} = encTraindata(dObj.l_idx,:);
            end
        end
        function [decodedData]= decode(dObj,encodedData)
            [decodedData] = decode(encodedData,dObj.encodedTraindata,dObj.LArray,dObj.l_idx); 
        end
        function trainData = getEncodedTraindata(dObj)
            trainData = dObj.encodedTraindata;
        end 
    end 
end 