classdef Ano_detector
   properties (Access = private, Hidden = true)
       raw_data
       data_enc
       dist_vect_ano
       dist_vect_norm
       metric
       grounddist
       k
       iter
       n_ouliers
       idx_outliers
       idx_norm
   end
   
   methods
       function Ad_obj = Ano_detector(raw_data,k,iter,metric,grounddist)
           if nargin < 2
               iter = 15;
           end
               Ad_obj.iter = iter;
           if nargin < 3
               k = 7;
           end
           Ad_obj.k = k;
           if nargin < 4
               metric = 'euclidean';
           end
           Ad_obj.metric = metric;
           if nargin < 5
               grounddist = 0;
           end 
           Ad_obj.grounddist = grounddist;
           Ad_obj.raw_data = raw_data;
       end
       
       function [dist_vect_ano,idx_outliers, Ad_obj] = calc_outliers(Ad_obj,n_outliers)
           n = size(Ad_obj.raw_data,1); 
           % Denoising Process
           for i=1:Ad_obj.iter % TODO try parfor
               local_dist = localDistMtrx(Ad_obj.raw_data,Ad_obj.k,Ad_obj.metric,Ad_obj.grounddist);
               L_vect{i} = weightedGraphLaplacian(local_dist);
               D = spdiags(diag(L_vect{i}),0,n,n);
               Ad_obj.raw_data=(D+0.25*L_vect{i})\(D*Ad_obj.raw_data); % Solve inverse Pseudo Markov
               Ad_obj.data_enc{i} = Ad_obj.raw_data;
           end
           
           % Calculate distances of every iteration
           d_total = {};
           for i=1:Ad_obj.iter
               if i==1
                   d{i} = abs(Ad_obj.data_enc{i} - Ad_obj.raw_data);
                   d_total = d{i};
               else
                   d{i} = abs(Ad_obj.data_enc{i} - Ad_obj.data_enc{i-1});
                   d_total = d_total + d{i}; % sum 'geodesic' distances
               end
           end
           d_total_eucl = sqrt(sum(d_total.^2,2));
           
           % Caclulate largest distances
           [Ad_obj.dist_vect_ano,Ad_obj.idx_outliers] = maxk(d_total_eucl,n_outliers);
           [Ad_obj.dist_vect_norm,Ad_obj.idx_norm] = mink(d_total_eucl,n_outliers); 
           dist_vect_ano = Ad_obj.dist_vect_ano;
           idx_outliers = Ad_obj.idx_outliers;
       end
       function idx_norm = get_idx_norm(Ad_obj)
           idx_norm = Ad_obj.idx_norm
       end
       function dist_vect_norm = get_dist_vect_norm(Ad_obj)
           dist_vect_norm = Ad_obj.dist_vect_norm;
       end
   end 
end 
    
    