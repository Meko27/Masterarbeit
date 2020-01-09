function diff_mat = rotation_dist(orientations,signed)
% angle_diff - calculates the distance matrix between of angles in a vector
%              uniformly distributed from 0 to max_angles degrees
% Inputs:
%       - orientations  - Number of angle orientations 
%       - signed        - 0: unsigned (0-180) or 1: Signed angles (0-360)    
% Outputs:
%       - diff_mat:     - distance matrix of angles
%
if nargin < 2
    signed = 0;
end 

if signed == 1
    max_angle = 360;
else 
    max_angle = 180;
end 
orients_vect = linspace(0,max_angle*(1-1/orientations),orientations);
idx = 0;
for i=1:orientations-1
                start = i+1;
            for j=start:orientations
                idx = idx+1;
                orients_dist_vect(idx) = angle_diff(orients_vect(i),orients_vect(j),max_angle);
            end 
end
angle_unit_cost = 20; % angle difference which is mapped to one cost unit
diff_mat = squareform(orients_dist_vect);
diff_mat = diff_mat/angle_unit_cost;

% Auxiliaray function to calculate the difference between two angles
function absDiffDeg = angle_diff(a,b,max_angle)
    normDeg = mod(a-b,max_angle);
    absDiffDeg = min(max_angle-normDeg, normDeg);
end 

end 