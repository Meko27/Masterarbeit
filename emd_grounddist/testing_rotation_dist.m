%%
orientations = 9;
max_angle = 360;
orients_vect = linspace(0,max_angle*(1-1/orientations),orientations);

rotation_cost_unit = 20;
%%
idx = 0;
for i=1:orientations-1
    start = i+1;
    for j=start:orientations
        idx = idx+1;
        orients_dist_vect(idx) = angle_diff(orients_vect(i),orients_vect(j),max_angle);
    end 
end
%%
angle_unit_cost = 20;
diff_mat = squareform(orients_dist_vect');
diff_mat = diff_mat/rotation_cost_unit;
%% 
a = [1 2 3 ; 12 4 5; 1 3 5 ; 134 5 3];
k = 3;

dist = localDistMtrx(a,k)



%% Auxiliaray function to calculate the difference between two angles
function absDiffDeg = angle_diff(a,b,max_angle)
    normDeg = mod(a-b,max_angle);
    absDiffDeg = min(max_angle-normDeg, normDeg);
end 

