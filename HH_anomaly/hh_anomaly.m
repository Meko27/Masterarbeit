%% Load csv and reshape data
data_table = readtable('/Users/meko/Documents/MATLAB/Anomaly_Detection/HH_anomaly/hh_17sm_stundenwerte.csv');

no2_vec = table2array(data_table(4:end,2));
no_samples = length(no2_vec);
n_months = 11;
%% Fill empty samples with value of pre sample and convert to num array

no2_vec_filled = no2_vec;
idx_filled = [];
j=1;
for i=1:no_samples
    if isempty(no2_vec{i})
        no2_vec_filled{i} = no2_vec_filled{i-1};
        idx_filled(j) = i;
        j=j+1;
    end
    no2_vec_filled_num(i) = str2num(no2_vec_filled{i});
end

%% Transform to matrix

dim = 24; %00:00 - 23:00 
no_days = no_samples/dim;
no2_mat = reshape(no2_vec_filled_num,dim,no_days)';

%% Denoising
k = 7; iter = 10;
DObj_hh = InvDiffDenoise(no2_mat,k,iter);
[no2_mat_d,DObj_hh] = DObj_hh.encode(no2_mat);

%% Calculate distance through diffusion process

n_ano = 10;
for i=1:iter
    if i==1
        d{i} = abs(no2_mat_d{i} - no2_mat);
        d_total = d{i};
    else
        d{i} = abs(no2_mat_d{i} - no2_mat_d{i-1});
        d_total = d_total + d{i}; % sum geodesic distances
    end
end
d_total_eucl = sqrt(sum(d_total.^2,2));

%% Calculate most anomalous points

% Eucl. distance of days
[ano_day.dist,ano_day.idx_day] = maxk(d_total_eucl,n_ano);
[nor_day.dist,nor_day.idx_day] = mink(d_total_eucl,n_ano); 

for i=1:n_ano
    [~,ano_day.hour(i)] = max(d_total(ano_day.idx_day(i),:));
end
        
ano_day.label = (1:10)'; 

%% date vector
n_months = 11;
t_start = datetime(2019,01,01,0,0,0); % first record of reduced data set
t_end = datetime(2019,11,31,23,30,0); % last record of reduced data set
date_vect(1) = t_start;
for i=2:no_samples
    date_vect(i) = date_vect(i-1) + 1/dim;
end 
date_vect = date_vect';

%% Ranges of months
date_vect_num = datenum(date_vect);
start_m(1) = 1; end_m(1) = find(date_vect_num==datenum(datetime(2019,1,31,23,00,0)));
start_m(2) = find(date_vect_num==datenum(datetime(2019,2,1,00,00,0)));
end_m(2) = find(date_vect_num==datenum(datetime(2019,2,30,23,00,0)));
start_m(3) = find(date_vect_num==datenum(datetime(2019,3,1,0,0,0)));
end_m(3) = find(date_vect_num==datenum(datetime(2019,3,31,23,00,0)));
start_m(4) = find(date_vect_num==datenum(datetime(2019,4,1,0,0,0)));
end_m(4) = find(date_vect_num==datenum(datetime(2019,4,30,23,00,0)));
start_m(5) = find(date_vect_num==datenum(datetime(2019,5,1,0,0,0)));
end_m(5) = find(date_vect_num==datenum(datetime(2019,5,31,23,0,0)));
start_m(6) = find(date_vect_num==datenum(datetime(2019,6,1,0,0,0)));
end_m(6) = find(date_vect_num==datenum(datetime(2019,6,30,23,0,0)));
start_m(7) = find(date_vect_num==datenum(datetime(2019,7,1,0,0,0)));
end_m(7) = find(date_vect_num==datenum(datetime(2019,7,31,23,0,0)));
start_m(8) = find(date_vect_num==datenum(datetime(2019,8,1,0,0,0)));
end_m(8) = find(date_vect_num==datenum(datetime(2019,8,31,23,0,0)));
start_m(9) = find(date_vect_num==datenum(datetime(2019,9,1,0,0,0)));
end_m(9) = find(date_vect_num==datenum(datetime(2019,9,30,23,0,0)));
start_m(10) = find(date_vect_num==datenum(datetime(2019,10,1,0,0,0)));
end_m(10) = find(date_vect_num==datenum(datetime(2019,10,31,23,0,0)));
start_m(11) = find(date_vect_num==datenum(datetime(2019,11,1,0,0,0)));
end_m(11) = find(date_vect_num==datenum(datetime(2019,11,30,23,0,0)));

% anomalous and normal regions
date_mat = reshape(date_vect,[dim,no_days])'; % days matrix (n_days x dim)
for i=1:n_ano
    ano_day.dates(i,:) = date_mat(ano_day.idx_day(i),:);
    nor_day.dates(i,:) = date_mat(nor_day.idx_day(i),:);
    ano_day.idx_start(i) = find(date_vect_num==datenum(datetime(ano_day.dates(i,1))));
    nor_day.idx_start(i) = find(date_vect_num==datenum(datetime(ano_day.dates(i,1))));
    ano_day.idx_end(i) = find(date_vect_num==datenum(datetime(ano_day.dates(i,end))));
    nor_day.idx_end(i) = find(date_vect_num==datenum(datetime(ano_day.dates(i,end))));
end

%% Plot all
figure(1)
for i=1:n_months
    subplot(n_months,1,i);
    plot(date_vect(start_m(i):end_m(i)) , no2_vec_filled_num(start_m(i):end_m(i)), 'LineWidth',1.5)
    hold on
    for j=1:n_ano
        if ano_day.idx_start(j) >= start_m(i) && ano_day.idx_end(j) <= end_m(i)
            plot(date_vect(ano_day.idx_start(j):ano_day.idx_end(j)) , no2_vec_filled_num(ano_day.idx_start(j):ano_day.idx_end(j)),'red','LineWidth',2)
            %text(t_array(ano_days_start(j)+i_hour_max(j)) , data_red(ano_days_start(j)+i_hour_max(j)) , num2str(ano_label(j)))
         end 
    end 
end 
