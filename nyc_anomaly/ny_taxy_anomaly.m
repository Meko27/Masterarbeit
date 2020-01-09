%% Load csv and reshape data
nyc_table = readtable('/Users/meko/Desktop/nyc_taxi.csv');

%% Transform data
data_raw = table2array(nyc_table(:,2));
dim = 48;
no_samples = length(data_raw);
no_days = no_samples/dim;

data_red = data_raw(9:end-40); % Cut first 9 and last 39 values
data_red_mat = reshape(data_red,dim,no_days-1)';
no_samples_red = length(data_red);

%% Denoising
k = 10; iter = 10;
DObj = InvDiffDenoise(data_red_mat,k,iter);
[data_red_mat_d,DObj] = DObj.encode(data_red_mat);

%% Calculate distance through diffusion process
n_ano = 10;
for i=1:iter
    if i==1
        d{i} = abs(data_red_mat_d{i} - data_red_mat);
        d_total = d{i};
    else
        d{i} = abs(data_red_mat_d{i} - data_red_mat_d{i-1});
        d_total = d_total + d{i}; % sum geodesic distances
    end
end
d_total_eucl = sqrt(sum(d_total.^2,2));

%% Calculate most anomalous points
% Eucl. distance of days
[ano_day.dist,ano_day.idx_day] = maxk(d_total_eucl,n_ano);
[nor_day.dist,nor_day.idx_day] = mink(d_total_eucl,n_ano); % needed?

for i=1:n_ano
    [~,ano_day.hour(i)] = max(d_total(ano_day.idx_day(i),:));
end
        
ano_day.label = (1:10)'; 

%% date vector
n_months = 7;
t_start = datetime(2014,7,01,4,0,0); % first record of reduced data set
t_end = datetime(2015,01,31,3,30,0); % last record of reduced data set
date_vect(1) = t_start;
for i=2:no_samples_red
    date_vect(i) = date_vect(i-1) + 1/48;
end 
date_vect = date_vect';

%% Ranges of months
date_vect_num = datenum(date_vect);
start_m(1) = 1; end_m(1) = find(date_vect_num==datenum(datetime(2014,8,1,3,30,0)));
start_m(2) = find(date_vect_num==datenum(datetime(2014,8,1,4,00,0)));
end_m(2) = find(date_vect_num==datenum(datetime(2014,9,1,3,30,0)));
start_m(3) = find(date_vect_num==datenum(datetime(2014,9,1,4,0,0)));
end_m(3) = find(date_vect_num==datenum(datetime(2014,10,1,3,30,0)));
start_m(4) = find(date_vect_num==datenum(datetime(2014,10,1,4,0,0)));
end_m(4) = find(date_vect_num==datenum(datetime(2014,11,1,3,30,0)));
start_m(5) = find(date_vect_num==datenum(datetime(2014,11,1,4,0,0)));
end_m(5) = find(date_vect_num==datenum(datetime(2014,12,1,3,30,0)));
start_m(6) = find(date_vect_num==datenum(datetime(2014,12,1,4,0,0)));
end_m(6) = find(date_vect_num==datenum(datetime(2015,1,1,3,30,0)));
start_m(7) = find(date_vect_num==datenum(datetime(2015,1,1,4,0,0)));
end_m(7) = find(date_vect_num==datenum(datetime(2015,1,31,3,30,0)));

% anomalous and normal regions
date_mat = reshape(date_vect,[dim,no_days-1])'; % days matrix (n_days x dim)
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
    plot(date_vect(start_m(i):end_m(i)) , data_red(start_m(i):end_m(i)), 'LineWidth',1.5)
    hold on
    for j=1:n_ano
        if ano_day.idx_start(j) >= start_m(i) && ano_day.idx_end(j) <= end_m(i)
            plot(date_vect(ano_day.idx_start(j):ano_day.idx_end(j)) , data_red(ano_day.idx_start(j):ano_day.idx_end(j)),'red','LineWidth',2)
            %text(t_array(ano_days_start(j)+i_hour_max(j)) , data_red(ano_days_start(j)+i_hour_max(j)) , num2str(ano_label(j)))
         end 
    end 
end 

%% Plot three least anomalous days against 10 most normal days
figure(2)
for i=1:n_ano
    plot(1:48,data_red_mat(nor_day.idx_day(i),:),'black')
    hold on
end
    plot(1:48,data_red_mat(ano_day.idx_day(8),:),'red','LineWidth',2)
    plot(1:48,data_red_mat(ano_day.idx_day(9),:),'green','LineWidth',2)
    plot(1:48,data_red_mat(ano_day.idx_day(10),:),'blue','LineWidth',2)
    