%% Loading

clc, clear
cd("D:\thesis-scripts\Visualizations and stats for the paper")
load("ant.mat")
load("calc.mat")
load("cos_small_first.mat")
load("cos_small_last.mat")

%% MDS

Y_a = mdscale(pdist(ant),2);
Y_c = mdscale(pdist(calc),2);

Y_last = zeros(6,2,5);
Y_first = zeros(6,2,5);

for k=1:5
    Y_last(:,:,k) = mdscale(pdist(squeeze(cos_small_last(k,:,:))),2);
    Y_first(:,:,k) = mdscale(pdist(squeeze(cos_small_first(k,:,:))),2);
end

%% Visualize

scatter(Y_a(:,1), Y_a(:,2))
scatter(Y_c(:,1), Y_c(:,2))
scatter(Y_last(:,1,1), Y_last(:,2,1))
scatter(Y_last(:,1,2), Y_last(:,2,2))
scatter(Y_last(:,1,4), Y_last(:,2,4))

%% SAVE

save("Y_a", "Y_a")
save("Y_c", "Y_c")
save("Y_first", "Y_first")
save("Y_last", "Y_last")

%% Visualize more

for i = 1:5
    subplot(5,1,i)
    imshow(squeeze(cos_small_last(i,:,:)))   
end