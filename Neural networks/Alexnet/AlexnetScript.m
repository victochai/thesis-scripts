%% feature extraction / AlexNet / m last

%% I. load 
% alexnet has 8 layers: 5 conv and 3 fc layers

clc, clear
net = alexnet;
layers = net.Layers;
disp(layers);
% deepNetworkDesigner;

%% II. loading input data (can be changed)
% alexnet takes as an input images of size [227, 227, 3]
% input data format here: 4D [227, 227, 3, m] --> numeric array
% also can be: imds, ds
% can find it here: 
% https://it.mathworks.com/help/deeplearning/ref/trainnetwork.html#mw_6a0ead40-d0f3-4af8-be23-37b407b8e923

load('images.mat');
images = permute(images, [2, 3, 4, 1]);
disp(size(images));

%% III. extract feature maps for one layer from 8

layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'};
feature_maps = {}
cos = {}
cos_small = {}

for i = 1:numel(layers)
    
    disp(layers{i})
    feature_map = activations(net, images, layers{i});
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    size(feature_map)  % data is now vectorized 
    feature_maps{i} = feature_map;


    %% IV. reshape feature maps / should be adjusted according to m
    % here 7 conds + 336 images in total / 48 images for each cond

    body(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 1:48), 2);
    hand(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 49:96), 2);
    face(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 97:144), 2);
    tool(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 145:192), 2);
    man(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 193:240), 2);
    nonman(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 241:288), 2);
    chair(1:size(feature_map, 1), 1) = mean(feature_map(1:size(feature_map, 1), 289:336), 2);

    matrix(:, 1) = body;
    matrix(:, 2) = hand;
    matrix(:, 3) = face;
    matrix(:, 4) = tool;
    matrix(:, 5) = man;
    matrix(:, 6) = nonman;
    matrix(:, 7) = chair;

    co_small = corr(matrix);
    cos_small{i} = co_small

    %% V. visualize small corr / save

    % imagesc(co_small);
    % fc8_co_small = co_small;
    % save('fc8_co_small', 'fc8_co_small');

    %% VI. visualize big corr / save

    co = corr(feature_map);
    cos{i} = co;
    % imagesc(co);
    % fc8_co = co;
    % save('fc8_co', 'fc8_co');
    
    clear feature_map co co_small matrix body hand face man tool nonman chair
    
end

cd("C:\Users\victo\Desktop\thesis-scripts\Brain\Images")
save("feature_maps_ALEX", "feature_maps")
save("cos_ALEX", "cos")
save("cos_small_ALEX", "cos_small")

%% VII. visualize all / small corr

clc, clear
conv_small_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\AlexNet\CONV s\CONV SMALL';
cd(conv_small_dir)

cd(conv_small_dir);
load('conv1_co_small.mat');
load('conv2_co_small.mat');
load('conv3_co_small.mat');
load('conv4_co_small.mat');
load('conv5_co_small.mat');
load('fc6_co_small.mat');
load('fc7_co_small.mat');
load('fc8_co_small.mat');

figure(1);
name = {'AlexNet. Every condition averaged: 7x7 matrix'; '1-body, 2-hand, 3-face, 4-tool, 5-MAN, 6-NMan, 7-chair'};
sgtitle(name);
  
subplot(3, 3, 1);
imagesc(cos_small{1,1});
h1 = colorbar;
name = ['conv' num2str(1) '. filter 11x11x96'];
title(name);
subplot(3, 3, 2);
imagesc(cos_small{1,2});
h2 = colorbar;
name = ['conv' num2str(2) '. filter 5x5x48'];
title(name);
subplot(3, 3, 3);
imagesc(cos_small{1,3});
h3 = colorbar;
name = ['conv' num2str(3) '. filter 3x3x256'];
title(name);
subplot(3, 3, 4);
imagesc(cos_small{1,4});
h4 = colorbar;
name = ['conv' num2str(4) '. filter 3x3x192'];
title(name);
subplot(3, 3, 5);
imagesc(cos_small{1,5});
h5 = colorbar;
name = ['conv' num2str(5) '. filter 3x3x192'];
title(name);
subplot(3, 3, 6);
imagesc(cos_small{1,6});
h6 = colorbar;
name = ['fc' num2str(6) '. 4096 neurons'];
title(name);
subplot(3, 3, 7);
imagesc(cos_small{1,7});
h7 = colorbar;
name = ['fc' num2str(7) '. 4096 neurons'];
title(name);
subplot(3, 3, 8);
imagesc(cos_small{1,8});
h8 = colorbar;
name = ['fc' num2str(8) '. 1000 neurons'];
title(name);

%% VIII. visualize all / big corr

clc, clear
conv_big_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\AlexNet\CONV s\CONV BIG';
cd(conv_big_dir)

cd(conv_big_dir);
load('conv1_co.mat');
load('conv2_co.mat');
load('conv3_co.mat');
load('conv4_co.mat');
load('conv5_co.mat');
load('fc6_co.mat');
load('fc7_co.mat');
load('fc8_co.mat');

figure(2);
name = {'AlexNet. 48 stimuli per condition'; '1-body, 2-hand, 3-face, 4-tool, 5-MAN, 6-NMan, 7-chair'};
sgtitle(name);

subplot(3, 3, 1);
imagesc(conv1_co);
h1 = colorbar;
name = ['conv' num2str(1) '. filter 11x11x96'];
title(name);
subplot(3, 3, 2);
imagesc(conv2_co);
h2 = colorbar;
name = ['conv' num2str(2) '. filter 5x5x48'];
title(name);
subplot(3, 3, 3);
imagesc(conv3_co);
h3 = colorbar;
name = ['conv' num2str(3) '. filter 3x3x256'];
title(name);
subplot(3, 3, 4);
imagesc(conv4_co);
h4 = colorbar;
name = ['conv' num2str(4) '. filter 3x3x192'];
title(name);
subplot(3, 3, 5);
imagesc(conv5_co);
h5 = colorbar;
name = ['conv' num2str(5) '. filter 3x3x192'];
title(name);
subplot(3, 3, 6);
imagesc(fc6_co);
h6 = colorbar;
name = ['fc' num2str(6) '. 4096 neurons'];
title(name);
subplot(3, 3, 7);
imagesc(fc7_co);
h7 = colorbar;
name = ['fc' num2str(7) '. 4096 neurons'];
title(name);
subplot(3, 3, 8);
imagesc(fc8_co);
h8 = colorbar;
name = ['fc' num2str(8) '. 1000 neurons'];
title(name);

%% IX. visualizing all on one scale / small corrs / weird way

clc, clear
conv_small_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\AlexNet\CONV s\CONV SMALL';
cd(conv_small_dir)
load('conv1_co_small.mat');
load('conv2_co_small.mat');
load('conv3_co_small.mat');
load('conv4_co_small.mat');
load('conv5_co_small.mat');
load('fc6_co_small.mat');
load('fc7_co_small.mat');
load('fc8_co_small.mat');

a = zeros(23, 23);

a(1:7, 1:7) = conv1_co_small(:, :);
a(1:7, 9:15) = conv2_co_small(:, :);
a(1:7, 17:23) = conv3_co_small(:, :);
a(9:15, 1:7) = conv4_co_small(:, :);
a(9:15, 9:15) = conv5_co_small(:, :);
a(9:15, 17:23) = fc6_co_small(:, :);
a(17:23, 1:7) = fc7_co_small(:, :);
a(17:23, 9:15) = fc7_co_small(:, :);

figure(3);
imagesc(a);
set(gca, 'XTick', [1:7,9:15,17:23], 'XTickLabel', [1:7,1:7,1:7])  
set(gca, 'YTick', [1:7,9:15,17:23], 'YTickLabel', [1:7,1:7,1:7])
colormap parula;
colorbar;
name = {'AlexNet. CONV layers'; '1-body, 2-hand, 3-face, 4-tool, 5-MAN, 6-NMan, 7-chair'};
title(name);

%% X. RDMs 

clc, clear
conv_small_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\AlexNet\CONV s\CONV SMALL';
cd(conv_small_dir)
load('conv1_co_small.mat');
load('conv2_co_small.mat');
load('conv3_co_small.mat');
load('conv4_co_small.mat');
load('conv5_co_small.mat');
load('fc6_co_small.mat');
load('fc7_co_small.mat');
load('fc8_co_small.mat');

rdm1 = 1 - conv1_co_small;
rdm2 = 1 - conv2_co_small;
rdm3 = 1 - conv3_co_small;
rdm4 = 1 - conv4_co_small;
rdm5 = 1 - conv5_co_small;
rdm6 = 1 - fc6_co_small;
rdm7 = 1 - fc7_co_small;
rdm8 = 1 - fc8_co_small;

save('rdm1', 'rdm1');
save('rdm2', 'rdm2');
save('rdm3', 'rdm3');
save('rdm4', 'rdm4');
save('rdm5', 'rdm5');
save('rdm6', 'rdm6');
save('rdm7', 'rdm7');
save('rdm8', 'rdm8');

%% XI. visualize RDMs of AlexNet

clc, clear
alexnet_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\RDMs\AlexNet';
cd(alexnet_dir)
load('rdm1.mat');  % conv1
load('rdm2.mat');  % conv2
load('rdm3.mat');  % conv3
load('rdm4.mat');  % conv4
load('rdm5.mat');  % conv5
load('rdm6.mat');  % fc6
load('rdm7.mat');  % fc7
load('rdm8.mat');  % fc8

figure(4);
name = {'AlexNet. RDMs'; '1-body, 2-hand, 3-face, 4-tool, 5-MAN, 6-NMan, 7-chair'};
sgtitle(name);

subplot(3, 3, 1);
imagesc(triu(rdm1));
h1 = colorbar;
name = ['conv' num2str(1) '. filter size 11x11x96'];
title(name);
subplot(3, 3, 2);
imagesc(triu(rdm2));
h2 = colorbar;
name = ['conv' num2str(2) '. filter size 5x5x48'];
title(name);
subplot(3, 3, 3);
imagesc(triu(rdm3));
h3 = colorbar;
name = ['conv' num2str(3) '. filter size 3x3x256'];
title(name);
subplot(3, 3, 4);
imagesc(triu(rdm4));
h4 = colorbar;
name = ['conv' num2str(4) '. filter size 3x3x192'];
title(name);
subplot(3, 3, 5);
imagesc(triu(rdm5));
h5 = colorbar;
name = ['conv' num2str(5) '. filter size 3x3x192'];
title(name);
subplot(3, 3, 6);
imagesc(triu(rdm6));
h6 = colorbar;
name = ['fc' num2str(6) '. 4096 neurons'];
title(name);
subplot(3, 3, 7);
imagesc(triu(rdm7));
h7 = colorbar;
name = ['fc' num2str(7) '. 4096 neurons'];
title(name);
subplot(3, 3, 8);
imagesc(triu(rdm8));
h8 = colorbar;
name = ['fc' num2str(8) '. 1000 neurons'];
title(name);
