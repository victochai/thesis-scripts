%% feature extraction / AlexNet / m last

%% 1.) load 
% alexnet has 8 layers: 5 conv and 3 fc layers

clc, clear
net = alexnet;
% layers = net.Layers;
% disp(layers);
% deepNetworkDesigner;

%% 2.) loading input data (can be changed)
% alexnet takes as an input images of size [227, 227, 3]
% input data format here: 4D [227, 227, 3, m] --> numeric array

cd('D:\THESIS\IMAGES EXPERIMENT, 336');
load('img1.mat');
images = img;
clear img
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % (400, 400, 3, 336) 
images = uint8(imresize(images, [227 227]));
disp(size(images)); % (227, 227, 3, 336) 

%% 3.) extract feature maps for one layer from 8

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


    %% 4.) reshape feature maps / should be adjusted according to m
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

    %% 5.) visualize small corr / save

    % imagesc(co_small);
    % fc8_co_small = co_small;
    % save('fc8_co_small', 'fc8_co_small');

    %% 6.) visualize big corr / save

    co = corr(feature_map);
    cos{i} = co;
    % imagesc(co);
    % fc8_co = co;
    % save('fc8_co', 'fc8_co');
    
    clear feature_map co co_small matrix body hand face man tool nonman chair
    
end

%% 7.) SAVE

cd("D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Big")
save("cos_alex", "cos")
cd("D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Small")
save("cos_small_alex", "cos_small")

%% 8.) Make RDMs

rmds = {}
rdms_small = {}

for i = 1:8
    rdms{i} = 1 - cos{i}
    rdms_small{i} = 1 - cos_small{i}   
end

cd("D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Big")
save("rmds_alex", "rdms")
cd("D:\thesis-scripts\Neural networks\Alexnet\Experimental images\Conv Small")
save("rdms_small_alex", "rdms_small")
