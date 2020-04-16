%% Important info

clc, clear
net1 = inceptionresnetv2;
cd("D:\thesis-scripts")
load("imagesV3.mat")
images = permute(imagesV3, [2, 3, 4, 1]);
disp(size(images));
clear imagesV3

layers = net.Layers;
l = {}

for i = 1:numel(layers)    
    l{i, 1} = layers(i).Name;        
end
clear layers

layers = {}
for i = 1:numel(l)
    a = l{i, 1}
    ans = regexp(a, "conv2d_\d|block\d+_\d+_conv|conv_7b$|predictions$")
    if ans == 1
        layers{i, 1} = l{i, 1}
    end       
end

layers = layers(~cellfun('isempty', layers))
clear a ans i l
% deepNetworkDesigner

%% NO CONCAT

layers{245, 1} % CHANGE

for i = 245 % CHANGE
    
    disp(layers{i})
    feature_map = activations(net, images, layers{i});
    size(feature_map)
    % CHANGE
    feature_maps{176, 1} = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    size(feature_map)
    clear feature_map i
    
end

%% CONCAT 2

layers{85:86, 1} % CHANGE

for i = 85:86 % CHANGE
    
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map) 
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_maps{1, 1} = vertcat(c{i-1,1},c{i,1}); % CHANGE
clear i feature_map c

%% CONCAT 3

layers{191:193, 1} % CHANGE

for i = 191:193 % CHANGE
    
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map)
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_maps{134, 1} = vertcat(c{i-2,1},c{i-1,1},c{i,1}); % CHANGE
clear i c feature_map

%% CONCAT 4

layers{9:12, 1} % CHANGE

for i = 9:12 % CHANGE
    
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map)
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_maps{8, 1} = vertcat(c{i-3,1},c{i-2,1},c{i-1,1},c{i,1}); % CHANGE
clear i c feature_map

%% F MAPS 

cd("D:\inception_resnet_v2")

for i = 1 % CHANGE
    
    feature_map = feature_maps{i, 1};
    
    % Co-s

    co = corr(feature_map);
    cos{i} = co;

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
    coS_small{i} = co_small;
    
    % text = ["conv2d_" + i + "_stem"]
    % save(text, "feature_map");
    % text = ["co_" + "conv2d_" + i + "_pred"] % CHANGE
    % save(text, "co");
    % text = ["co_small_" + "conv2d_" + i + "_pred"] % CHANGE
    % save(text, "co_small");
    
    clear matrix body face hand tool man nonman chair feature_map i co co_small
    
end

clear feature_maps