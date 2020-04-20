%% Prepare | RESNET50 

clc, clear
net = densenet201;
% deepNetworkDesigner % So, yes, it is 50 layers

%% Layers | Images

cd('D:\thesis-scripts\images');
load('images_forDNN.mat');
images = images_forDNN;
clear images_forDNN
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % (400, 400, 3, 336) or (width, height, rgb_channels, n_samples)
images = uint8(imresize(images, [299 299]));

% cd("D:\THESIS\DATA\DATA. CAT12")
% load("images_vgg19.mat")
% images = permute(images_vgg, [2, 3, 4, 1]);
% disp(size(images));
% clear images_vgg

layers = net.Layers;
l = {}
for i = 1:numel(layers)    
    l{i, 1} = layers(i).Name;        
end
clear layers

layers = {}
for i = 1:numel(l)
    a = l{i, 1}
    % For ResNet101
    ans = regexp(a, "conv1\|conv$|fc1000$|pool\d_conv|conv\d_block\d_\d_conv$|conv\d_block\d\d_\d_conv$")
    % For ResNet50
    % ans = regexp(a, "conv1$|fc1000$|res\d\w_branch\d\w$|fc1000$|")
    if ans == 1
        layers{i, 1} = l{i, 1}
    end       
end

layers = layers(~cellfun('isempty', layers))
clear a ans i % Should be 201 layers

%% Feature maps

cd("D:\thesis-scripts\Neural networks\Densenet201")

for i = 100:201
    
    disp(layers{i});
    feature_map = activations(net, images, layers{i});
    size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);

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
    % cos_small{i} = co_small
    
    co = corr(feature_map);
    % cos{i} = co;
        
    text = ["co_orig_" + i + "_" + layers{i}] % CHANGE
    save(text, "co");
    text = ["co_small_orig_" + i + "_" + layers{i}] % CHANGE
    save(text, "co_small");
    
    clear feature_map body hand face man nonman tool chair co co_small matrix
    
end