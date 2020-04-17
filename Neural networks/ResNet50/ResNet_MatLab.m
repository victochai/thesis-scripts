%% Prepare | RESNET50 

clc, clear
net = resnet50;
% deepNetworkDesigner % So, yes, it is 50 layers

%% RESNET 101

clc, clear
net = resnet101;
% deepNetworkDesigner % So, yes, it is 50 layers

%% Layers | Images

cd("D:\THESIS\DATA\DATA. CAT12")
load("images_vgg19.mat")
images = permute(images_vgg, [2, 3, 4, 1]);
disp(size(images));
clear images_vgg

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
    ans = regexp(a, "conv1$|fc1000$|res\d\w_branch\d\w$|fc1000$|res\d\w\d_branch\d\w$|res\d\w\d\d_branch\d\w$")
    % For ResNet50
    % ans = regexp(a, "conv1$|fc1000$|res\d\w_branch\d\w$|fc1000$|")
    if ans == 1
        layers{i, 1} = l{i, 1}
    end       
end

layers = layers(~cellfun('isempty', layers))
clear a ans i % Should be 50 layers

%% Feature maps

cd("D:\thesis-scripts\Neural networks\ResNet50")

for i = 54:101
    
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
        
    text = ["co_" + "conv2d_" + i + "_" + layers{i}] % CHANGE
    save(text, "co");
    text = ["co_small_" + "conv2d_" + i + "_"+ layers{i}] % CHANGE
    save(text, "co_small");
    
    clear feature_map body hand face man nonman tool chair co co_small matrix
    
end
