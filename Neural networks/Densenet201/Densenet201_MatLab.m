%% 1.) Prepare | RESNET50 

clc, clear
net = densenet201;
% deepNetworkDesigner % So, yes, it is 50 layers

%% 2.) Layers | Images

cd('D:\THESIS\IMAGES EXPERIMENT, 336');
load('img1.mat');
images = img;
clear img
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % (400, 400, 3, 336) 
images = uint8(imresize(images, [299 299]));
disp(size(images)); % (299, 299, 3, 336) 

layers = net.Layers;
l = {}
for i = 1:numel(layers)    
    l{i, 1} = layers(i).Name;        
end
clear layers

layers = {}
for i = 1:numel(l)
    a = l{i, 1}
    ans = regexp(a, "conv1\|conv$|fc1000$|pool\d_conv|conv\d_block\d_\d_conv$|conv\d_block\d\d_\d_conv$")
    if ans == 1
        layers{i, 1} = l{i, 1}
    end       
end

layers = layers(~cellfun('isempty', layers))
clear a ans i % Should be 201 layers

%% 3.) Feature maps

cd("D:\thesis-scripts\Neural networks\Densenet201")

for i = :201
    
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
    rdm_small = 1 - co_small;
    % cos_small{i} = co_small
    
    co = corr(feature_map);
    rdm = 1 - co;
    % cos{i} = co;
    
    cd("D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv big")
    text = ["co_" + i + "_"] % CHANGE
    save(text, "co");
    text = ["rdm_" + i + "_"] % CHANGE
    save(text, "rdm");
    cd("D:\thesis-scripts\Neural networks\Densenet201\Experimental images\Conv small")
    text = ["co_small_" + i + "_"] % CHANGE
    save(text, "co_small");
    text = ["rdm_small_" + i + "_"] % CHANGE
    save(text, "rdm_small");
    
    clear feature_map body hand face man nonman tool chair co co_small matrix
    
end