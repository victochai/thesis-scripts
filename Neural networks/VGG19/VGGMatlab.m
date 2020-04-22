%% 1.) Images

clc, clear
cd('D:\THESIS\IMAGES EXPERIMENT, 336');
load('img1.mat');
images = img;
clear img
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % (400, 400, 3, 336) 
images = uint8(imresize(images, [224 224]));
disp(size(images)); % (224, 224, 3, 336) 


%% 2.) VGG-19

net = vgg19;
layers = net.Layers;
l = {}
for i = 1:length(layers)
    l{i,1} = layers(i).Name    
end
clear layers
layers_ind = [2 4 7 9 12 14 16 18 21 23 25 27 30 32 34 36 39 42 45];
layers = {}

for i = 1:length(layers_ind)
    index = layers_ind(i)
    layers{i, 1} = l{index}
end

clear l index layers_ind i 

%% 3.) Feature maps and correlations

for i = 1:numel(layers)
    
    disp(layers{i});
    feature_map = activations(net, images, layers{i});
    size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    %size(feature_map)  % data is now vectorized 

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
    
    cd("D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv small")
    co_small = corr(matrix);
    text = ["co_small_" + layers{i}]
    save(text, "co_small")
    % cos_small{i} = co_small

    %% 5.) visualize big corr / save
    
    cd("D:\thesis-scripts\Neural networks\VGG19\Experimental images\Conv big")
    co = corr(feature_map);
    text = ["co_" + layers{i}]
    save(text, "co")
    % cos{i} = co;
    
    clear feature_map body hand face man nonman tool chair co co_small matrix
    % imagesc(co);

end
