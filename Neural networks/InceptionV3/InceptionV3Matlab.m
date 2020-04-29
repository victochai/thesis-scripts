%% inception V3

clc,clear
net = inceptionv3;

cd('D:\THESIS\IMAGES EXPERIMENT, 336');
load('img1.mat');
images = img;
clear img
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % (400, 400, 3, 336) 
images = uint8(imresize(images, [299 299]));
disp(size(images)); % (299, 299, 3, 336) 

%% layers

layers = {"conv2d_1", "predictions"}

for i = 1:2

feature_map = activations(net, images, layers{i});
size(feature_map)  
feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
size(feature_map)  % data is now vectorized 

co = corr(feature_map)

cd("D:\thesis-scripts\Neural networks\InceptionV3\Experimental images\First and last layers")
text = ["co_" + layers{i}]
save(text, "co")

clear co feature_map text

end
