%% inception V3

clc,clear
net = inceptionv3;
cd("C:\Users\victo\Desktop\thesis-scripts")
load("imagesV3.mat")
images = permute(imagesV3, [2, 3, 4, 1]);
disp(size(images));
clear imagesV3

%% Layers

layers = net.Layers;
l = {}

% deepNetworkDesigner
for i = 1:numel(layers)    
    l{i, 1} = layers(i).Name;        
end
clear layers

layers = {}
for i = 1:numel(l)
    a = l{i, 1}
    ans = regexp(a, "conv2d_\d")
    if ans == 1
        layers{i, 1} = l{i, 1}
    end       
end

layers = layers(~cellfun('isempty', layers))
clear a ans i l

%% For no concat 

% cd("C:\Users\victo\Desktop\nns_for_thesis")

layers{86}

for i = 86
    
    feature_map = activations(net, images, layers{i});
    % size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    size(feature_map)  % data is now vectorized 
end

%% For concat (2)

layers{87:88, 1}

for i = 87:88
    
    % Concat 2 f maps
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_map = vertcat(c{i-1,1},c{i,1});
clear i c

%% For concat (4)

layers{67:70, 1}

for i = 67:70
    
    % Concat 2 f maps
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_map = vertcat(c{i-3,1},c{i-2,1},c{i-1,1},c{i,1});
clear i c

%% For concat 1 (5)

layers{89:94, 1}

for i = 89:94
    
    % Concat 2 f maps
    layers{i}
    feature_map = activations(net, images, layers{i});
    size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    c{i,1} = feature_map;
    clear feature_map
    
end

feature_map = vertcat(c{i-4,1},c{i-3,1},c{i-2,1},c{i-1,1},c{i,1});
clear i c

%% Last layer

feature_map = activations(net, images, "predictions");
size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc
feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
size(feature_map)  % should be [h w ch m] for conv and [1 1 num.of.neurons m] for fc

%% F MAPS 

for i = 48
    
    % Co-s

    co = corr(feature_map);

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
    
    objects = mean([tool, man, nonman], 2);
    
    % Correlations Objects
    corr_body_objects(i, 1) = corr(body, objects);
    corr_hand_objects(i, 1) = corr(hand, objects);
    corr_face_objects(i, 1) = corr(face, objects);

    % Tools
    corr_body_tools(i, 1) = corr(body, tool);
    corr_hand_tools(i, 1) = corr(hand, tool);
    corr_face_tools(i, 1) = corr(face, tool);

    % Man
    corr_body_man(i, 1) = corr(body, man);
    corr_hand_man(i, 1) = corr(hand, man);
    corr_face_man(i, 1) = corr(face, man);

    % Nman
    corr_body_nman(i, 1) = corr(body, nonman);
    corr_hand_nman(i, 1) = corr(hand, nonman);
    corr_face_nman(i, 1) = corr(face, nonman);
    
    text = ["conv2d_" + i + "_pred"]
    save(text, "feature_map");
    text = ["co_" + "conv2d_" + i + "_pred"]
    save(text, "co");
    text = ["co_small_" + "conv2d_" + i + "_pred"]
    save(text, "co_small");
    
    clear matrix body face hand tool man nonman chair feature_map i co co_small
    
end

%% Save results

correlations_objects_INC = {corr_body_objects; corr_hand_objects; corr_face_objects}
correlations_tools_INC = {corr_body_tools; corr_hand_tools; corr_face_tools}
correlations_man_INC = {corr_body_man; corr_hand_man; corr_face_man}
correlations_nman_INC = {corr_body_nman; corr_hand_nman; corr_face_nman}
correlations_INC = {correlations_objects_INC; correlations_tools_INC; ...
                    correlations_man_INC; correlations_nman_INC}

save('correlations_INC', 'correlations_INC')
