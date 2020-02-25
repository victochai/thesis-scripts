% Body vs. objects
% Face vs. objects
% Hand vs. objects

%% Images

clc, clear
cd('D:\THESIS\DATA\DATA. CAT12');
load('images_vgg19.mat');
images = images_vgg;
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % for VGG (224, 224, 3, 336) or (width, height, rgb_channels, n_samples)
clear images_vgg

%% Layers

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

%% Feature maps and correlations
% First 8 separately due to memory restrictions
% i = 8

for i = 9:numel(layers)
    
    feature_maps = {}
    disp(layers{i})

    % [height width n_channels m(n_samples)]
    feature_map = activations(net, images, layers{i});
    % vectorize: [everything_else m(n_samples)]
    feature_map = reshape(feature_map, [size(feature_map,1)*size(feature_map, 2)*size(feature_map, 3), size(feature_map, 4)]);
    feature_maps{i} = feature_map;
    clear feature_map

    bodies = {}
    hands = {}
    faces = {}
    objects = {}
    m_objects = {}

    bodies{i}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 1:48),2);
    hands{i}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 49:96),2);
    faces{i}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 97:144),2);
    objects{i, 1}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 145:192),2);
    objects{i, 2}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 193:240),2);
    objects{i, 3}(1:size(feature_maps{i}, 1), 1) = mean(feature_maps{i}(1:size(feature_maps{i}, 1), 241:288),2);
    mean_objects(:,:,1) = objects{i, 1};
    mean_objects(:,:,2) = objects{i, 2};
    mean_objects(:,:,3) = objects{i, 3};
    m_objects{i} = mean_objects;
    clear mean_objects
    mean_objects{i} = mean(m_objects{i}, 3);   
    clear m_objects

    % Correlations Objects
    corr_body_objects(i, 1) = corr(bodies{i}, mean_objects{i});
    corr_hand_objects(i, 1) = corr(hands{i}, mean_objects{i});
    corr_face_objects(i, 1) = corr(faces{i}, mean_objects{i});

    % Tools
    corr_body_tools(i, 1) = corr(bodies{i}, objects{i, 1});
    corr_hand_tools(i, 1) = corr(hands{i}, objects{i, 1});
    corr_face_tools(i, 1) = corr(faces{i}, objects{i, 1});

    % Man
    corr_body_man(i, 1) = corr(bodies{i}, objects{i, 2});
    corr_hand_man(i, 1) = corr(hands{i}, objects{i, 2});
    corr_face_man(i, 1) = corr(faces{i}, objects{i, 2});

    % Nman
    corr_body_nman(i, 1) = corr(bodies{i}, objects{i, 3});
    corr_hand_nman(i, 1) = corr(hands{i}, objects{i, 3});
    corr_face_nman(i, 1) = corr(faces{i}, objects{i, 3});

    clear feature_maps mean_objects objects faces bodies hands

end

%% Save results

correlations_objects_VGG = {corr_body_objects; corr_hand_objects; corr_face_objects}
correlations_tools_VGG = {corr_body_tools; corr_hand_tools; corr_face_tools}
correlations_man_VGG = {corr_body_man; corr_hand_man; corr_face_man}
correlations_nman_VGG = {corr_body_nman; corr_hand_nman; corr_face_nman}
correlations_VGG = {correlations_objects_VGG; correlations_tools_VGG; ...
                    correlations_man_VGG; correlations_nman_VGG}

save('correlations_VGG', 'correlations_VGG')

%% Plot (objects)

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('VGG19: body parts vs. objects correlations (19 layers)')
plot(corr_body_objects, '-o')
hold on
plot(corr_hand_objects, '-o')
hold on
plot(corr_face_objects, '-o')
legend('Bodies vs. objects', 'Hands vs. objects', 'Faces vs. objects')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (tools)

figure();
sgtitle('VGG19: body parts vs. tools correlations (19 layers)')
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
plot(corr_body_tools, '-o')
hold on
plot(corr_hand_tools, '-o')
hold on
plot(corr_face_tools, '-o')
legend('Bodies vs. tools', 'Hands vs. tools', 'Faces vs. tools')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (man)

figure();
sgtitle('VGG19: body parts vs. manipulable objects correlations (19 layers)')
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
plot(corr_body_man, '-o')
hold on
plot(corr_hand_man, '-o')
hold on
plot(corr_face_man, '-o')
legend('Bodies vs. man', 'Hands vs. man', 'Faces vs. man')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (Nman)

figure();
sgtitle('VGG19: body parts vs. non-manipulable objects correlations (19 layers)')
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
plot(corr_body_nman, '-o')
hold on
plot(corr_hand_nman, '-o')
hold on
plot(corr_face_nman, '-o')
legend('Bodies vs. Nman', 'Hands vs. Nman', 'Faces vs. Nman')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on

%% PLOT ALL

clc, clear
load('correlations_VGG.mat');
correlations_objects_VGG = correlations_VGG{1, 1}
correlations_tools_VGG = correlations_VGG{2, 1}
correlations_man_VGG = correlations_VGG{3, 1}
correlations_nman_VGG = correlations_VGG{4, 1}

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('VGG19: body parts vs. all objects correlations (19 layers)')
% First subplot
subplot(2,2,1)
plot(correlations_objects_VGG{1, 1}, '-o')
hold on
plot(correlations_objects_VGG{2, 1}, '-o')
hold on
plot(correlations_objects_VGG{3, 1}, '-o')
ylabel('Correlations')
legend('Bodies vs. Objects', 'Hands vs. Objects', 'Faces vs. Objects')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on
% Second subplot
subplot(2,2,2)
plot(correlations_tools_VGG{1, 1}, '-o')
hold on
plot(correlations_tools_VGG{2, 1}, '-o')
hold on
plot(correlations_tools_VGG{3, 1}, '-o')
ylabel('Correlations')
legend('Bodies vs. Tools', 'Hands vs. Tools', 'Faces vs. Tools')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on
% Third subplot
subplot(2,2,3)
plot(correlations_man_VGG{1, 1}, '-o')
hold on
plot(correlations_man_VGG{2, 1}, '-o')
hold on
plot(correlations_man_VGG{3, 1}, '-o')
ylabel('Correlations')
legend('Bodies vs. Man', 'Hands vs. Man', 'Faces vs. Man')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on
% Forth subplot
subplot(2,2,4)
plot(correlations_nman_VGG{1, 1}, '-o')
hold on
plot(correlations_nman_VGG{2, 1}, '-o')
hold on
plot(correlations_nman_VGG{3, 1}, '-o')
ylabel('Correlations')
legend('Bodies vs. Nman', 'Hands vs. Nman', 'Faces vs. Nman')
ylim([0 1])
xlim([1 19])
xticks([1:19])
xticklabels(layers)
xtickangle(45)
grid on
