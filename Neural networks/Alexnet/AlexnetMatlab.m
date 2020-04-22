% Body vs. objects
% Face vs. objects
% Hand vs. objects

%% Images

clc, clear
cd('D:\THESIS\DATA\DATA. CAT12');
load('img1.mat');
images = img;
clear img
images = permute(images, [2, 3, 4, 1]);
disp(size(images)); % for Alexnet (227, 227, 3, 336) or (width, height, rgb_channels, n_samples)

%% Imresize if necessary

images = uint8(imresize(images, [227 227]));

%% Net

net = alexnet;
layers = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'};

%% Feature maps and correlations

for i = 1:numel(layers)
    
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

correlations_objects_AlEX = {corr_body_objects; corr_hand_objects; corr_face_objects}
correlations_tools_AlEX = {corr_body_tools; corr_hand_tools; corr_face_tools}
correlations_man_AlEX = {corr_body_man; corr_hand_man; corr_face_man}
correlations_nman_AlEX = {corr_body_nman; corr_hand_nman; corr_face_nman}
correlations_AlEX = {correlations_objects_AlEX; correlations_tools_AlEX; ...
                    correlations_man_AlEX; correlations_nman_AlEX}

save('correlations_ALEX_original', 'correlations_AlEX')

%% Plot (objects)

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('Alexnet: body parts vs. objects correlations (8 layers)')
plot(corr_body_objects, '-o')
hold on
plot(corr_hand_objects, '-o')
hold on
plot(corr_face_objects, '-o')
legend('Bodies vs. objects', 'Hands vs. objects', 'Faces vs. objects')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (tools)

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('Alexnet: body parts vs. tools correlations (8 layers)')
plot(corr_body_tools, '-o')
hold on
plot(corr_hand_tools, '-o')
hold on
plot(corr_face_tools, '-o')
legend('Bodies vs. tools', 'Hands vs. tools', 'Faces vs. tools')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (man)

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('Alexnet: body parts vs. manipulable objects correlations (8 layers)')
plot(corr_body_man, '-o')
hold on
plot(corr_hand_man, '-o')
hold on
plot(corr_face_man, '-o')
legend('Bodies vs. man', 'Hands vs. man', 'Faces vs. man')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on

%% Plot (Nman)

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('Alexnet: body parts vs. non-manipulable objects correlations (8 layers)')
plot(corr_body_nman, '-o')
hold on
plot(corr_hand_nman, '-o')
hold on
plot(corr_face_nman, '-o')
legend('Bodies vs. Nman', 'Hands vs. Nman', 'Faces vs. Nman')
xlabel('Layers')
ylabel('Correlations')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on

%% PLOT ALL

figure();
set(groot, 'DefaultAxesTickLabelInterpreter', 'none')
sgtitle('Alexnet: body parts vs. all objects correlations (8 layers)')
% First subplot
subplot(2,2,1)
plot(corr_body_objects, '-o')
hold on
plot(corr_hand_objects, '-o')
hold on
plot(corr_face_objects, '-o')
ylabel('Correlations')
legend('Bodies vs. Objects', 'Hands vs. Objects', 'Faces vs. Objects')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on
% Second subplot
subplot(2,2,2)
plot(corr_body_tools, '-o')
hold on
plot(corr_hand_tools, '-o')
hold on
plot(corr_face_tools, '-o')
ylabel('Correlations')
legend('Bodies vs. Tools', 'Hands vs. Tools', 'Faces vs. Tools')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on
% Third subplot
subplot(2,2,3)
plot(corr_body_man, '-o')
hold on
plot(corr_hand_man, '-o')
hold on
plot(corr_face_man, '-o')
ylabel('Correlations')
legend('Bodies vs. Man', 'Hands vs. Man', 'Faces vs. Man')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on
% Forth subplot
subplot(2,2,4)
plot(corr_body_nman, '-o')
hold on
plot(corr_hand_nman, '-o')
hold on
plot(corr_face_nman, '-o')
ylabel('Correlations')
legend('Bodies vs. Nman', 'Hands vs. Nman', 'Faces vs. Nman')
ylim([0 1])
xlim([1 8])
xticks([1:8])
xticklabels(layers)
xtickangle(45)
grid on
