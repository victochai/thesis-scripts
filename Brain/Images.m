%% Loading images

clc,clear
cd("C:\Users\victo\Desktop\thesis-scripts\images")
load("images_mirrored.mat")
images = images_original;
size(images)
clear images_mirrorred

%% Making silhoettes

silhouettes = zeros(size(images, 1), 400, 400);
for i = 1:size(images, 1)
    img = squeeze(images(i,:,:));
    for x = 1:299
        for y = 1:299
            if img(x,y) ~= 255
                img(x,y) = 0;
            end
        end
    end
    silhouettes(i, :, :) = img(:, :);
    clear img x y
end

imshow(squeeze(silhouettes(159, :, :)))
silhoettes_reshaped = reshape(silhouettes, [size(images, 1), 400*400])';
co = corr(silhoettes_reshaped);
imagesc(co)

% CO small
% 168
body(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 1:24), 2);
hand(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 25:48), 2);
face(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 49:72), 2);
tool(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 73:96), 2);
man(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 97:120), 2);
nonman(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 121:144), 2);
chair(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 145:168), 2);
% 336
body(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 1:48), 2);
hand(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 49:96), 2);
face(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 97:144), 2);
tool(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 145:192), 2);
man(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 193:240), 2);
nonman(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 241:288), 2);
chair(1:size(silhoettes_reshaped, 1), 1) = mean(silhoettes_reshaped(1:size(silhoettes_reshaped, 1), 289:336), 2);

matrix(:, 1) = body;
matrix(:, 2) = hand;
matrix(:, 3) = face;
matrix(:, 4) = tool;
matrix(:, 5) = man;
matrix(:, 6) = nonman;
matrix(:, 7) = chair;

co_small = corr(matrix);
imagesc(co_small);

body_obj = zeros(4, 1)
hand_obj = zeros(4, 1)
face_obj = zeros(4, 1)

body_obj(1, 1) = corr(body, objects);
body_obj(2, 1) = corr(body, tool);
body_obj(3, 1) = corr(body, man);
body_obj(4, 1) = corr(body, nonman);

body_obj(1, 1) = corr(hand, objects);
hand_obj(2, 1) = corr(hand, tool);
hand_obj(3, 1) = corr(hand, man);
hand_obj(4, 1) = corr(hand, nonman);

face_obj(1, 1) = corr(face, objects);
face_obj(2, 1) = corr(face, tool);
face_obj(3, 1) = corr(face, man);
face_obj(4, 1) = corr(face, nonman);

correlations = {body_obj; hand_obj; face_obj; co; co_small}
cd("C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations");
save("images_correlations_for_experiment", "correlations")
