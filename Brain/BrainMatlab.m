%% Correlation between body parts vs. objects

clc,clear 
cd("C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations")
load("anterior_big_MATRIX.mat")
load("CALC_MATRIX.mat")
load("pos_res2_mvpa_MATRIX.mat")
load("OP_MATRIX.mat")
labels = ["body"; "hand"; "face"; "tool"; "man"; "nman"; "chair"]
body = zeros(4, 4)
hand = zeros(4, 4)
face = zeros(4, 4)

%% Anterior (4)
% Body
body(2,4) = mean((anterior_big_MATRIX(1, 4, :) + anterior_big_MATRIX(4, 1, :))/2)
body(3,4) = mean((anterior_big_MATRIX(1, 5, :) + anterior_big_MATRIX(5, 1, :))/2)
body(4,4) = mean((anterior_big_MATRIX(1, 6, :) + anterior_big_MATRIX(6, 1, :))/2)

body(1,4) = mean((anterior_big_MATRIX(1, 4, :) + anterior_big_MATRIX(4, 1, :) + ...
    anterior_big_MATRIX(1, 5, :) + anterior_big_MATRIX(5, 1, :) + ...
    anterior_big_MATRIX(1, 6, :) + anterior_big_MATRIX(6, 1, :))/6)

% Hand
hand(2,4) = mean((anterior_big_MATRIX(2, 4, :) + anterior_big_MATRIX(4, 2, :))/2)
hand(3,4) = mean((anterior_big_MATRIX(2, 5, :) + anterior_big_MATRIX(5, 2, :))/2)
hand(4,4) = mean((anterior_big_MATRIX(2, 6, :) + anterior_big_MATRIX(6, 2, :))/2)

hand(1,4) = mean((anterior_big_MATRIX(2, 4, :) + anterior_big_MATRIX(4, 2, :) + ...
    anterior_big_MATRIX(2, 5, :) + anterior_big_MATRIX(5, 2, :) + ...
    anterior_big_MATRIX(2, 6, :) + anterior_big_MATRIX(6, 2, :))/6)

% Face
face(2,4) = mean((anterior_big_MATRIX(3, 4, :) + anterior_big_MATRIX(4, 3, :))/2)
face(3,4) = mean((anterior_big_MATRIX(3, 5, :) + anterior_big_MATRIX(5, 3, :))/2)
face(4,4) = mean((anterior_big_MATRIX(3, 6, :) + anterior_big_MATRIX(6, 3, :))/2)

face(1,4) = mean((anterior_big_MATRIX(3, 4, :) + anterior_big_MATRIX(4, 3, :) + ...
    anterior_big_MATRIX(3, 5, :) + anterior_big_MATRIX(5, 3, :) + ...
    anterior_big_MATRIX(3, 6, :) + anterior_big_MATRIX(6, 3, :))/6)

%% Calcarine (1)
% Body
body(2,1) = mean((CALC_MATRIX(1, 4, :) + CALC_MATRIX(4, 1, :))/2)
body(3,1) = mean((CALC_MATRIX(1, 5, :) + CALC_MATRIX(5, 1, :))/2)
body(4,1) = mean((CALC_MATRIX(1, 6, :) + CALC_MATRIX(6, 1, :))/2)

body(1,1) = mean((CALC_MATRIX(1, 4, :) + CALC_MATRIX(4, 1, :) + ...
    CALC_MATRIX(1, 5, :) + CALC_MATRIX(5, 1, :) + ...
    CALC_MATRIX(1, 6, :) + CALC_MATRIX(6, 1, :))/6)

% Hand
hand(2,1) = mean((CALC_MATRIX(2, 4, :) + CALC_MATRIX(4, 2, :))/2)
hand(3,1) = mean((CALC_MATRIX(2, 5, :) + CALC_MATRIX(5, 2, :))/2)
hand(4,1) = mean((CALC_MATRIX(2, 6, :) + CALC_MATRIX(6, 2, :))/2)

hand(1,1) = mean((CALC_MATRIX(2, 4, :) + CALC_MATRIX(4, 2, :) + ...
    CALC_MATRIX(2, 5, :) + CALC_MATRIX(5, 2, :) + ...
    CALC_MATRIX(2, 6, :) + CALC_MATRIX(6, 2, :))/6)

% Face
face(2,1) = mean((CALC_MATRIX(3, 4, :) + CALC_MATRIX(4, 3, :))/2)
face(3,1) = mean((CALC_MATRIX(3, 5, :) + CALC_MATRIX(5, 3, :))/2)
face(4,1) = mean((CALC_MATRIX(3, 6, :) + CALC_MATRIX(6, 3, :))/2)

face(1,1) = mean((CALC_MATRIX(3, 4, :) + CALC_MATRIX(4, 3, :) + ...
    CALC_MATRIX(3, 5, :) + CALC_MATRIX(5, 3, :) + ...
    CALC_MATRIX(3, 6, :) + CALC_MATRIX(6, 3, :))/6)

%% OP (2)
% Body
body(2,2) = mean((OP_MATRIX(1, 4, :) + OP_MATRIX(4, 1, :))/2)
body(3,2) = mean((OP_MATRIX(1, 5, :) + OP_MATRIX(5, 1, :))/2)
body(4,2) = mean((OP_MATRIX(1, 6, :) + OP_MATRIX(6, 1, :))/2)

body(1,2) = mean((OP_MATRIX(1, 4, :) + OP_MATRIX(4, 1, :) + ...
    OP_MATRIX(1, 5, :) + OP_MATRIX(5, 1, :) + ...
    OP_MATRIX(1, 6, :) + OP_MATRIX(6, 1, :))/6)

% Hand
hand(2,2) = mean((OP_MATRIX(2, 4, :) + OP_MATRIX(4, 2, :))/2)
hand(3,2) = mean((OP_MATRIX(2, 5, :) + OP_MATRIX(5, 2, :))/2)
hand(4,2) = mean((OP_MATRIX(2, 6, :) + OP_MATRIX(6, 2, :))/2)

hand(1,2) = mean((OP_MATRIX(2, 4, :) + OP_MATRIX(4, 2, :) + ...
    OP_MATRIX(2, 5, :) + OP_MATRIX(5, 2, :) + ...
    OP_MATRIX(2, 6, :) + OP_MATRIX(6, 2, :))/6)

% Face
face(2,2) = mean((OP_MATRIX(3, 4, :) + OP_MATRIX(4, 3, :))/2)
face(3,2) = mean((OP_MATRIX(3, 5, :) + OP_MATRIX(5, 3, :))/2)
face(4,2) = mean((OP_MATRIX(3, 6, :) + OP_MATRIX(6, 3, :))/2)

face(1,2) = mean((OP_MATRIX(3, 4, :) + OP_MATRIX(4, 3, :) + ...
    OP_MATRIX(3, 5, :) + OP_MATRIX(5, 3, :) + ...
    OP_MATRIX(3, 6, :) + OP_MATRIX(6, 3, :))/6)

%% Posterior (3)
% Body
body(2,3) = mean((pos_res2_mvpa(1, 4, :) + pos_res2_mvpa(4, 1, :))/2)
body(3,3) = mean((pos_res2_mvpa(1, 5, :) + pos_res2_mvpa(5, 1, :))/2)
body(4,3) = mean((pos_res2_mvpa(1, 6, :) + pos_res2_mvpa(6, 1, :))/2)

body(1,3) = mean((pos_res2_mvpa(1, 4, :) + pos_res2_mvpa(4, 1, :) + ...
    pos_res2_mvpa(1, 5, :) + pos_res2_mvpa(5, 1, :) + ...
    pos_res2_mvpa(1, 6, :) + pos_res2_mvpa(6, 1, :))/6)

% Hand
hand(2,3) = mean((pos_res2_mvpa(2, 4, :) + pos_res2_mvpa(4, 2, :))/2)
hand(3,3) = mean((pos_res2_mvpa(2, 5, :) + pos_res2_mvpa(5, 2, :))/2)
hand(4,3) = mean((pos_res2_mvpa(2, 6, :) + pos_res2_mvpa(6, 2, :))/2)

hand(1,3) = mean((pos_res2_mvpa(2, 4, :) + pos_res2_mvpa(4, 2, :) + ...
    pos_res2_mvpa(2, 5, :) + pos_res2_mvpa(5, 2, :) + ...
    pos_res2_mvpa(2, 6, :) + pos_res2_mvpa(6, 2, :))/6)

% Face
face(2,3) = mean((pos_res2_mvpa(3, 4, :) + pos_res2_mvpa(4, 3, :))/2)
face(3,3) = mean((pos_res2_mvpa(3, 5, :) + pos_res2_mvpa(5, 3, :))/2)
face(4,3) = mean((pos_res2_mvpa(3, 6, :) + pos_res2_mvpa(6, 3, :))/2)

face(1,3) = mean((pos_res2_mvpa(3, 4, :) + pos_res2_mvpa(4, 3, :) + ...
    pos_res2_mvpa(3, 5, :) + pos_res2_mvpa(5, 3, :) + ...
    pos_res2_mvpa(3, 6, :) + pos_res2_mvpa(6, 3, :))/6)

%% Plotting

figure(1)
sgtitle("Correlation between body parts and objects in differet brain regions ")
plot(body(1,:), '-o')
hold on
plot(hand(1,:), '-o')
hold on
plot(face(1,:), '-o')
legend("body", "hand", "face")
xticks([1 2 3 4])
xticklabels(["Calcarine cortex", "Occipital pole", "Posterior IOG", "ITG + Anterior IOG"])
xtickangle(45)

%% Save

save("hand_right", "hand")
save("body_right", "body")
save("face_right", "face")
