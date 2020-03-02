%% Loading 

clc,clear 
cd("C:\Users\victo\Desktop\thesis-scripts\Brain\Brain representations")
load("anterior_big_MATRIX.mat")
load("anterior_left.mat")
load("anterior_right.mat")
load("OP_MATRIX.mat")
load("CALC_MATRIX.mat")
load("OP_CALC_MATRIX.mat")
load("pos_res2_mvpa_MATRIX.mat")

%% T tests | ON-OFF diagonal

ON = zeros(17, 7);
OFF = zeros(17, 7);

for subject = 1:17
    
    % Anterior
    % ON
    % Columns:
    % 1. ITG + ant. IOG
    % 2. ITG + ant. IOG (left)
    % 3. ITG + ant. IOG (right)
    % 4. Post. IOG
    % 5. Occipital pole
    % 6. Calcarine cortex
    % 7. Occ. pole + calcarine cortex
    
    ON(subject, 1) = mean(diag(anterior_big_MATRIX(:, :, subject)));
    ON(subject, 2) = mean(diag(anterior_left(:, :, subject)));
    ON(subject, 3) = mean(diag(anterior_right(:, :, subject)));
    ON(subject, 4) = mean(diag(pos_res2_mvpa(:, :, subject)));
    ON(subject, 5) = mean(diag(OP_MATRIX(:, :, subject)));
    ON(subject, 6) = mean(diag(CALC_MATRIX(:, :, subject)));
    ON(subject, 7) = mean(diag(OP_CALC_MATRIX(:, :, subject)));
    
    % OFF
    OFF(subject, 1) = mean(nonzeros(triu(anterior_big_MATRIX(:,:,subject), 1) + tril(anterior_big_MATRIX(:,:,subject), -1)));
    OFF(subject, 2) = mean(nonzeros(triu(anterior_left(:,:,subject), 1) + tril(anterior_left(:,:,subject), -1)));
    OFF(subject, 3) = mean(nonzeros(triu(anterior_right(:,:,subject), 1) + tril(anterior_right(:,:,subject), -1)));
    OFF(subject, 4) = mean(nonzeros(triu(pos_res2_mvpa(:,:,subject), 1) + tril(pos_res2_mvpa(:,:,subject), -1)));
    OFF(subject, 5) = mean(nonzeros(triu(OP_MATRIX(:,:,subject), 1) + tril(OP_MATRIX(:,:,subject), -1)));
    OFF(subject, 6) = mean(nonzeros(triu(CALC_MATRIX(:,:,subject), 1) + tril(CALC_MATRIX(:,:,subject), -1)));
    OFF(subject, 7) = mean(nonzeros(triu(OP_CALC_MATRIX(:,:,subject), 1) + tril(OP_CALC_MATRIX(:,:,subject), -1)));
   
end

% T tests
ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};
T = table(ROIs);

[h(1, 1), pi(1, 1)] = ttest2(ON(:,1), OFF(:,1));
[h(2, 1), pi(2, 1)] = ttest2(ON(:,2), OFF(:,2));
[h(3, 1), pi(3, 1)] = ttest2(ON(:,3), OFF(:,3));
[h(4, 1), pi(4, 1)] = ttest2(ON(:,4), OFF(:,4));
[h(5, 1), pi(5, 1)] = ttest2(ON(:,5), OFF(:,5));
[h(6, 1), pi(6, 1)] = ttest2(ON(:,6), OFF(:,6));
[h(7, 1), pi(7, 1)] = ttest2(ON(:,7), OFF(:,7));

T = table(ROIs, h, pi);
% save("ON_OFF_TTEST", "T")

%% Hand vs. Tool VS. Hand. vs. Man VS. Hand vs. NMAN

% Columns:
% 1. Subject (17 IN TOTAL)
% 2.        % 1. OBJECTS
            % 2. BODY
            % 3. HAND
            % 4. TOOL
% 3.        % 1. ITG + ant. IOG
            % 2. ITG + ant. IOG (left)
            % 3. ITG + ant. IOG (right)
            % 4. Post. IOG
            % 5. Occipital pole
            % 6. Calcarine cortex
            % 7. Occ. pole + calcarine cortex 

body_objects = zeros(17, 4, 7);
hand_objects = zeros(17, 4, 7);
face_objects = zeros(17, 4, 7);

for subject = 1:17
    
    %% Body vs. OBJECTS
    % ITG + Ant. IOG
    body_objects(subject, 2, 1) = mean([anterior_big_MATRIX(1, 4, subject) anterior_big_MATRIX(4, 1, subject)]);
    body_objects(subject, 3, 1) = mean([anterior_big_MATRIX(1, 5, subject) anterior_big_MATRIX(5, 1, subject)]);
    body_objects(subject, 4, 1) = mean([anterior_big_MATRIX(1, 6, subject) anterior_big_MATRIX(6, 1, subject)]);
    % ITG + Ant. IOG (left)
    body_objects(subject, 2, 2) = mean([anterior_left(1, 4, subject) anterior_left(4, 1, subject)]);
    body_objects(subject, 3, 2) = mean([anterior_left(1, 5, subject) anterior_left(5, 1, subject)]);
    body_objects(subject, 4, 2) = mean([anterior_left(1, 6, subject) anterior_left(6, 1, subject)]);
    % ITG + Ant. IOG (right)
    body_objects(subject, 2, 3) = mean([anterior_right(1, 4, subject) anterior_right(4, 1, subject)]);
    body_objects(subject, 3, 3) = mean([anterior_right(1, 5, subject) anterior_right(5, 1, subject)]);
    body_objects(subject, 4, 3) = mean([anterior_right(1, 6, subject) anterior_right(6, 1, subject)]);    
    % Post. IOG
    body_objects(subject, 2, 4) = mean([pos_res2_mvpa(1, 4, subject) pos_res2_mvpa(4, 1, subject)]);
    body_objects(subject, 3, 4) = mean([pos_res2_mvpa(1, 5, subject) pos_res2_mvpa(5, 1, subject)]);
    body_objects(subject, 4, 4) = mean([pos_res2_mvpa(1, 6, subject) pos_res2_mvpa(6, 1, subject)]);       
    % Occipital pole
    body_objects(subject, 2, 5) = mean([OP_MATRIX(1, 4, subject) OP_MATRIX(4, 1, subject)]);
    body_objects(subject, 3, 5) = mean([OP_MATRIX(1, 5, subject) OP_MATRIX(5, 1, subject)]);
    body_objects(subject, 4, 5) = mean([OP_MATRIX(1, 6, subject) OP_MATRIX(6, 1, subject)]);       
    % Calcarine cortex
    body_objects(subject, 2, 6) = mean([CALC_MATRIX(1, 4, subject) CALC_MATRIX(4, 1, subject)]);
    body_objects(subject, 3, 6) = mean([CALC_MATRIX(1, 5, subject) CALC_MATRIX(5, 1, subject)]);
    body_objects(subject, 4, 6) = mean([CALC_MATRIX(1, 6, subject) CALC_MATRIX(6, 1, subject)]);       
    % Occ. pole + calcarine cortex
    body_objects(subject, 2, 7) = mean([OP_CALC_MATRIX(1, 4, subject) OP_CALC_MATRIX(4, 1, subject)]);
    body_objects(subject, 3, 7) = mean([OP_CALC_MATRIX(1, 5, subject) OP_CALC_MATRIX(5, 1, subject)]);
    body_objects(subject, 4, 7) = mean([OP_CALC_MATRIX(1, 6, subject) OP_CALC_MATRIX(6, 1, subject)]);              
    
    % Body part vs. OBJECT
    body_objects(subject, 1, 1) = mean([anterior_big_MATRIX(1, 4, subject) anterior_big_MATRIX(4, 1, subject) ...
        anterior_big_MATRIX(1, 5, subject) anterior_big_MATRIX(5, 1, subject) ...
        anterior_big_MATRIX(1, 6, subject) anterior_big_MATRIX(6, 1, subject)]);
    body_objects(subject, 1, 2) = mean([anterior_left(1, 4, subject) anterior_left(4, 1, subject) ...
        anterior_left(1, 5, subject) anterior_left(5, 1, subject) ...
        anterior_left(1, 6, subject) anterior_left(6, 1, subject)]);    
    body_objects(subject, 1, 3) = mean([anterior_right(1, 4, subject) anterior_right(4, 1, subject) ...
        anterior_right(1, 5, subject) anterior_right(5, 1, subject) ...
        anterior_right(1, 6, subject) anterior_right(6, 1, subject)]);    
    body_objects(subject, 1, 4) = mean([pos_res2_mvpa(1, 4, subject) pos_res2_mvpa(4, 1, subject) ...
        pos_res2_mvpa(1, 5, subject) pos_res2_mvpa(5, 1, subject) ...
        pos_res2_mvpa(1, 6, subject) pos_res2_mvpa(6, 1, subject)]);     
    body_objects(subject, 1, 5) = mean([OP_MATRIX(1, 4, subject) OP_MATRIX(4, 1, subject) ...
        OP_MATRIX(1, 5, subject) OP_MATRIX(5, 1, subject) ...
        OP_MATRIX(1, 6, subject) OP_MATRIX(6, 1, subject)]);      
    body_objects(subject, 1, 6) = mean([CALC_MATRIX(1, 4, subject) CALC_MATRIX(4, 1, subject) ...
        CALC_MATRIX(1, 5, subject) CALC_MATRIX(5, 1, subject) ...
        CALC_MATRIX(1, 6, subject) CALC_MATRIX(6, 1, subject)]);    
    body_objects(subject, 1, 6) = mean([OP_CALC_MATRIX(1, 4, subject) OP_CALC_MATRIX(4, 1, subject) ...
        OP_CALC_MATRIX(1, 5, subject) OP_CALC_MATRIX(5, 1, subject) ...
        OP_CALC_MATRIX(1, 6, subject) OP_CALC_MATRIX(6, 1, subject)]);        

    
    %% HAND vs. OBJECTS
    % ITG + Ant. IOG
    hand_objects(subject, 2, 1) = mean([anterior_big_MATRIX(2, 4, subject) anterior_big_MATRIX(4, 2, subject)]);
    hand_objects(subject, 3, 1) = mean([anterior_big_MATRIX(2, 5, subject) anterior_big_MATRIX(5, 2, subject)]);
    hand_objects(subject, 4, 1) = mean([anterior_big_MATRIX(2, 6, subject) anterior_big_MATRIX(6, 2, subject)]);
    % ITG + Ant. IOG (left)
    hand_objects(subject, 2, 2) = mean([anterior_left(2, 4, subject) anterior_left(4, 2, subject)]);
    hand_objects(subject, 3, 2) = mean([anterior_left(2, 5, subject) anterior_left(5, 2, subject)]);
    hand_objects(subject, 4, 2) = mean([anterior_left(2, 6, subject) anterior_left(6, 2, subject)]);
    % ITG + Ant. IOG (right)
    hand_objects(subject, 2, 3) = mean([anterior_right(2, 4, subject) anterior_right(4, 2, subject)]);
    hand_objects(subject, 3, 3) = mean([anterior_right(2, 5, subject) anterior_right(5, 2, subject)]);
    hand_objects(subject, 4, 3) = mean([anterior_right(2, 6, subject) anterior_right(6, 2, subject)]);    
    % Post. IOG
    hand_objects(subject, 2, 4) = mean([pos_res2_mvpa(2, 4, subject) pos_res2_mvpa(4, 2, subject)]);
    hand_objects(subject, 3, 4) = mean([pos_res2_mvpa(2, 5, subject) pos_res2_mvpa(5, 2, subject)]);
    hand_objects(subject, 4, 4) = mean([pos_res2_mvpa(2, 6, subject) pos_res2_mvpa(6, 2, subject)]);       
    % Occipital pole
    hand_objects(subject, 2, 5) = mean([OP_MATRIX(2, 4, subject) OP_MATRIX(4, 2, subject)]);
    hand_objects(subject, 3, 5) = mean([OP_MATRIX(2, 5, subject) OP_MATRIX(5, 2, subject)]);
    hand_objects(subject, 4, 5) = mean([OP_MATRIX(2, 6, subject) OP_MATRIX(6, 2, subject)]);       
    % Calcarine cortex
    hand_objects(subject, 2, 6) = mean([CALC_MATRIX(2, 4, subject) CALC_MATRIX(4, 2, subject)]);
    hand_objects(subject, 3, 6) = mean([CALC_MATRIX(2, 5, subject) CALC_MATRIX(5, 2, subject)]);
    hand_objects(subject, 4, 6) = mean([CALC_MATRIX(2, 6, subject) CALC_MATRIX(6, 2, subject)]);       
    % Occ. pole + calcarine cortex
    hand_objects(subject, 2, 7) = mean([OP_CALC_MATRIX(2, 4, subject) OP_CALC_MATRIX(4, 2, subject)]);
    hand_objects(subject, 3, 7) = mean([OP_CALC_MATRIX(2, 5, subject) OP_CALC_MATRIX(5, 2, subject)]);
    hand_objects(subject, 4, 7) = mean([OP_CALC_MATRIX(2, 6, subject) OP_CALC_MATRIX(6, 2, subject)]);              
    
    % Body part vs. OBJECT
    face_objects(subject, 1, 1) = mean([anterior_big_MATRIX(2, 4, subject) anterior_big_MATRIX(4, 2, subject) ...
        anterior_big_MATRIX(2, 5, subject) anterior_big_MATRIX(5, 2, subject) ...
        anterior_big_MATRIX(2, 6, subject) anterior_big_MATRIX(6, 2, subject)]);
    face_objects(subject, 1, 2) = mean([anterior_left(2, 4, subject) anterior_left(4, 2, subject) ...
        anterior_left(2, 5, subject) anterior_left(5, 2, subject) ...
        anterior_left(2, 6, subject) anterior_left(6, 2, subject)]);    
    face_objects(subject, 1, 3) = mean([anterior_right(2, 4, subject) anterior_right(4, 2, subject) ...
        anterior_right(2, 5, subject) anterior_right(5, 2, subject) ...
        anterior_right(2, 6, subject) anterior_right(6, 2, subject)]);    
    face_objects(subject, 1, 4) = mean([pos_res2_mvpa(2, 4, subject) pos_res2_mvpa(4, 2, subject) ...
        pos_res2_mvpa(2, 5, subject) pos_res2_mvpa(5, 2, subject) ...
        pos_res2_mvpa(2, 6, subject) pos_res2_mvpa(6, 2, subject)]);     
    face_objects(subject, 1, 5) = mean([OP_MATRIX(2, 4, subject) OP_MATRIX(4, 2, subject) ...
        OP_MATRIX(2, 5, subject) OP_MATRIX(5, 2, subject) ...
        OP_MATRIX(2, 6, subject) OP_MATRIX(6, 2, subject)]);      
    face_objects(subject, 1, 6) = mean([CALC_MATRIX(2, 4, subject) CALC_MATRIX(4, 2, subject) ...
        CALC_MATRIX(2, 5, subject) CALC_MATRIX(5, 2, subject) ...
        CALC_MATRIX(2, 6, subject) CALC_MATRIX(6, 2, subject)]);    
    face_objects(subject, 1, 6) = mean([OP_CALC_MATRIX(2, 4, subject) OP_CALC_MATRIX(4, 2, subject) ...
        OP_CALC_MATRIX(2, 5, subject) OP_CALC_MATRIX(5, 2, subject) ...
        OP_CALC_MATRIX(2, 6, subject) OP_CALC_MATRIX(6, 2, subject)]);     
    
    %%  FACE vs. OBJECTS
    % ITG + Ant. IOG
    face_objects(subject, 2, 1) = mean([anterior_big_MATRIX(3, 4, subject) anterior_big_MATRIX(4, 3, subject)]);
    face_objects(subject, 3, 1) = mean([anterior_big_MATRIX(3, 5, subject) anterior_big_MATRIX(5, 3, subject)]);
    face_objects(subject, 4, 1) = mean([anterior_big_MATRIX(3, 6, subject) anterior_big_MATRIX(6, 3, subject)]);
    % ITG + Ant. IOG (left)
    face_objects(subject, 2, 2) = mean([anterior_left(3, 4, subject) anterior_left(4, 3, subject)]);
    face_objects(subject, 3, 2) = mean([anterior_left(3, 5, subject) anterior_left(5, 3, subject)]);
    face_objects(subject, 4, 2) = mean([anterior_left(3, 6, subject) anterior_left(6, 3, subject)]);
    % ITG + Ant. IOG (right)
    face_objects(subject, 2, 3) = mean([anterior_right(3, 4, subject) anterior_right(4, 3, subject)]);
    face_objects(subject, 3, 3) = mean([anterior_right(3, 5, subject) anterior_right(5, 3, subject)]);
    face_objects(subject, 4, 3) = mean([anterior_right(3, 6, subject) anterior_right(6, 3, subject)]);    
    % Post. IOG
    face_objects(subject, 2, 4) = mean([pos_res2_mvpa(3, 4, subject) pos_res2_mvpa(4, 3, subject)]);
    face_objects(subject, 3, 4) = mean([pos_res2_mvpa(3, 5, subject) pos_res2_mvpa(5, 3, subject)]);
    face_objects(subject, 4, 4) = mean([pos_res2_mvpa(3, 6, subject) pos_res2_mvpa(6, 3, subject)]);       
    % Occipital pole
    face_objects(subject, 2, 5) = mean([OP_MATRIX(3, 4, subject) OP_MATRIX(4, 3, subject)]);
    face_objects(subject, 3, 5) = mean([OP_MATRIX(3, 5, subject) OP_MATRIX(5, 3, subject)]);
    face_objects(subject, 4, 5) = mean([OP_MATRIX(3, 6, subject) OP_MATRIX(6, 3, subject)]);       
    % Calcarine cortex
    face_objects(subject, 2, 6) = mean([CALC_MATRIX(3, 4, subject) CALC_MATRIX(4, 3, subject)]);
    face_objects(subject, 3, 6) = mean([CALC_MATRIX(3, 5, subject) CALC_MATRIX(5, 3, subject)]);
    face_objects(subject, 4, 6) = mean([CALC_MATRIX(3, 6, subject) CALC_MATRIX(6, 3, subject)]);       
    % Occ. pole + calcarine cortex
    face_objects(subject, 2, 7) = mean([OP_CALC_MATRIX(3, 4, subject) OP_CALC_MATRIX(4, 3, subject)]);
    face_objects(subject, 3, 7) = mean([OP_CALC_MATRIX(3, 5, subject) OP_CALC_MATRIX(5, 3, subject)]);
    face_objects(subject, 4, 7) = mean([OP_CALC_MATRIX(3, 6, subject) OP_CALC_MATRIX(6, 3, subject)]);              
    
    % Body part vs. OBJECT
    face_objects(subject, 1, 1) = mean([anterior_big_MATRIX(3, 4, subject) anterior_big_MATRIX(4, 3, subject) ...
        anterior_big_MATRIX(3, 5, subject) anterior_big_MATRIX(5, 3, subject) ...
        anterior_big_MATRIX(3, 6, subject) anterior_big_MATRIX(6, 3, subject)]);
    face_objects(subject, 1, 2) = mean([anterior_left(3, 4, subject) anterior_left(4, 3, subject) ...
        anterior_left(3, 5, subject) anterior_left(5, 3, subject) ...
        anterior_left(3, 6, subject) anterior_left(6, 3, subject)]);    
    face_objects(subject, 1, 3) = mean([anterior_right(3, 4, subject) anterior_right(4, 3, subject) ...
        anterior_right(3, 5, subject) anterior_right(5, 3, subject) ...
        anterior_right(3, 6, subject) anterior_right(6, 3, subject)]);    
    face_objects(subject, 1, 4) = mean([pos_res2_mvpa(3, 4, subject) pos_res2_mvpa(4, 3, subject) ...
        pos_res2_mvpa(3, 5, subject) pos_res2_mvpa(5, 3, subject) ...
        pos_res2_mvpa(3, 6, subject) pos_res2_mvpa(6, 3, subject)]);     
    face_objects(subject, 1, 5) = mean([OP_MATRIX(3, 4, subject) OP_MATRIX(4, 3, subject) ...
        OP_MATRIX(3, 5, subject) OP_MATRIX(5, 3, subject) ...
        OP_MATRIX(3, 6, subject) OP_MATRIX(6, 3, subject)]);      
    face_objects(subject, 1, 6) = mean([CALC_MATRIX(3, 4, subject) CALC_MATRIX(4, 3, subject) ...
        CALC_MATRIX(3, 5, subject) CALC_MATRIX(5, 3, subject) ...
        CALC_MATRIX(3, 6, subject) CALC_MATRIX(6, 3, subject)]);    
    face_objects(subject, 1, 6) = mean([OP_CALC_MATRIX(3, 4, subject) OP_CALC_MATRIX(4, 3, subject) ...
        OP_CALC_MATRIX(3, 5, subject) OP_CALC_MATRIX(5, 3, subject) ...
        OP_CALC_MATRIX(3, 6, subject) OP_CALC_MATRIX(6, 3, subject)]);     
           
end

% T tests
ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};
T = table(ROIs);

% TODO: significance tests

T = table(ROIs, h, pi);
save("ON_OFF_TTEST", "T")
