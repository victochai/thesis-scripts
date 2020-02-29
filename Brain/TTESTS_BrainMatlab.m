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

%% T tests

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
save("ON_OFF_TTEST", "T")

%% Hand vs. Tool

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
    
    ON(subject, 1) = mean([anterior_big_MATRIX(1, 4, subject) anterior_big_MATRIX(4, 1, subject)]);
    ON(subject, 2) = mean([anterior_left(1, 4, subject) anterior_left(4, 1, subject)]);
    ON(subject, 3) = mean([anterior_right(1, 4, subject) anterior_right(4, 1, subject)]);
    ON(subject, 4) = mean([pos_res2_mvpa(1, 4, subject) anterior_right(4, 1, subject)]);
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
save("ON_OFF_TTEST", "T")


