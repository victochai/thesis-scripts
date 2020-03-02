%% Loading

clc, clear
main_dir = 'D:\THESIS\DATA\DATA. CAT12';
stats_dir2 = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS';
cd(stats_dir2);

subs = {'SUB01', 'SUB02', 'SUB03', 'SUB04', 'SUB05', 'SUB06', 'SUB07', ...
        'SUB08', 'SUB09', 'SUB10', 'SUB11', 'SUB12', 'SUB13', 'SUB14', ...
        'SUB15', 'SUB16', 'SUB17'}';
    
rois = {'anterior_big'; 'anterior_big_left.mldatx'; ...
    'anterior_big_right.mldatx'; 'POSTERIOR2'; ...
    'OP_all'; 'CALC_all'; 'OP_and_CALC_all'};

%% Decoding | ALL

config = cosmo_config();
decodings = zeros(17, 7)

for subject = 1:length(subs)
    for mask = 1:length(rois)
    
    data_path = ['D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\' subs{subject} '\SPM.mat'];
    mask_path = ['D:\THESIS\DATA\DATA. CAT12\masks2\' rois{mask} '.nii'];
    ds = cosmo_fmri_dataset(data_path, 'mask', mask_path);
    ds = cosmo_remove_useless_data(ds);
    ds.sa.targets = repmat((1:7)', length(unique(ds.sa.chunks)), 1);
    
    args = struct();
    args.classifier = @cosmo_classify_lda;
    measure = @cosmo_crossvalidation_measure;
    args.partitions = cosmo_nfold_partitioner(ds);
    decodings(subject, mask) = measure(ds, args).samples;
    clear args ds data_path mask_path
    
    end
end

clear mask subject
clear chance_level config h pi
clear measure

%% Testing for Significance

chance_level = 1/7
decodings = decodings - chance_level

ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};

% Tool vs. Man
[h(1, 1), pi(1, 1)] = ttest(decodings(:, 1));
[h(2, 1), pi(2, 1)] = ttest(decodings(:, 2));
[h(3, 1), pi(3, 1)] = ttest(decodings(:, 3));
[h(4, 1), pi(4, 1)] = ttest(decodings(:, 4));
[h(5, 1), pi(5, 1)] = ttest(decodings(:, 5));
[h(6, 1), pi(6, 1)] = ttest(decodings(:, 6));
[h(7, 1), pi(7, 1)] = ttest(decodings(:, 7));

T = table(ROIs, h, pi);
T.Properties.VariableNames = {'ROI' 'decodings H', 'decodings PI'}
% save("decodings_all", "T")

%% Decoding | OBJECTS

config = cosmo_config();
decodings_tool_man = zeros(17, 7)
decodings_tool_nman = zeros(17, 7)
decodings_man_nman = zeros(17, 7)

for subject = 1:length(subs)
    for mask = 1:length(rois)
    
    data_path = ['D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\' subs{subject} '\SPM.mat'];
    mask_path = ['D:\THESIS\DATA\DATA. CAT12\masks2\' rois{mask} '.nii'];
    ds = cosmo_fmri_dataset(data_path, 'mask', mask_path);
    ds = cosmo_remove_useless_data(ds);
    ds.sa.targets = repmat((1:7)', length(unique(ds.sa.chunks)), 1);
    args = struct();
    args.classifier = @cosmo_classify_lda;
    measure = @cosmo_crossvalidation_measure;

    % tools vs. man
    targets = [4 5];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_tool_man = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_tool_man);
    decodings_tool_man(subject, mask) = measure(ds_tool_man, args).samples;
    clear args.partitions msk

    % tool vs. nman
    targets = [4 6];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_tool_nman = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_tool_nman);
    decodings_tool_nman(subject, mask) = measure(ds_tool_nman, args).samples;
    clear args.partitions msk
    
    % man vs. nman
    targets = [5 6];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_man_nman = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_man_nman);
    decodings_man_nman(subject, mask) = measure(ds_man_nman, args).samples;
    clear args.partitions msk
    
    end
end

%% Test for Significance

decodings_tool_man = decodings_tool_man - .5
decodings_tool_nman = decodings_tool_nman - .5
decodings_man_nman = decodings_man_nman - .5

ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};

% Tool vs. Man
[h1(1, 1), pi1(1, 1)] = ttest(decodings_tool_man(:, 1));
[h1(2, 1), pi1(2, 1)] = ttest(decodings_tool_man(:, 2));
[h1(3, 1), pi1(3, 1)] = ttest(decodings_tool_man(:, 3));
[h1(4, 1), pi1(4, 1)] = ttest(decodings_tool_man(:, 4));
[h1(5, 1), pi1(5, 1)] = ttest(decodings_tool_man(:, 5));
[h1(6, 1), pi1(6, 1)] = ttest(decodings_tool_man(:, 6));
[h1(7, 1), pi1(7, 1)] = ttest(decodings_tool_man(:, 7));

% Tool vs. Nman
[h2(1, 1), pi2(1, 1)] = ttest(decodings_tool_nman(:, 1));
[h2(2, 1), pi2(2, 1)] = ttest(decodings_tool_nman(:, 2));
[h2(3, 1), pi2(3, 1)] = ttest(decodings_tool_nman(:, 3));
[h2(4, 1), pi2(4, 1)] = ttest(decodings_tool_nman(:, 4));
[h2(5, 1), pi2(5, 1)] = ttest(decodings_tool_nman(:, 5));
[h2(6, 1), pi2(6, 1)] = ttest(decodings_tool_nman(:, 6));
[h2(7, 1), pi2(7, 1)] = ttest(decodings_tool_nman(:, 7));

% Man vs. Nman
[h3(1, 1), pi3(1, 1)] = ttest(decodings_man_nman(:, 1));
[h3(2, 1), pi3(2, 1)] = ttest(decodings_man_nman(:, 2));
[h3(3, 1), pi3(3, 1)] = ttest(decodings_man_nman(:, 3));
[h3(4, 1), pi3(4, 1)] = ttest(decodings_man_nman(:, 4));
[h3(5, 1), pi3(5, 1)] = ttest(decodings_man_nman(:, 5));
[h3(6, 1), pi3(6, 1)] = ttest(decodings_man_nman(:, 6));
[h3(7, 1), pi3(7, 1)] = ttest(decodings_man_nman(:, 7));

T = table(ROIs, h1, h2, h3, pi1, pi2, pi3);
T.Properties.VariableNames = {'ROI', 'tool vs. man H', 'tool vs. nman H', ...
    'man vs. nman H', 'tool vs. man PI', 'tool vs. nman PI', 'man vs. nman PI'}
% save("decodings_objects", "T")

%% Decoding | ALL | PREDICTIONS

config = cosmo_config();
confusion_matrix = zeros(17, 7, 7, 7)

for subject = 1:length(subs)
    for mask = 1:length(rois)
    
    data_path = ['D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\' subs{subject} '\SPM.mat'];
    mask_path = ['D:\THESIS\DATA\DATA. CAT12\masks2\' rois{mask} '.nii'];
    ds = cosmo_fmri_dataset(data_path, 'mask', mask_path);
    ds = cosmo_remove_useless_data(ds);
    ds.sa.targets = repmat((1:7)', length(unique(ds.sa.chunks)), 1);
    
    args = struct();
    args.output = 'predictions'
    args.classifier = @cosmo_classify_lda;
    measure = @cosmo_crossvalidation_measure;
    args.partitions = cosmo_nfold_partitioner(ds);
    decodings = measure(ds, args);
    confusion_matrix(subject, :, :, mask) = cosmo_confusion_matrix(decodings);
    clear args ds data_path mask_path decodings
    
    end
end

%% Visualize the results

confusion_matrix_mean = zeros(7,7,7)
size(confusion_matrix)
confusion_matrix_mean(:,:,1) = squeeze(mean(confusion_matrix(:,:,:,1), 1))
confusion_matrix_mean(:,:,2) = squeeze(mean(confusion_matrix(:,:,:,2), 1))
confusion_matrix_mean(:,:,3) = squeeze(mean(confusion_matrix(:,:,:,3), 1))
confusion_matrix_mean(:,:,4) = squeeze(mean(confusion_matrix(:,:,:,4), 1))
confusion_matrix_mean(:,:,5) = squeeze(mean(confusion_matrix(:,:,:,5), 1))
confusion_matrix_mean(:,:,6) = squeeze(mean(confusion_matrix(:,:,:,6), 1))
confusion_matrix_mean(:,:,7) = squeeze(mean(confusion_matrix(:,:,:,7), 1))

subplot(2,4,1)
imagesc(confusion_matrix_mean(:,:,1))
subplot(2,4,2)
imagesc(confusion_matrix_mean(:,:,2))
subplot(2,4,3)
imagesc(confusion_matrix_mean(:,:,3))
subplot(2,4,4)
imagesc(confusion_matrix_mean(:,:,4))
subplot(2,4,5)
imagesc(confusion_matrix_mean(:,:,5))
subplot(2,4,6)
imagesc(confusion_matrix_mean(:,:,6))
subplot(2,4,7)
imagesc(confusion_matrix_mean(:,:,7))

save("decodings_all_confusion_matrix","confusion_matrix_mean")

%% Decoding | OBJECTS | PREDICTIONS

config = cosmo_config();
confusion_matrix = zeros(17, 3, 3, 7)

for subject = 1:length(subs)
    for mask = 1:length(rois)
    
    data_path = ['D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\' subs{subject} '\SPM.mat'];
    mask_path = ['D:\THESIS\DATA\DATA. CAT12\masks2\' rois{mask} '.nii'];
    ds = cosmo_fmri_dataset(data_path, 'mask', mask_path);
    ds = cosmo_remove_useless_data(ds);
    ds.sa.targets = repmat((1:7)', length(unique(ds.sa.chunks)), 1);
   
    args = struct();
    args.output = 'predictions'
    args.classifier = @cosmo_classify_lda;
    measure = @cosmo_crossvalidation_measure;

    targets = [4 5 6];
    msk = cosmo_match(ds.sa.targets, targets);
    ds = cosmo_slice(ds, msk);    
    
    args.partitions = cosmo_nfold_partitioner(ds);
    decodings = measure(ds, args);
    confusion_matrix(subject, :, :, mask) = cosmo_confusion_matrix(decodings);
    clear args ds data_path mask_path decodings
    
    end
end

%% Test for significance (weird results)

diagonals = zeros(7, 17)
off_diagonals = zeros(7, 17)

for roi = 1:7
    
    a = confusion_matrix(:,:,:,roi)
    
    for subject = 1:17
        
        diagonals(roi, subject) = mean(diag(squeeze(a(subject, :, :))))
        off_diagonals(roi, subject) = mean(nonzeros(triu(squeeze(a(subject, :, :)), 1) + tril(squeeze(a(subject, :, :)), -1)));
        
    end
    
    clear a
end

[h(1, 1), pi(1, 1)] = ttest2(diagonals(:, 1), off_diagonals(:, 1));
[h(2, 1), pi(2, 1)] = ttest2(diagonals(:, 2), off_diagonals(:, 2));
[h(3, 1), pi(3, 1)] = ttest2(diagonals(:, 3), off_diagonals(:, 3));
[h(4, 1), pi(4, 1)] = ttest2(diagonals(:, 4), off_diagonals(:, 4));
[h(5, 1), pi(5, 1)] = ttest2(diagonals(:, 5), off_diagonals(:, 5));
[h(6, 1), pi(6, 1)] = ttest2(diagonals(:, 6), off_diagonals(:, 6));
[h(7, 1), pi(7, 1)] = ttest2(diagonals(:, 7), off_diagonals(:, 7));

ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};
T = table(ROIs, h, pi);

%% Visualize the results

confusion_matrix_mean = zeros(3,3,7)
size(confusion_matrix)
confusion_matrix_mean(:,:,1) = squeeze(mean(confusion_matrix(:,:,:,1), 1))
confusion_matrix_mean(:,:,2) = squeeze(mean(confusion_matrix(:,:,:,2), 1))
confusion_matrix_mean(:,:,3) = squeeze(mean(confusion_matrix(:,:,:,3), 1))
confusion_matrix_mean(:,:,4) = squeeze(mean(confusion_matrix(:,:,:,4), 1))
confusion_matrix_mean(:,:,5) = squeeze(mean(confusion_matrix(:,:,:,5), 1))
confusion_matrix_mean(:,:,6) = squeeze(mean(confusion_matrix(:,:,:,6), 1))
confusion_matrix_mean(:,:,7) = squeeze(mean(confusion_matrix(:,:,:,7), 1))

subplot(2,4,1)
imagesc(confusion_matrix_mean(:,:,1))
subplot(2,4,2)
imagesc(confusion_matrix_mean(:,:,2))
subplot(2,4,3)
imagesc(confusion_matrix_mean(:,:,3))
subplot(2,4,4)
imagesc(confusion_matrix_mean(:,:,4))
subplot(2,4,5)
imagesc(confusion_matrix_mean(:,:,5))
subplot(2,4,6)
imagesc(confusion_matrix_mean(:,:,6))
subplot(2,4,7)
imagesc(confusion_matrix_mean(:,:,7))

save("decodings_objects_confusion_matrix","confusion_matrix_mean")

%% %% Decoding | OBJECTS | PREDICTIONS

config = cosmo_config();
decodings_tool_man = zeros(17, 2, 2, 7)
decodings_tool_nman = zeros(17, 2, 2, 7)
decodings_man_nman = zeros(17, 2, 2, 7)

for subject = 1:length(subs)
    for mask = 1:length(rois)
    
    data_path = ['D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\' subs{subject} '\SPM.mat'];
    mask_path = ['D:\THESIS\DATA\DATA. CAT12\masks2\' rois{mask} '.nii'];
    ds = cosmo_fmri_dataset(data_path, 'mask', mask_path);
    ds = cosmo_remove_useless_data(ds);
    ds.sa.targets = repmat((1:7)', length(unique(ds.sa.chunks)), 1);
    args = struct();
    args.classifier = @cosmo_classify_lda;
    measure = @cosmo_crossvalidation_measure;
    args.output = 'predictions'

    % tools vs. man
    targets = [4 5];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_tool_man = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_tool_man);
    decodings_tool_man(subject, :, :, mask) = cosmo_confusion_matrix(measure(ds_tool_man, args));
    clear args.partitions msk

    % tool vs. nman
    targets = [4 6];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_tool_nman = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_tool_nman);
    decodings_tool_nman(subject, :, :, mask) = cosmo_confusion_matrix(measure(ds_tool_nman, args));
    clear args.partitions msk
    
    % man vs. nman
    targets = [5 6];
    msk = cosmo_match(ds.sa.targets, targets);
    ds_man_nman = cosmo_slice(ds, msk);
    args.partitions = cosmo_nfold_partitioner(ds_man_nman);
    decodings_man_nman(subject, :, :, mask) = cosmo_confusion_matrix(measure(ds_man_nman, args));
    clear args.partitions msk
    
    end
end

diagonals1 = zeros(7, 17)
off_diagonals1 = zeros(7, 17)
diagonals2 = zeros(7, 17)
off_diagonals2 = zeros(7, 17)
diagonals3 = zeros(7, 17)
off_diagonals3 = zeros(7, 17)

for roi = 1:7
    
    a1 = decodings_tool_man(:,:,:,roi)
    a2 = decodings_tool_nman(:,:,:,roi)
    a3 = decodings_man_nman(:,:,:,roi)
    
    for subject = 1:17
        
        diagonals1(roi, subject) = mean(diag(squeeze(a1(subject, :, :))))
        off_diagonals1(roi, subject) = mean(nonzeros(triu(squeeze(a1(subject, :, :)), 1) + tril(squeeze(a1(subject, :, :)), -1)));
        
        diagonals2(roi, subject) = mean(diag(squeeze(a2(subject, :, :))))
        off_diagonals2(roi, subject) = mean(nonzeros(triu(squeeze(a2(subject, :, :)), 1) + tril(squeeze(a2(subject, :, :)), -1)));
        
        diagonals3(roi, subject) = mean(diag(squeeze(a3(subject, :, :))))
        off_diagonals3(roi, subject) = mean(nonzeros(triu(squeeze(a3(subject, :, :)), 1) + tril(squeeze(a3(subject, :, :)), -1)));
        
    end
    
    clear a1 a2 a3
end

[h1(1, 1), pi1(1, 1)] = ttest2(diagonals1(:, 1), off_diagonals1(:, 1));
[h1(2, 1), pi1(2, 1)] = ttest2(diagonals1(:, 2), off_diagonals1(:, 2));
[h1(3, 1), pi1(3, 1)] = ttest2(diagonals1(:, 3), off_diagonals1(:, 3));
[h1(4, 1), pi1(4, 1)] = ttest2(diagonals1(:, 4), off_diagonals1(:, 4));
[h1(5, 1), pi1(5, 1)] = ttest2(diagonals1(:, 5), off_diagonals1(:, 5));
[h1(6, 1), pi1(6, 1)] = ttest2(diagonals1(:, 6), off_diagonals1(:, 6));
[h1(7, 1), pi1(7, 1)] = ttest2(diagonals1(:, 7), off_diagonals1(:, 7));

[h2(1, 1), pi2(1, 1)] = ttest2(diagonals2(:, 1), off_diagonals2(:, 1));
[h2(2, 1), pi2(2, 1)] = ttest2(diagonals2(:, 2), off_diagonals2(:, 2));
[h2(3, 1), pi2(3, 1)] = ttest2(diagonals2(:, 3), off_diagonals2(:, 3));
[h2(4, 1), pi2(4, 1)] = ttest2(diagonals2(:, 4), off_diagonals2(:, 4));
[h2(5, 1), pi2(5, 1)] = ttest2(diagonals2(:, 5), off_diagonals2(:, 5));
[h2(6, 1), pi2(6, 1)] = ttest2(diagonals2(:, 6), off_diagonals2(:, 6));
[h2(7, 1), pi2(7, 1)] = ttest2(diagonals2(:, 7), off_diagonals2(:, 7));

[h3(1, 1), pi3(1, 1)] = ttest2(diagonals3(:, 1), off_diagonals3(:, 1));
[h3(2, 1), pi3(2, 1)] = ttest2(diagonals3(:, 2), off_diagonals3(:, 2));
[h3(3, 1), pi3(3, 1)] = ttest2(diagonals3(:, 3), off_diagonals3(:, 3));
[h3(4, 1), pi3(4, 1)] = ttest2(diagonals3(:, 4), off_diagonals3(:, 4));
[h3(5, 1), pi3(5, 1)] = ttest2(diagonals3(:, 5), off_diagonals3(:, 5));
[h3(6, 1), pi3(6, 1)] = ttest2(diagonals3(:, 6), off_diagonals3(:, 6));
[h3(7, 1), pi3(7, 1)] = ttest2(diagonals3(:, 7), off_diagonals3(:, 7));

ROIs = {'ITG + ant. IOG'; 'ITG + ant. IOG (left)'; ...
    'ITG + ant. IOG (right)'; 'Post. IOG'; 'Occipital pole'; 'Calc. cortex'; 'Occip. pole + calc. cortex'};
T = table(ROIs, h1, h2, h3, pi1, pi2, pi3);

T.Properties.VariableNames = {'ROI', 'tool vs. man H', 'tool vs. nman H', ...
    'man vs. nman H', 'tool vs. man PI', 'tool vs. nman PI', 'man vs. nman P'}
