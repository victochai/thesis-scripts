%% MVPA / RDMs, clear code
%% subs, folders

clc, clear
main_dir = 'D:\THESIS\DATA\DATA. CAT12';
stats_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS';
mvpa_dir = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\MVPA (finals)';
cd(main_dir);

subs = {'SUB01', 'SUB02', 'SUB03', 'SUB04', 'SUB05', 'SUB06', 'SUB07', ...
        'SUB08', 'SUB09', 'SUB10', 'SUB11', 'SUB12', 'SUB13', 'SUB14', ...
        'SUB15', 'SUB16', 'SUB17'}';

rois = {'new'};

%% the loop / prepare the data / ewmvpa

for subject = 1:length(subs)
    
    for roi = 1:length(rois)
        
        % _____ VARIABLES WE NEED TO ADJUST _____
        
        % name of spm file of a particular subject
        spm_name = ([stats_dir '\' subs{subject} '\SPM.mat']);
        SPM_file = load(spm_name);
        
        % which ROI we use
        roi_name = ([main_dir '\ROIS. MatLab File\' rois{roi} '.mat']);
        load(roi_name);  % makeROI variable
        
        % where are beta files for this subject?
        beta_dir = ([stats_dir '\' subs{subject}]);
        cd(beta_dir);
        
        % do the model contains fixation condition? (0 - no, 1 - yes)
        fix = 0;
    
        % _____ PREPARATION OF THE ANALYSIS _____
        
        n_runs = str2num(SPM_file.SPM.xsDes.Number_of_sessions); 
        n_cond = str2num(SPM_file.SPM.xsDes.Trials_per_session(1,1:2)) - fix;
        n_rep = 100;  
        
        % calculate which beta IS for each run and condition
        % BETAS --> 1 RUN: body, hand, face, tool, mani, nman, chairs {7} +
        % 6 motion parameters
        % like that 7+6 * 8 = 104 + 8 CONSTANTS = 112 BETAS
        % so here in condToTake there only numbers of betas that are for
        % conditions and that makes sense
        % there are 56 of them. bcos 7 * 8 = 56. other stuff isn't importat
        % now
        condToTake = [];
        for run = 1:n_runs
            condToTake = [condToTake SPM_file.SPM.Sess(run).col(1,1:n_cond)];              
        end
        
        % load in ROI coordinates
        ROI.XYZ = makeROI.selected.anatXYZ;
        nrVoxels = size(ROI.XYZ, 2); 
        
        % cell of betas with zero values
        for i = 1:length(condToTake)
            a{i} = zeros(SPM_file.SPM.Vbeta(1).dim(1), SPM_file.SPM.Vbeta(1).dim(2), SPM_file.SPM.Vbeta(1).dim(3));
        end
        
        % read in all the volumes for betas in variable a
        for condition = 1:length(condToTake)
            imgfile = load_nii([SPM_file.SPM.Vbeta(condToTake(condition)).fname]);
            matrix = imgfile.img;
            a{condition} = matrix(:,:,:);
        end
        
       % make one big matrix. each column --> one BETA VALUE for a specific
       % run and condition. 7 * 8 = 56 of them 
        alltogether = zeros(nrVoxels, length(condToTake));
        for voxel = 1:nrVoxels
            telcond = 0;
            for condition = 1:length(condToTake);
                condSig = a{condition}(ROI.XYZ(1,voxel),ROI.XYZ(2,voxel),ROI.XYZ(3,voxel));
                telcond = telcond + 1;
                alltogether(voxel,telcond) = condSig;
            end
        end
        
        % rearrange the data. for each run, separate cell in ROI.PSC_all
        for i = 1:n_runs
            ROI.PSC_all{i} = alltogether(:,(i - 1) * n_cond + 1:(i - 1) * n_cond + n_cond);
        end
        
        % standardize with subtraction of mean across all conditons for each voxel
         for i = 1:n_runs
            for v = 1:nrVoxels
                ROI.PSC_all{i}(v,:) = ROI.PSC_all{i}(v,:) - mean(ROI.PSC_all{i}(v,:));
            end
         end
        
        % take out the voxels with NaN (should not be there with proper masking)
        telKept = 0;
        for v = 1:nrVoxels
            remove = 0;
            for r = 1:n_runs
                if abs(mean(ROI.PSC_all{r}(v,:))) < 25
                else  % mean higher than 25,  also happens if mean is NaN
                    remove = 1;
                end
            end
            if remove == 0
                telKept = telKept + 1;
                for r=1:n_runs
                    ROI.PSC_select{r}(telKept, 1:n_cond) = ROI.PSC_all{r}(v,:);
                end
            end
        end
         
        % how many voxels are left? how many training and testing-runs?
        nrVoxels = telKept;
        if mod(n_runs, 2) == 0
            nrRuns_training = n_runs/2;
            nrRuns_test = n_runs/2;
        elseif mod(n_runs, 2) == 1
            nrRuns_training = (n_runs + 1)/2;
            nrRuns_test = (n_runs - 1)/2;
        end
            
      
        % _____ MVPA _____
        % now we have 4 training + 4 testing runs
        % we have a variable ROI
        % ROI.PSC_select --> 
        % each run: 8 runs. each run: 7 conditions
        % each condition --> one column. every colums: 336 voxels with BETA
        % values for a specific ROI!
        
        for rep = 1:n_rep
            
            clear trainingSamples
            clear trainingLabels
            clear indTestSamples
            clear indTestLabels
            clear testSamples
            clear testLabels
                        
            for c1 = 1:n_cond - 1  % [1 2 3 4 5 6]
                
                for c2 = c1 + 1:n_cond  % [2 3 4 5 6 7] OR [3 4 5 6 7]
                                                            
                    % first runs in this list will be training runs
                    runOrder = randperm(n_runs);  % let's say [5 2 1 7 6 8 3 4]
                    
                    for r = 1:nrRuns_training  
                        trainingSamples(1:nrVoxels, r) = ROI.PSC_select{runOrder(r)}(:,c1);
                        trainingLabels(r) = 1;
                        trainingSamples(1:nrVoxels, nrRuns_training + r) = ROI.PSC_select{runOrder(r)}(:,c2);
                        trainingLabels(nrRuns_training + r) = 2;
                    end
                        
                    for r = 1:nrRuns_test 
                        indTestSamples(1:nrVoxels, r) = ROI.PSC_select{runOrder(nrRuns_training + r)}(:,c1);
                        indTestLabels(r) = 1;
                        indTestSamples(1:nrVoxels,nrRuns_test + r) = ROI.PSC_select{runOrder(nrRuns_training + r)}(:,c2);
                        indTestLabels(nrRuns_test + r) = 2;
                    end
                    
                    testSamples(1:nrVoxels,1) = mean(indTestSamples(1:nrVoxels,1:nrRuns_test),2);
                    testLabels(1) = 1;
                    testSamples(1:nrVoxels,2) = mean(indTestSamples(1:nrVoxels,nrRuns_test+1:nrRuns_test+nrRuns_test),2);
                    testLabels(2) = 2;
                        
                    % make test data from one  random run also as a control for the smoothing analysis
                    % testRun = randperm(nrRuns_test);  % for example [3 2 4 1]
                    % testSamplesOneRun(1:nrVoxels,1) = indTestSamples(1:nrVoxels,testRun(1));
                    % testLabelsOneRun(1) = 1;
                    % testSamplesOneRun(1:nrVoxels,2) = indTestSamples(1:nrVoxels,nrRuns_test+testRun(1));
                    % testLabelsOneRun(2) = 2;                 
                      
                    % now pairwise comparisons with correlations
                    % co = corrcoef(mean(trainingSamples(:,1:nrRuns_training),2) - mean(trainingSamples(:,nrRuns_training+1:2*nrRuns_training),2), testSamples(:,1) - testSamples(:,2));
                    % corrAll(c1,c2,rep) = co(2,1);
                            
                    % correlations between selectivity patterns (asymmetrical matrix)
                    co = corrcoef(mean(trainingSamples(:,1:nrRuns_training),2),testSamples(:,1));
                    symMatrix(c1,c1,rep) = co(2,1);
                    co = corrcoef(mean(trainingSamples(:,nrRuns_training+1:2*nrRuns_training),2),testSamples(:,2));
                    symMatrix(c2,c2,rep) = co(2,1);
                    co = corrcoef(mean(trainingSamples(:,1:nrRuns_training),2),testSamples(:,2));
                    symMatrix(c1,c2,rep) = co(2,1);
                    co = corrcoef(mean(trainingSamples(:,nrRuns_training+1:2*nrRuns_training),2),testSamples(:,1));
                    symMatrix(c2,c1,rep) = co(2,1);
                    
                end  % c2
            end  % c1
        end  % rep
        
        % SNOW and SAVE THE RESULTS OF MVPA
        results_mvpa = mean(symMatrix, 3); 
        % results_pairwise = mean(corrAll, 3);
        cd([mvpa_dir '\' rois{roi}]);
        save([subs{subject} '_' rois{roi} '_NEW_results_mvpa'],'results_mvpa')
        % save([subs{subject} '_' rois{roi} '_pairwise_corr'],'results_pairwise')
        clearvars -except subs rois roi subject stats_dir mvpa_dir main_dir
        
    end  % roi
end  % subject

%% loading / rearranging the results

clc, clear
main_dir = 'D:\THESIS\DATA\DATA. CAT12';
cd(main_dir);
subs = {'SUB01', 'SUB02', 'SUB03', 'SUB04', 'SUB05', 'SUB06', 'SUB07', ...
        'SUB08', 'SUB09', 'SUB10', 'SUB11', 'SUB12', 'SUB13', 'SUB14', ...
        'SUB15', 'SUB16', 'SUB17'}';

dir1 = 'D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\MVPA (finals)\new'

new = [];

for subject = 1:length(subs)
    
    name1 = ([dir1 '\' subs{subject} '_new_NEW_results_mvpa.mat'])
    load(name1);
    new(:, :, subject) = results_mvpa;
    clear results_mvpa
    
end

cd('D:\THESIS\DATA\DATA. CAT12\STATISTICS\OTHER STATS\MVPA (finals)')
save('new', 'new')
