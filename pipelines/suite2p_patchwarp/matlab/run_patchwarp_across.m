%% run_patchwarp_across.m (Validated for MATLAB R2022b)
% Registers Day 1 and Day 3 2-photon calcium imaging sessions onto Day 2
% using PatchWarp across-session piecewise affine registration.

clear; clc; close all;

%% 1. Version Guardrail
matlab_release = version('-release');
fprintf('Executing on MATLAB %s\n', matlab_release);
if str2double(matlab_release(1:4)) < 2022
    warning('PatchWarp functions were tested on R2022b. Older versions may trigger syntax deprecations.');
end

%% 2. Paths & Environment Setup
repo_root          = 'C:\Users\jacob\Documents\dadarlat_labs';
patchwarp_code_dir = fullfile(repo_root, 'external', 'PatchWarp', 'PatchWarp');
input_dir          = fullfile(repo_root, 'data', 'mouse_6416_1R1L', 'patchwarp_prep');
output_dir         = fullfile(input_dir, 'patchwarp_results');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if exist(patchwarp_code_dir, 'dir')
    addpath(genpath(patchwarp_code_dir));
else
    error('PatchWarp submodule directory not found at: %s', patchwarp_code_dir);
end

% Optional: Launch parallel pool if Parallel Computing Toolbox is available
try
    if isempty(gcp('nocreate'))
        parpool('threads'); % In R2022b, thread pools launch in <1 second with low memory overhead
    end
catch
    fprintf('Parallel pool not initialized; running single-threaded.\n');
end

%% 3. Load Session Preps
fprintf('Loading exported session preps from Python bridge...\n');
d1 = load(fullfile(input_dir, 'day1_patchwarp_prep.mat'));
d2 = load(fullfile(input_dir, 'day2_patchwarp_prep.mat')); % Target session (Day 2)
d3 = load(fullfile(input_dir, 'day3_patchwarp_prep.mat'));

target_session = d2;
target_img = double(target_session.meanImgE); 

sources = {d1, d3};
source_names = {'day1', 'day3'};

%% 4. Hyperparameters
pw_params = struct();
pw_params.n_levels       = 3;           
pw_params.transform_type = 'affine';    
pw_params.patch_size     = [128, 128];  
pw_params.patch_overlap  = 0.5;         
pw_params.regularization = 1e-2;        

%% 5. Register Sessions
results = struct();
results.matlab_release = matlab_release;
results.target_name    = 'day2';
results.target_meanImg = target_session.meanImg;
results.target_masks   = target_session.roi_masks;

for s = 1:length(sources)
    src = sources{s};
    s_name = source_names{s};
    fprintf('\n=== Aligning %s to Day 2 ===\n', s_name);
    
    source_img = double(src.meanImgE);
    
    % Step 5A: Estimate Across-Session Warp Field
    t_start = tic;
    [tform_global, tform_patches, warp_field] = patchwarp_across_sessions(...
        target_img, source_img, pw_params);
    fprintf('Registration calculated in %.2f s.\n', toc(t_start));
    
    % Step 5B: Apply Warps to Summary Images
    warped_meanImg  = patchwarp_across_sessions_apply(double(src.meanImg), warp_field);
    warped_meanImgE = patchwarp_across_sessions_apply(source_img, warp_field);
    
    % Step 5C: Apply Warps to ROI Masks
    n_rois = size(src.roi_masks, 3);
    fprintf('Transforming %d ROI footprints...\n', n_rois);
    warped_masks = zeros(size(src.roi_masks), 'uint8');
    
    for r = 1:n_rois
        m = double(src.roi_masks(:, :, r));
        m_warped = patchwarp_across_sessions_apply(m, warp_field, 'method', 'nearest');
        warped_masks(:, :, r) = uint8(m_warped > 0.3);
    end
    
    % Store
    results.(s_name).tform_global   = tform_global;
    results.(s_name).tform_patches  = tform_patches;
    results.(s_name).warp_field     = warp_field;
    results.(s_name).warped_meanImg = warped_meanImg;
    results.(s_name).warped_masks   = warped_masks;
    results.(s_name).roi_indices    = src.roi_indices;
    
    % Step 5D: QC Visualization Overlay (R2022b imshow compatible)
    fig = figure('Name', sprintf('QC: %s -> Day 2', s_name), ...
                 'Units', 'normalized', 'Position', [0.1, 0.1, 0.75, 0.4]);
    
    T_norm = mat2gray(target_img);
    S_raw_norm = mat2gray(source_img);
    S_warp_norm = mat2gray(warped_meanImgE);
    
    subplot(1, 2, 1);
    imshow(cat(3, S_raw_norm, T_norm, S_raw_norm));
    title(sprintf('Pre-Registration: %s (M) vs Day 2 (G)', s_name), 'Interpreter', 'none');
    
    subplot(1, 2, 2);
    imshow(cat(3, S_warp_norm, T_norm, S_warp_norm));
    title(sprintf('Post-PatchWarp: %s (M) vs Day 2 (G)', s_name), 'Interpreter', 'none');
    
    saveas(fig, fullfile(output_dir, sprintf('qc_overlay_%s_to_day2.png', s_name)));
end

%% 6. Save Formatted Outputs
out_mat = fullfile(output_dir, 'across_sessions_patchwarp_results.mat');
save(out_mat, '-struct', 'results', '-v7.3');
fprintf('\n✓ Processed successfully in MATLAB %s.\nSaved: %s\n', matlab_release, out_mat);