%% run_patchwarp_across.m
% Executes PatchWarp across-session registration using the exact positional
% signatures from ryhattori/PatchWarp/patchwarp_across_sessions_demo.m

fprintf('Executing on MATLAB %s\n', version('-release'));

%% 1. Default Fallbacks (if run standalone inside MATLAB without Python wrapper)
if ~exist('repo_root', 'var')
    repo_root = 'C:\Users\jacob\Documents\dadarlat_labs';
end
if ~exist('mouse_id', 'var')
    mouse_id = 'mouse_6416_1R1L';
end
if ~exist('session_names', 'var')
    session_names = {'day_1', 'day_2', 'day_3'};
end
if ~exist('target_name', 'var')
    target_name = 'day_2';
end

%% 2. Configure Directories & Add PatchWarp to Path
patchwarp_code_dir = fullfile(repo_root, 'external', 'PatchWarp', 'PatchWarp');
mouse_dir          = fullfile(repo_root, 'data', mouse_id);
input_dir          = fullfile(mouse_dir, 'patchwarp_prep');
output_dir         = fullfile(mouse_dir, 'patchwarp_results');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

if exist(patchwarp_code_dir, 'dir')
    addpath(genpath(patchwarp_code_dir));
else
    error('PatchWarp directory not found at: %s', patchwarp_code_dir);
end

%% 3. Official PatchWarp Across-Sessions Parameters (from patchwarp_across_sessions_demo.m)
transform1            = 'euclidean';           % 1st step whole-FOV registration
transform2            = 'affine';              % 2nd step subfield registration
warp_blocksize        = 6;                     % 6x6 grid of subfields
warp_overlap_pix_frac = 0.15;                  % Fraction of edge overlap between adjacent subfields
norm_radius           = 32;                    % Radius of local normalization filter (0 = off)

%% 4. Load Target Session
target_prep_file = fullfile(input_dir, sprintf('%s_patchwarp_prep.mat', target_name));
if ~exist(target_prep_file, 'file')
    error('Target prep file not found: %s', target_prep_file);
end
target_data = load(target_prep_file);

% PatchWarp expects image1_all (Target) as [Ly, Lx, 2] where ch1 = meanImg, ch2 = meanImgE
image1_all = cat(3, double(target_data.meanImg), double(target_data.meanImgE));

master_results = struct();
master_results.mouse_id       = mouse_id;
master_results.target_name    = target_name;
master_results.target_meanImg = target_data.meanImg;
master_results.target_masks   = target_data.roi_masks;
master_results.target_indices = target_data.roi_indices;

%% 5. Loop Through Source Sessions and Register to Target
for i = 1:length(session_names)
    s_name = session_names{i};
    if strcmp(s_name, target_name)
        continue; % Skip registering target to itself
    end
    
    fprintf('\n========================================\n');
    fprintf('Registering %s (Source) -> %s (Target)\n', s_name, target_name);
    fprintf('========================================\n');
    
    src_file = fullfile(input_dir, sprintf('%s_patchwarp_prep.mat', s_name));
    src_data = load(src_file);
    
    % image2_all (Source): [Ly, Lx, 2]
    image2_all = cat(3, double(src_data.meanImg), double(src_data.meanImgE));
    
    % Step 5A: Run PatchWarp Across-Sessions Registration
    t_start = tic;
    pw_res = patchwarp_across_sessions(...
        image1_all, image2_all, ...
        transform1, transform2, ...
        warp_blocksize, warp_overlap_pix_frac, norm_radius);
    fprintf('Warp field estimated in %.2f seconds.\n', toc(t_start));
    
    % Step 5B: Apply Warp to ROI Spatial Masks
    n_rois = size(src_data.roi_masks, 3);
    fprintf('Warping %d ROI footprints into %s coordinate space...\n', n_rois, target_name);
    
    % Inspect exact argument count of patchwarp_across_sessions_apply dynamically
    n_in = nargin('patchwarp_across_sessions_apply');
    if n_in == 2
        [masks_warp2, ~] = patchwarp_across_sessions_apply(double(src_data.roi_masks), pw_res);
    else
        % Pass norm_radius = 0 so binary masks are not bandpass-filtered
        [masks_warp2, ~] = patchwarp_across_sessions_apply(double(src_data.roi_masks), pw_res, 0);
    end
    
    warped_masks = uint8(masks_warp2 > 0.3);
    
    % Extract warped summary images from pw_res
    warped_meanImg  = pw_res.image2_warp2(:, :, 1);
    warped_meanImgE = pw_res.image2_warp2(:, :, 2);
    
    % Store in structure
    f_name = matlab.lang.makeValidName(s_name);
    master_results.(f_name).patchwarp_results = pw_res;
    master_results.(f_name).warped_meanImg    = warped_meanImg;
    master_results.(f_name).warped_meanImgE   = warped_meanImgE;
    master_results.(f_name).warped_masks      = warped_masks;
    master_results.(f_name).roi_indices       = src_data.roi_indices;
    
    % Step 5C: Save QC Overlay Figure
    fig = figure('Visible', 'off', 'Units', 'normalized', 'Position', [0.1, 0.1, 0.75, 0.4]);
    T_norm      = mat2gray(image1_all(:, :, 2));
    S_raw_norm  = mat2gray(image2_all(:, :, 2));
    S_warp_norm = mat2gray(warped_meanImgE);
    
    subplot(1, 2, 1);
    imshow(cat(3, S_raw_norm, T_norm, S_raw_norm));
    title(sprintf('Pre-Warp: %s (Mag) vs %s (Grn)', s_name, target_name), 'Interpreter', 'none');
    
    subplot(1, 2, 2);
    imshow(cat(3, S_warp_norm, T_norm, S_warp_norm));
    title(sprintf('Post-PatchWarp: %s (Mag) vs %s (Grn)', s_name, target_name), 'Interpreter', 'none');
    
    qc_path = fullfile(output_dir, sprintf('qc_overlay_%s_to_%s.png', s_name, target_name));
    saveas(fig, qc_path);
    close(fig);
    fprintf('Saved QC overlay: %s\n', qc_path);
end

%% 6. Save Final Results
out_mat = fullfile(output_dir, 'across_sessions_patchwarp_results.mat');
save(out_mat, '-struct', 'master_results', '-v7.3');
fprintf('\n✓ All sessions aligned! Saved to:\n  %s\n', out_mat);