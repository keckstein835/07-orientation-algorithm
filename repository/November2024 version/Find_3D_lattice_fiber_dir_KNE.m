% File for determining orientation vectors spatially for each voxel in a full TIFF (or DICOM, etc.) stack
% This main file is part of the 3D Lattice Fiber Tracking Toolbox (Hey that's a cool name)

% Here, we need to take the image, crop it ?? actually idk yet
clear all
close all


%% User inputs
addpath('functions'); % Add the functions folder to the path

do_preset_filepath = 0; % Set to 1 to use preset filepaths, 0 to manually select files
vox_orientation_spacing = 2; % Adjust this value as needed; this sets resolution of orientation vectors in voxels
cropWidth_voxels = 20; % Default neighborhood size in voxels; 20 works well

output_filename = 'FFTO3D_20231227_DD.mat';

% secondary inputs that don't normally need adjusting:
voxel_size = 0.5; % Default voxel size in mm (typically 0.5mm for bruker scan)

do_primary_plots = false;
do_ignore_ROI_mask = true; % Set to true to ignore the ROI mask and process all voxels

% load image stack
if do_preset_filepath
    % % For the aligned lattice (which I manually resliced in ITK-snap with directions aligned to X-axis):
    %     file = 'resliced_T1.nii.gz';
    %     path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';
    %     mask_file = 'resliced_mask.nii.gz';
    %     mask_path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';

    % % For the 2024-02-09 scaled phantom:
    %     file = '02092024_Disk_Unscaled_Scaled_OldScaled_MRE_Disk_Scaled_E3_P1_Im01.dcm';
    %     path = 'Z:\MRE Gel Characterization\LumenX_lattice_phantom_MRE\Fri - 20240209\20240209_093911_02092024_MRE_Disk_Scaled_1_2\3 - T1_FLASH_EchoTime_4ms\pdata\1\dicom\';
    %     mask_file = 'T1_scaled_mask.nii.gz';
    %     mask_path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';

    % For the 2024-12-28 Xbox phantom:
        file = '12282023_Xbox_T1.nii.gz';
        path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';
        mask_file = '12282023_Xbox_T1_mask.nii.gz';
        mask_path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';
        % Load another mask called ROI_mask, if you want to just analyze an isolated volume to save time
        if ~do_ignore_ROI_mask
        ROI_mask_file = '12282023_Xbox_T1_ROI_mask.nii.gz';
        ROI_mask_path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';
        end


else
    % Prompt user to select a stack of DICOM files, a single DICOM volume, or a TIFF stack
    [file, path] = uigetfile({'*.*', 'All Files (*.*)';'*.dcm', 'DICOM Files (*.dcm)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'}, ...
                            'Select DICOM or TIFF Files', 'MultiSelect', 'on');

    % display file and path so that it may be copy and pasted back into code
    disp(file)  % display file
    disp(path)  % display path

    % Prompt user to select a mask file, accepting nii, tif, and tiff files, and ITK-snap mask files (nii.gz)
    [mask_file, mask_path] = uigetfile({'*.*', 'All Files (*.*)';'*.nii;*.nii.gz', 'NIfTI Files (*.nii, *.nii.gz)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'; '*.*', 'All Files (*.*)'}, ...
                                       'Select Mask File', 'MultiSelect', 'off');

    % Repeat for ROI mask
    if ~do_ignore_ROI_mask
    [ROI_mask_file, ROI_mask_path] = uigetfile({'*.*', 'All Files (*.*)';'*.nii;*.nii.gz', 'NIfTI Files (*.nii, *.nii.gz)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'; '*.*', 'All Files (*.*)'}, ...
                                       'Select ROI Mask File', 'MultiSelect', 'off');
    end
    % Display mask file and path so it may be copy and pasted back into code
    % disp(mask_file)  % display file
    % disp(mask_path)  % display path

end

    %% Import image
    [~, ~, ext] = fileparts(file);
    if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
        imageStack = niftiread(fullfile(path, file));
    else
        imageStack = getimageStack(file, path);
    end
 
    % Check the file extension and read the mask accordingly
    [~, ~, ext] = fileparts(mask_file);
    if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
        mask = niftiread(fullfile(mask_path, mask_file));
    elseif strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        mask = imread(fullfile(mask_path, mask_file));
    else
        error('Unsupported mask file format.');
    end
    mask = imbinarize(mask);

if ~do_ignore_ROI_mask
    % Check the file extension and read the ROI mask accordingly
    [~, ~, ext] = fileparts(ROI_mask_file);

    if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
        ROI_mask = niftiread(fullfile(ROI_mask_path, ROI_mask_file));
    elseif strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        ROI_mask = imread(fullfile(ROI_mask_path, ROI_mask_file));
    else
        error('Unsupported ROI mask file format.');
    end
    ROI_mask = imbinarize(ROI_mask);
    
else
    ROI_mask = ones(size(mask));
end

% if mask is different size than imageStack, return error
if size(mask) ~= size(imageStack)
    error('Mask and image stack are different sizes');
end 
% if imagestack is empty, return error
if isempty(imageStack)
    error('No image stack found.');
end    
% if mask is empty, return error
if isempty(mask)
    error('No mask found.');
end

% Display the selected filenames
disp(['Selected image stack file: ', fullfile(path, file)]);
disp(['Selected mask file: ', fullfile(mask_path, mask_file)]);
if ~do_ignore_ROI_mask
    disp(['Selected ROI mask file: ', fullfile(ROI_mask_path, ROI_mask_file)]);
end

% Get the size of the image stack
[dimX, dimY, dimZ] = size(imageStack);

%% Estimate time for user
neighborhood_coordinate = [-4, 6, -8] + [60, 60, 36]; % This is a good point for the X-box anyways
% Time the execution of Image_grid_3D_FFT_KNE for the selected voxel

% Prime the chamber here so we get an accurate time
[V_orientation_test,  FA_test] = Image_grid_3D_FFT_KNE(imageStack, mask, neighborhood_coordinate, cropWidth_voxels, voxel_size);
tic;
[V_orientation_test,  FA_test] = Image_grid_3D_FFT_KNE(imageStack, mask, neighborhood_coordinate, cropWidth_voxels, voxel_size);
elapsed_time = toc;

% Display the elapsed time
disp(['Time taken for Image_grid_3D_FFT_KNE at voxel position [', num2str(neighborhood_coordinate(1)), ', ', num2str(neighborhood_coordinate(2)), ', ', num2str(neighborhood_coordinate(3)), ']: ', num2str(elapsed_time,2), ' seconds']);

% Estimate the time to iterate through the entire volume with the specified voxel spacing
total_voxels = 0;
for x = 1:vox_orientation_spacing:dimX
    for y = 1:vox_orientation_spacing:dimY
        for z = 1:vox_orientation_spacing:dimZ
            if ROI_mask(x, y, z)
                total_voxels = total_voxels + 1;
            end
        end
    end
end

time_per_iteration = elapsed_time; % seconds
estimated_time_seconds = total_voxels * time_per_iteration;
estimated_time_minutes = estimated_time_seconds / 60;
estimated_time_hours = estimated_time_minutes / 60;
% Display the total number of voxels
disp(['Total number of voxels to iterate through: ', num2str(total_voxels)]);

% Display the estimated time
disp(['Estimated time to iterate through the entire volume: ', num2str(estimated_time_minutes,2), ' minutes (' num2str(estimated_time_hours,2), ' hours), given ' num2str(time_per_iteration,2) ' seconds per iteration.']);


%% Image grid 3D FFT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Define the voxel spacing for orientation calculation

% Initialize a cell array to store orientation vectors for each voxel
V_orientation_all = cell(dimX, dimY, dimZ);
FA_all = zeros(dimX, dimY, dimZ);

% Display that the iteration routine is beginning
disp('Beginning iteration routine through the entire volume...');
pause(2);

tic
iters_completed = 0;
% Iterate through the entire volume with the specified voxel spacing
for x = 1:vox_orientation_spacing:dimX
    for y = 1:vox_orientation_spacing:dimY
        for z = 1:vox_orientation_spacing:dimZ
            % Define the crop center location for the current voxel
            neighborhood_coordinate = [x, y, z];
            % Check if the current coordinate is within the ROI mask
            if ROI_mask(x, y, z)
                % Calculate the orientation vector for the current voxel
                [V_orientation, FA] = Image_grid_3D_FFT_KNE(imageStack, mask, neighborhood_coordinate, cropWidth_voxels, voxel_size);
                
                % Store the orientation vector in the cell array
                V_orientation_all{x, y, z} = V_orientation;
                % Store the FA value in a 3D array
                FA_all(x, y, z) = FA;
                iters_completed = iters_completed + 1;
            % else
            %     disp('Voxel outside of ROI mask. Empty cell');
            end

        end
    end
    % Display percent progress after each x iteration
    percent_complete = (iters_completed / total_voxels * 100);
    disp(['~~~~~~~~~~~~~Progress: ', num2str(percent_complete, '%.2f'), '% ~~~~~~~~~~~~~']);
end

%% Save to mat file
% Check if the file already exists and number the name if necessary

file_number = 1;
while exist(output_filename, 'file')
    output_filename = ['FFT_orientation_results_V_FA_', num2str(file_number), '.mat'];
    file_number = file_number + 1;
end






% Calculate and display the total time in minutes and hours
total_time_seconds = toc;
total_time_minutes = total_time_seconds / 60;
total_time_hours = total_time_minutes / 60;

disp(['Total time to run script: ', num2str(total_time_seconds, '%.2f'), ' seconds (', num2str(total_time_minutes, '%.2f'), ' minutes, ', num2str(total_time_hours, '%.2f'), ' hours).']);

% Compare total time to run script to estimated time
disp(['Estimated time: ', num2str(estimated_time_seconds, '%.2f'), ' seconds']);
disp(['Actual total time: ', num2str(total_time_seconds, '%.2f'), ' seconds']);


% Compare total time to run script to estimated time in percent
time_difference_percent = ((total_time_seconds - estimated_time_seconds) / estimated_time_seconds) * 100;
disp(['Time difference: ', num2str(time_difference_percent, '%.2f'), '%']);

if time_difference_percent > 0
    disp(['Actual total time exceeded the estimated time by ', num2str(time_difference_percent, '%.2f'), '%.']);
else
    disp(['Actual total time was less than or equal to the estimated time by ', num2str(abs(time_difference_percent), '%.2f'), '%.']);
end




% Save V_orientation_all to the determined .mat file
save(output_filename, 'V_orientation_all', 'FA_all','imageStack', 'mask', 'ROI_mask', 'vox_orientation_spacing', 'cropWidth_voxels', 'voxel_size', 'dimX', 'dimY', 'dimZ');

%% post-processing; open this function
% post_process_visualization(output_filename)
