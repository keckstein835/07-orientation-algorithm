% File for determining orientation vectors spatially for each voxel in a full TIFF (or DICOM, etc.) stack
% This main file is part of Kevin's new 3D Lattice Fiber Tracking Toolbox (Hey that's a cool name)
% Written by Kevin Eckstein 2024-11-4
% Edited by Kevin Eckstein 2024-12-27 to make it user-friendly

clear all
close all
addpath('functions'); % Add the functions folder to the path

%% [1] Step 1: Load image stack and mask

% Ensure the current directory is where this code is saved
script_fullpath = mfilename('fullpath');
[script_dir, ~, ~] = fileparts(script_fullpath);
cd(script_dir);

if isfile('recent_filepath.mat')
    load('recent_filepath.mat')
    %this has the most recent filepath, saved from previous run. Not required, but can be useful. 
else
    recent_file = 'empty';
    recent_path = 'empty';
end

disp('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');
disp('Current directory: ');
disp(pwd);
disp('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');

disp('Step 1: Load image stack and mask');
% Ask user if they want to load a T1 image
disp('Do you want to load a T1 image?')
disp('- Enter 1 to select T1 (.dcm, .nii, .tif)')
disp('- Enter 2 to use example preset filepath [default]')
disp(['- Enter 3 to use most recent filepath,   which is: ' fullfile(recent_path, recent_file)])

load_T1_image = input('');
if load_T1_image == 1
    % Prompt user to select a T1 image file
    [file, path] = uigetfile({'*.*', 'All Files (*.*)'; '*.dcm', 'DICOM Files (*.dcm)'; '*.nii;*.nii.gz', 'NIfTI Files (*.nii, *.nii.gz)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'}, ...
                             'Select T1 Image File', 'MultiSelect', 'off');

elseif load_T1_image == 3
    % Use the most recent file path
    file = recent_file;
    path = recent_path;
    disp(['Using most recent file: ', fullfile(path, file)]);
else
    % Use preset file-path "example" (can change for debugging and testing)
    % For the 2024-12-28 Xbox phantom:
    file = 'Example_12282023_Xbox_T1.nii.gz';
    path = [pwd '\Example images and masks 2025\'];

    % For the 2024-02-09 scaled phantom:
    % file = '02092024_Disk_Unscaled_Scaled_OldScaled_MRE_Disk_Scaled_E3_P1_Im01.dcm';
    % path = 'Z:\MRE Gel Characterization\LumenX_lattice_phantom_MRE\Fri - 20240209\20240209_093911_02092024_MRE_Disk_Scaled_1_2\3 - T1_FLASH_EchoTime_4ms\pdata\1\dicom\';

    % For the aligned lattice (which I manually resliced in ITK-snap with directions aligned to X-axis):
    % file = 'resliced_T1.nii.gz';
    % path = 'Z:\0_User_Folders\Kevin Eckstein\03 MRE and bioprinting\B Bioprinting and NLI\B5 Orientation vectors via FFT\images\';

    disp(['Using preset file: ', fullfile(path, file)]);
end

% Save the most recent file path for repeated use
recent_file = file;
recent_path = path;
save('recent_filepath.mat', 'recent_file', 'recent_path');

% Import image
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
    imageStack = niftiread(fullfile(path, file));
else
    imageStack = getimageStack(file, path);
end

imageStack = squeeze(imageStack); % Remove singleton dimensions

% Ask user if they want to load a mask, use a preset file mask, or forgo any mask
disp('Do you want to load a mask? (masks reduce computation time and can improve results near material boundaries)')
disp('- Enter 1 to select mask (.dcm, .nii)')
disp('- Enter 2 to use example mask ')
disp('- Enter 3 to use most recent mask, which is: ')
if isfile('recent_maskpath.mat')
    load('recent_maskpath.mat')
    disp(fullfile(recent_maskpath, recent_mask))
else
    disp('No recent mask found')
end
disp('- Enter 4 to forgo mask')
disp('- Enter 5 to automatically mask with OTSU split [default]')
load_mask_option = input('');

if load_mask_option == 1
    % Prompt user to select a mask file
    [mask_file, mask_path] = uigetfile({'*.*', 'All Files (*.*)'; '*.nii;*.nii.gz', 'NIfTI Files (*.nii, *.nii.gz)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'}, ...
                                        'Select Mask File', 'MultiSelect', 'off');

elseif load_mask_option == 4
        % No mask will be used
        mask_file = '';
        mask_path = '';
        disp('No mask will be used.');
elseif load_mask_option == 3
        % Use the most recent mask file path
        if isfile('recent_maskpath.mat')
            load('recent_maskpath.mat')
        else
            disp('No recent mask found')
        end
        mask_file = recent_mask;
        mask_path = recent_maskpath;
        disp(['Using most recent mask file: ', fullfile(mask_path, mask_file)]);
elseif load_mask_option == 2
        % Use preset mask file-path
        mask_file = 'Example_12282023_Xbox_T1_mask.nii.gz';
        mask_path = [pwd '\Example images and masks 2025\'];
        disp(['Using preset mask file: ', fullfile(mask_path, mask_file)]);
else
        % Automatically generate a mask using OTSU split
        mask_file = 'OTSU';
        mask_path = 'OTSU';
        disp('Automatically generating mask using OTSU split.');

end

% Save the most recent mask path for repeated use
recent_mask = mask_file;
recent_maskpath = mask_path;
save('recent_maskpath.mat', 'recent_mask', 'recent_maskpath');

% Check the file name and extension and read the mask accordingly
if strcmp(mask_file, 'OTSU') %otsu split automatic mask
    % Apply OTSU split to the image stack
    
    mask = imbinarize(rescale(imageStack));
    
    % Display the masked image
    figure;
    subplot(1, 2, 1);
    slice = round(size(mask, 3) / 2); % Display the middle slice of the mask
    imagesc(squeeze(mask(:, :, slice)));
    colormap(gray);
    title('Automatic mask (mid-plane)');
    xlabel('Y');
    ylabel('X');
    axis image;
    
    subplot(1, 2, 2);
    imagesc(squeeze(imageStack(:, :, slice)));
    colormap(gray);
    title('Original Image (mid-plane)');
    xlabel('Y');
    ylabel('X');
    axis image;

elseif isempty(mask_file) %forgo mask
    mask = ones(size(imageStack));
else  %load mask (either from predefined example filepath or user-defined filepath)
    [~, ~, ext] = fileparts(mask_file);
    if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
        mask = imbinarize(niftiread(fullfile(mask_path, mask_file)));
    elseif strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
        mask = imbinarize(imread(fullfile(mask_path, mask_file)));
    else
        error('Unsupported mask file format.');
    end
end

% mask = imbinarize(mask);
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


% Gather some metadata about the image stack
disp('- Enter voxel size, in [mm] [default: 0.5]')
voxel_size = input('');

if isempty(voxel_size)
    voxel_size = 0.5; %default for Bruker scans; typically 0.5mm
end

% Get the size of the image stack
[dimX, dimY, dimZ] = size(imageStack);

disp('~~~~~~~~~ Step 1 complete ~~~~~~~~~');

%% [2] Step 2: User inputs
disp('Step 2: select run mode and parameters');
disp('Do you want to iterate through the full volume or analyze a single voxel sub-domain?');
disp('Enter 1 to iterate through full volume');
disp('Enter 2 to analyze a single voxel sub-domain [default]');
run_mode = input('');
if isempty(run_mode)
    run_mode = 2;
end
if run_mode ~= 1 && run_mode ~= 2
    run_mode = 2;
end

disp('Enter sub-domain width, in voxels [default: 20]');
cropWidth_voxels = input('');
if isempty(cropWidth_voxels)
    cropWidth_voxels = 20; %default
end

% Check if cropWidth_voxels is a scalar and an integer
if ~isscalar(cropWidth_voxels) || cropWidth_voxels <= 0 || mod(cropWidth_voxels, 1) ~= 0
    error('cropWidth_voxels must be a non-zero integer scalar.');
end

disp('Enter periodicity length, e.g. unit-cell size for lattice transverse to anisotropy direction, in [mm] [default: 2]');
periodicity_mm = input('');
if isempty(periodicity_mm)
    periodicity_mm = 2; %default
end

disp('~~~~~~~~~~ end of step 2 ~~~~~~~~~~')

%% [3] Step 3: two cases: a) full volume or b) single-voxel subdomain analysis

if run_mode == 1 % First, we will consider full volume analysis
    disp('Step 3, option 1: iterate through full volume');
    disp('Enter a results filename, in single quotes [default: FFTO3D_output.mat]');
    output_filename = input('');
    if isempty(output_filename)
        output_filename = 'FFTO3D_output'; %default
    end

    disp('Enter voxel analysis spacing (non-zero integer), in voxels [default: 2]');
    vox_orientation_spacing = input('');
    if isempty(vox_orientation_spacing) || vox_orientation_spacing <= 0 || mod(vox_orientation_spacing, 1) ~= 0
        vox_orientation_spacing = 2; %default
    end
    disp(['Voxel spacing = ', num2str(vox_orientation_spacing), ' (i.e. down-sampled by a factor of ', num2str(vox_orientation_spacing)]);

    %% Image grid 3D FFT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    % Define the voxel spacing for orientation calculation

    % Initialize a cell array to store orientation vectors for each voxel
    V_orientation_all = cell(dimX, dimY, dimZ);
    R_basis_all = cell(dimX, dimY, dimZ);
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
                if mask(x, y, z)
                    disp(['Voxel coordinate: [', num2str(x), ', ', num2str(y), ', ', num2str(z), ']']);
                    
                    
                    % Crop the image stack and the mask.
                    subImageStack = neighborhood_crop(imageStack, neighborhood_coordinate, cropWidth_voxels);
                    subImageMask = neighborhood_crop(mask, neighborhood_coordinate, cropWidth_voxels);

                    % Then apply the mask, as a "grey mask" where masked regions are replaced with the average intensity
                    subImageStack = grey_mask(subImageStack, subImageMask);
                        if ( sum(subImageMask(:)) < length(subImageMask(:))*0.10 || isnan(sum(subImageStack(:))) ) %if it's a mostly empty mask, just don't bother running algorithm
                            disp('Masked image stack is >90% empty; empty vector returned.');
                            % V_orientation = [1 0 0]; % return a default orientation vector
                            V_orientation = []; % return a default orientation vector
                            FA = 0; % return a default FA
                        else
                            % Calculate the orientation vector for the current voxel !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                 
                            [V_orientation, R_basis, FA] = Image_grid_3D_FFT_KNE_2025(subImageStack, voxel_size, periodicity_mm);
                        end

                    % Store the orientation vector in the cell array
                    V_orientation_all{x, y, z} = V_orientation;
                    R_basis_all{x, y, z} = R_basis;
                    % Store the FA value in a 3D array (although, as of Dec 2024, FA is useless)
                    FA_all(x, y, z) = FA;
                    iters_completed = iters_completed + 1;
                % else
                %     disp('Voxel outside of ROI mask. Empty cell');
                end

            end
        end
        % Display percent progress after each x iteration
        % percent_complete = (iters_completed / total_voxels * 100);
        % disp(['~~~~~~~~~~~~~Progress: ', num2str(percent_complete, '%.2f'), '% ~~~~~~~~~~~~~']);
    end


elseif run_mode == 2 % Or, consider single-voxel subdomain analysis
    do_primary_plots = true;
    do_secondary_plots = true;
    tic
    disp('Step 3, option 2: analyze a single voxel sub-domain');
        % Plot every other slice of the image stack in a tiled plot
        disp('Plotting every 4th slice of the image stack...');
        plot_slices(imageStack);
    default_neighborhood_coordinate = [80, 90, 28]; %default
    disp(['Enter voxel coordinate for sub-domain analysis, in [x, y, z] [default: ' num2str(default_neighborhood_coordinate) ']' ]);
    neighborhood_coordinate = input('');
    if isempty(neighborhood_coordinate)
        neighborhood_coordinate = default_neighborhood_coordinate; %default
    end

    % Now we need to crop the image stack and the mask.
    subImageStack = neighborhood_crop(imageStack, neighborhood_coordinate, cropWidth_voxels);
    subImageMask = neighborhood_crop(mask, neighborhood_coordinate, cropWidth_voxels);

    % Then apply the mask, as a "grey mask" where masked regions are replaced with the average intensity
    subImageStack = grey_mask(subImageStack, subImageMask);

    % Display crop region
        
        % Display the original image with the crop region outlined
        figure;
        crop_midslice = neighborhood_coordinate(3);
        % crop_slice = floor(size(originalImage, 3) / 2)+cropStartZ-paddingSize+1;
        imageStack_with_mask = grey_mask(imageStack, mask);
        imagesc(squeeze(imageStack_with_mask(:, :, crop_midslice)));
        
        colormap(gray);

        title(['Slice ' num2str(crop_midslice) ' with Crop Region Outlined']);
        hold on;
        % Calculate the position of the rectangle in the original image

        cropStartX = (neighborhood_coordinate(1))  - cropWidth_voxels/2;
        cropStartY = (neighborhood_coordinate(2))  - cropWidth_voxels/2;
        cropStartZ = (neighborhood_coordinate(3))  - cropWidth_voxels/2;

        rectX = cropStartY;
        rectY = cropStartX;
        rectWidth = cropWidth_voxels-1 ;
        rectHeight = cropWidth_voxels-1 ;

        % Draw the rectangle on the original image
        rectangle('Position', [rectX, rectY, rectWidth, rectHeight], 'EdgeColor', 'r', 'LineWidth', 2);
        xlabel('Y');
        ylabel('X');
        hold off;

%Almost ready to run algorithm: check if the mask is mostly empty
    if ( sum(subImageMask(:)) < length(subImageMask(:))*0.10 || isnan(sum(subImageStack(:))) ) %if it's a mostly empty mask, just don't bother running algorithm
        disp('Masked image stack is >90% empty; empty vector returned.');
        % V_orientation = [1 0 0]; % return a default orientation vector
        V_orientation = []; % return a default orientation vector
        FA = 0; % return a default FA

%Otherwise, call the magic function~!!!!!!!!!!!!!!!~!!!!!!!!!!!!!!~!!!!!!!!!!!!!~!!!!!!!!!!!
    else 
        [V_orientation, R_basis,FA] = Image_grid_3D_FFT_KNE_2025(subImageStack, voxel_size, periodicity_mm, do_primary_plots,do_secondary_plots);
        % Display the orientation vector and FA value
        disp(['Orientation vector: [', num2str(V_orientation(1), '%.2f'), ', ', num2str(V_orientation(2), '%.2f'), ', ', num2str(V_orientation(3), '%.2f'), ']']);
        % disp(['Fractional anisotropy (do not trust): ', num2str(FA, '%.2f')]);
        disp('R_basis (columns are the unit vectors):');
        R_basis
    end

    % disp('Enter a results filename, in single quotes [default: FFTO3D_output.mat]');
    % output_filename = input('');
    % if isempty(output_filename)
        output_filename = 'FFTO3D_output'; %default (doesn't get saved in single-shot mode)
    % end

end


%% Save to mat file
% Check if the file already exists and number the name if necessary

file_number = 1;
while exist(output_filename, 'file')
    output_filename = [output_filename, num2str(file_number), '.mat'];
    file_number = file_number + 1;
end

% Calculate and display the total time in minutes and hours
total_time_seconds = toc;
total_time_minutes = total_time_seconds / 60;
total_time_hours = total_time_minutes / 60;

disp(['Total time to run script: ', num2str(total_time_seconds, '%.2f'), ' seconds (', num2str(total_time_minutes, '%.2f'), ' minutes, ', num2str(total_time_hours, '%.2f'), ' hours).']);
if run_mode == 1
    % Save V_orientation_all to the determined .mat file for full volume analysis
    save(output_filename, 'V_orientation_all', 'R_basis_all','FA_all', 'imageStack', 'mask', 'vox_orientation_spacing', 'cropWidth_voxels', 'voxel_size', 'dimX', 'dimY', 'dimZ');
    disp(['Results saved to ', output_filename]);
else
    % Save V_orientation and FA for single voxel sub-domain analysis
    % save(output_filename, 'V_orientation', 'FA','R_basis','imageStack', 'mask', 'neighborhood_coordinate', 'cropWidth_voxels', 'voxel_size', 'periodicity_mm');
end



%% post-processing; open this function
% post_process_visualization_2025

return



%% Functions
function plot_slices(imageStack)
    % Determine the number of slices to plot
    dimZ = size(imageStack, 3);
    % Create a tiled layout for the plots
    tiledlayout('flow');
    % Loop through every other slice and plot it
    for slice_idx = 1:4:dimZ
        % Select the next tile
        nexttile;
        
        % Display the slice
        imagesc(imageStack(:, :, slice_idx));
        colormap gray;
        axis image;
        title(['Slice ', num2str(slice_idx)]);
    end
end


function NeighborhoodVolume = neighborhood_crop(imageStack, neighborhood_coordinate, cropWidth_voxels)

    % Add padding to the imageStack and mask to allow crops centered on edges
    paddingSize = ceil(cropWidth_voxels / 2);

    % Pad the imageStack and mask with the average intensity and zeros respectively
    paddedImageStack = padarray(imageStack, [paddingSize, paddingSize, paddingSize], 0, 'both');

    % % Update the crop center location to account for the padding
    % crop_center_Location_voxels = crop_center_Location_voxels + paddingSize;

    % Crop dicom stack to 3D cube of specified width, centered in the volume

    % cropStartX = (neighborhood_coordinate(1)) + round((size(paddedImageStack, 1)) / 2 - cropWidth_voxels/2);
    % cropStartY = (neighborhood_coordinate(2)) + round((size(paddedImageStack, 2)) / 2 - cropWidth_voxels/2);
    % cropStartZ = (neighborhood_coordinate(3)) + round((size(paddedImageStack, 3)) / 2 - cropWidth_voxels/2);
    cropStartX = (neighborhood_coordinate(1)) + paddingSize - cropWidth_voxels/2;
    cropStartY = (neighborhood_coordinate(2)) + paddingSize - cropWidth_voxels/2;
    cropStartZ = (neighborhood_coordinate(3)) + paddingSize - cropWidth_voxels/2;


    % Ensure the starting indices are within the valid range
    cropStartX = max(1, min(cropStartX, size(paddedImageStack, 1) - cropWidth_voxels ));
    cropStartY = max(1, min(cropStartY, size(paddedImageStack, 2) - cropWidth_voxels ));
    cropStartZ = max(1, min(cropStartZ, size(paddedImageStack, 3) - cropWidth_voxels ));

    % Crop the DICOM stack
    NeighborhoodVolume = paddedImageStack(cropStartX:cropStartX+cropWidth_voxels-1, cropStartY:cropStartY+cropWidth_voxels-1, cropStartZ:cropStartZ+cropWidth_voxels-1);
    % Crop the mask too
    % mask(:,:,10) % display the 10th slice of the mask
    % imageStack(:,:,10) % display the 10th slice of the image stack
end

function [grey_masked_Volume] = grey_mask(Volume, mask)
    % Apply the mask to the  DICOM stack
    grey_masked_Volume = Volume .* cast(mask, 'like', Volume);
    % Replace masked regions with the average voxel intensity
    averageIntensity = median(grey_masked_Volume(mask == 1), 'all');

    grey_masked_Volume(mask == 0) = averageIntensity;
end