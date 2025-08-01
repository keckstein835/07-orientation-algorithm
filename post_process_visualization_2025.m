% Script Name: [Name of the Script]
% Date Written: 2024-11-8
% Author: Kevin Eckstein
% Description: This script takes data from "Find_3D_lattice_fiber_dir_KNE.m" and plots the data, also saves a "DTI.mat" at the correct resolution for input to NLI.
% this function also applies smoothing; so yoou can pull raw data from the output file and apply your own smoothing if desired.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function post_process_visualization(output_filename)
% if nargin < 1
%     % output_filename = 'FFTO3D 20241228xbox.mat';
%     output_filename = 'FFTO3D_20240207scaled.mat';
% end

% Now working in 2025_working; whew it feels good to have that all done!
clear variables
close all
% 
disp('Running post_process_visualization_2025.m; KNE 2025 v1.0');

%% User settings (note: output_filename is actually the input to this code):
    % output_filename = 'FFTO3D 20241228xbox.mat'; % requires index flip (Why??? I don't know what happened)
    % output_filename = 'FFTO3D_20240207scaled.mat'; 
    % output_filename = 'FFTO3D_20250204_arch.mat';
    % output_filename = 'FFTO3D_20231227_DD.mat'; % requires index flip (Why??? I don't know what happened)
    % output_filename = 'FFTO3D_20241122_agar_XBOX.mat';
    % output_filename = 'FFTO3D_20241122xboxAgar.mat';
    % output_filename = 'FFTO3D_20241214supersoft.mat';
    % output_filename = 'FFTO3D_TPUagar20241214';
    % output_filename = 'FFTO3D_TPUagar20241214';
    % output_filename = 'FFTO3D_20240204_arch_multilat.mat'; %use 28mmNH instead of this one
    % output_filename = 'FFTO3D_archMultiLat_28mmNH.mat';
    output_filename = 'FFTO3D_archMultiLat_28mmNH_FAadded.mat';
    % output_filename = 'FFTO3D_output_FA_test2';
    % output_filename = '20250508_PC_syl_FFTO3D_output.mat';
    % output_filename = 'test';
    
    % fraction = 6; % Sample every n-th point for the quiver plot
    fraction = 3; %3 is usually good for 60x60x36 images.

    downsample_factor = 2; % should be 2. Downsample factor for the imageStack and orientation vectors (going from your anatomical images, e.g. 120 x 120 x 72, to MRE images, 60 x 60 x 36 -KNE 2025-2-18)

    do_index_flip = false; % Set to true if the indices need to be flipped to match the appearance of the T1 image on ITK-snap (check against ITK-snap after running code)
    % do_index_flip = true;

%% Load the data from the output file
vars = {'V_orientation_all', 'dimX', 'dimY', 'dimZ', 'imageStack', 'FA_all','vox_orientation_spacing', 'cropWidth_voxels', 'voxel_size'};
data = load(output_filename, vars{:});
V_orientation_all = data.V_orientation_all;
dimX = data.dimX;
dimY = data.dimY;
dimZ = data.dimZ;
imageStack = data.imageStack;
FA_all = data.FA_all;
vox_orientation_spacing = data.vox_orientation_spacing;
cropWidth_voxels = data.cropWidth_voxels;
voxel_size = data.voxel_size;
% Convert V_orientation_all from cell array to 4D numeric array
V_orientation_all_numeric = nan(dimX, dimY, dimZ, 3); % Initialize with NaNs


for x = 1:dimX
    for y = 1:dimY
        for z = 1:dimZ
            if ~isempty(V_orientation_all{x, y, z})
                V_orientation_all_numeric(x, y, z, :) = V_orientation_all{x, y, z};
                
            end
        end
    end
end


%% Get rid of FA_all because FA is not meaningful enough to be useful; set to for numbers that are not NaN
% FA_all(isnan(FA_all)) = 0;
% FA_all(~isnan(FA_all)) = 1;


%% interpolate the orientation vectors to fill in the nan values (currently not implemented)
% V_orientation_all_interp = fillmissing(V_orientation_all_numeric, 'movmean',vox_orientation_spacing);

%% Switch indices here, if needed
V_orientation_all_interp = V_orientation_all_numeric;

if do_index_flip
    disp('Switching indices to match appearance of T1 image on ITK-snap');
    % Flip the indices 1 and 2 of imageStack and FA_all to match the itk-snap view when plotted by matlab
    imageStack = permute(imageStack, [2, 1, 3]);
    FA_all = permute(FA_all, [2, 1, 3]);

    % Flip indices 1 and 2 of V_orientation_all_interp to match the itk-snap view when plotted by matlab
    V_orientation_all_interp = permute(V_orientation_all_interp, [2, 1, 3, 4]);

    % Swap the 1st and 2nd entries in the 4th dimension of V_orientation_all_interp (i.e. orientation vector)  
    V_orientation_all_interp = V_orientation_all_interp(:,:,:,[2,1,3]);
end





%% Plot: first, find an appropriate (not empty) slice:
% slice = round(dimZ/2);
% Find the closest slice above the midplane that contains data
midplane = round(dimZ / 2);
slice = midplane;
while slice >= 1  && all(isnan(V_orientation_all_interp(:,:,slice,:)), 'all')
    slice = slice - 1;
end

% If no slice above the midplane contains data, use the midplane slice
if slice < 1
    slice = midplane;
    disp('No slice above the midplane contains data. Using midplane slice.');
end

% plot_multislice_XY(imageStack, V_orientation_all_interp, FA_all, slice, vox_orientation_spacing) % Plot here (if desired)


%% Downsample to MRE resolution

% Downsample V_orientation_all_interp
V_downsampled = V_orientation_all_interp(1:downsample_factor:end, 1:downsample_factor:end, 1:downsample_factor:end, :);
FA_downsampled = FA_all(1:downsample_factor:end, 1:downsample_factor:end, 1:downsample_factor:end);

% Replace NaN or empty values in V_downsampled with [0, 0, 1]
for x = 1:size(V_downsampled, 1)
    for y = 1:size(V_downsampled, 2)
        for z = 1:size(V_downsampled, 3)
            if any(isnan(V_downsampled(x, y, z, :))) || isempty(V_downsampled(x, y, z, :))
                V_downsampled(x, y, z, :) = [0, 0, 1];
            end
        end
    end
end


% Downsample imageStack so that each dimension is half the original length
imageStack_downsampled = imageStack(1:2:end, 1:2:end, 1:2:end);

plot_multislice_XY(imageStack_downsampled, V_downsampled, FA_downsampled, round(slice/2), 1,fraction)


% now smooth the downsampled vector field
sigma = 0.75; % Standard deviation for Gaussian kernel (1 works well. 0.75 works better)
V_downsampled_smooth(:,:,:,1) = imgaussfilt3(V_downsampled(:,:,:,1), sigma);
V_downsampled_smooth(:,:,:,2) = imgaussfilt3(V_downsampled(:,:,:,2), sigma);
V_downsampled_smooth(:,:,:,3) = imgaussfilt3(V_downsampled(:,:,:,3), sigma);
plot_multislice_XY(imageStack_downsampled, V_downsampled_smooth, FA_downsampled, round(slice/2), 1,fraction)
disp('Two multi-slice figures: first before smoothing, second after smoothing')

%% Plot 3-view midplane slices with quivers
figure;
% Plot midplane slices of downsampled imageStack and quivers of V_downsampled_smooth
midplane_downsampled = round(size(imageStack_downsampled, 3) / 2);

% Plot midplane slices for 3 orthogonal directions


% Define the grid for quiver plot
% fraction = 3; % Sample every n-th point for the quiver plot
disp(['Note: fraction = ' num2str(fraction) ' for quiver plot']);
[xGrid, yGrid, zGrid] = meshgrid(1:fraction:size(imageStack_downsampled, 1), 1:fraction:size(imageStack_downsampled, 2), 1:fraction:size(imageStack_downsampled, 3));

% XY plane at midplane
subplot(2, 2, 1);
imagesc(imageStack_downsampled(:, :, midplane_downsampled));
colormap(gray);
title('XY Plane');
hold on;
quiver(xGrid(:,:,1), yGrid(:,:,1), V_downsampled_smooth(1:fraction:end, 1:fraction:end, midplane_downsampled, 2), V_downsampled_smooth(1:fraction:end,1:fraction:end, midplane_downsampled, 1), 'r', 'LineWidth', 1, 'MaxHeadSize', 1, 'AutoScale', 'on', 'AutoScaleFactor', 0.5);
hold off;
xlabel('Y');
ylabel('X');
% Make pixels square by adjusting the aspect ratio
daspect([1 1 1]);

% XZ plane at midplane
midplane_XZ = round(size(imageStack_downsampled, 2) / 2);
[xGrid_XZ, zGrid_XZ] = meshgrid(1:fraction:size(imageStack_downsampled, 1), 1:fraction:size(imageStack_downsampled, 3));
subplot(2, 2, 2);
imagesc(squeeze(imageStack_downsampled(:, midplane_XZ, :)));
colormap(gray);
title('XZ Plane');
hold on;
quiver( zGrid_XZ',xGrid_XZ', squeeze(V_downsampled_smooth(1:fraction:end, midplane_XZ, 1:fraction:end, 3)), squeeze(V_downsampled_smooth(1:fraction:end, midplane_XZ, 1:fraction:end, 1)), 'r', 'LineWidth', 1, 'MaxHeadSize', 1, 'AutoScale', 'on', 'AutoScaleFactor', 0.5);
hold off;
% Make pixels square by adjusting the aspect ratio
daspect([1 1 1]);

% YZ plane at midplane
midplane_YZ = round(size(imageStack_downsampled, 1) / 2);
[yGrid_YZ, zGrid_YZ] = meshgrid(1:fraction:size(imageStack_downsampled, 2), 1:fraction:size(imageStack_downsampled, 3));
subplot(2, 2, 3);
imagesc(squeeze(imageStack_downsampled(midplane_YZ, :, :))');
colormap(gray);
title('YZ Plane');
hold on;
quiver(yGrid_YZ, zGrid_YZ, squeeze(V_downsampled_smooth(midplane_YZ, 1:fraction:end, 1:fraction:end, 2))', squeeze(V_downsampled_smooth(midplane_YZ, 1:fraction:end, 1:fraction:end, 3))', 'r', 'LineWidth', 1, 'MaxHeadSize', 1, 'AutoScale', 'on', 'AutoScaleFactor', 0.5);
hold off;
% Make pixels square by adjusting the aspect ratio
daspect([1 1 1]);

% 3D Quiver plot of the downsampled and smoothed vector field (middle 1/4)
figure;

% Define the range for the middle 1/4 in the z-direction only
zRange = round(size(V_downsampled_smooth, 3) / 4):round(3 * size(V_downsampled_smooth, 3) / 4);

[xGrid3D, yGrid3D, zGrid3D] = meshgrid(1:fraction:size(V_downsampled_smooth, 1), 1:fraction:size(V_downsampled_smooth, 2), zRange(1):fraction:zRange(end));
quiver3(xGrid3D, yGrid3D, zGrid3D, ...
    V_downsampled_smooth(1:fraction:end, 1:fraction:end, zRange(1):fraction:zRange(end), 2), ...% x and y are intentionally flipped here (it's just the way it is with quiver plots here)
    V_downsampled_smooth(1:fraction:end, 1:fraction:end, zRange(1):fraction:zRange(end), 1), ...
    V_downsampled_smooth(1:fraction:end, 1:fraction:end, zRange(1):fraction:zRange(end), 3), ...
    'r');
title('3D Quiver Plot of Downsampled and Smoothed Vector Field (Middle 1/4 in Z)');
xlabel('Y');
ylabel('X');
zlabel('Z');
axis equal;
grid on;

%% Plot heatmap of FA in midslices

% Define the mid slices for each plane
mid_XY = round(size(FA_downsampled, 3) / 2);
mid_XZ = round(size(FA_downsampled, 2) / 2);
mid_YZ = round(size(FA_downsampled, 1) / 2);

figure;

% Plot FA heatmap for XY plane at mid slice
subplot(2, 2, 1);
imagesc(FA_downsampled(:, :, mid_XY-5));
colormap(jet);
colorbar;
title('FA Heatmap - XY Plane');
xlabel('Y');
ylabel('X');
axis equal;

% Plot FA heatmap for XZ plane at mid slice
subplot(2, 2, 2);
imagesc(squeeze(FA_downsampled(:, mid_XZ+1, :)));
colormap(jet);
colorbar;
title('FA Heatmap - XZ Plane');
xlabel('Z');
ylabel('X');
axis equal;

% Plot FA heatmap for YZ plane at mid slice
subplot(2, 2, 3);
imagesc(squeeze(FA_downsampled(mid_YZ+1, :, :))');
colormap(jet);
colorbar;
title('FA Heatmap - YZ Plane');
xlabel('Y');
ylabel('Z');
axis equal;

sgtitle('FA Heatmaps for Mid Slices');


%% Save the variables V_orientation_all_smooth and FA_all to a file named "DTI.mat"
V1 = V_downsampled;
FA = FA_downsampled;

% Flip indices 1 and 2 of V1 and FA to match the NLI convention (if desired -- not sure if that's correct. KNE 2025-1-16)
% V1 = permute(V1, [2, 1, 3, 4]);
% FA = permute(FA, [2, 1, 3]);

% Swap the 1st and 2nd entries in the 4th dimension of V1 (i.e. orientation vector)  (if desired -- not sure if that's correct. KNE 2025-1-16)  
    % V1 = V1(:,:,:,[2,1,3]);

% as of 2025-2-6, I don't think these flips (above ~6 lines) are necessary so DONT do them.
% as of 2025-2-25, flips were necessary but were done earlier in the script; so don't do them here.

save('DTI.mat', 'V1', 'FA');
disp('Variables V1 and FA saved to DTI.mat');

% Plot slice of imageStack
figure;
imagesc(imageStack(:, :, 32));
colormap(gray);
title('Slice 32 of imageStack');
xlabel('X');
ylabel('Y');
axis equal;
colorbar;

% % Plot histogram of FA_all with 9 bins
% figure;
% n_bins = 20;

% % Adjust y-axis to fit data excluding the first bin
% h = histogram(FA_all, n_bins);
% counts = h.Values;
% counts(1) = 0; % Exclude the first bin
% ylim([0 max(counts)]);
% % label the plot
% xlabel('FA values');
% ylabel('Frequency');
% title('Histogram of FA\_all');
% xlim([0 1]);


% [V_orientation] = Image_grid_3D_FFT_KNE(imageStack, mask, crop_center_Location_voxels, cropWidth_voxels, voxel_size);

% crop_center_Location_voxels = crop_center_Location_voxels + [0, 0, 1]; % Shift the crop center up by 1 voxel

% [V_orientation_next] = Image_grid_3D_FFT_KNE(imageStack, mask, crop_center_Location_voxels, cropWidth_voxels, voxel_size);




%% Archived code
% you may find these lines useful:
    % % Prompt user to select a stack of DICOM files, a single DICOM volume, or a TIFF stack
    % [file, path] = uigetfile({'*.*', 'All Files (*.*)';'*.dcm', 'DICOM Files (*.dcm)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'}, ...
    %                         'Select DICOM or TIFF Files', 'MultiSelect', 'on');

    % % display file and path so that it may be copy and pasted back into code
    % disp(file)  % display file
    % disp(path)  % display path

    % % Prompt user to select a mask file, accepting nii, tif, and tiff files, and ITK-snap mask files (nii.gz)
    % [mask_file, mask_path] = uigetfile({'*.*', 'All Files (*.*)';'*.nii;*.nii.gz', 'NIfTI Files (*.nii, *.nii.gz)'; '*.tif;*.tiff', 'TIFF Files (*.tif, *.tiff)'; '*.*', 'All Files (*.*)'}, ...
    %                                    'Select Mask File', 'MultiSelect', 'off');

    % % Display mask file and path so it may be copy and pasted back into code
    % disp(mask_file)  % display file
    % disp(mask_path)  % display path


function plot_multislice_XY(imageStack, V_orientation_all_interp, FA_all, slice, vox_orientation_spacing,fraction)
% Plot the orientation vectors for slices around the middle slice
figure;
% Define the range of slices to plot
slices_to_plot = slice-7*vox_orientation_spacing:vox_orientation_spacing:slice+7*vox_orientation_spacing;

% Create subplots for each slice
    for i = 1:length(slices_to_plot)
        subplot(3, 5, i);
        current_slice = slices_to_plot(i);
        
        % Ensure the slice index is within bounds
        if current_slice < 1 || current_slice > size(imageStack, 3)
            continue;
        end
        
        imagesc(imageStack(:,:,current_slice));
        colormap(gray);
        hold on;
        % fraction = 10; % Sample every n-th point for the quiver plot
        % Create a grid to sample only the specified fraction of the quivers
        [xGrid, yGrid] = meshgrid(1:fraction:size(imageStack,1), 1:fraction:size(imageStack,2));
        
        % Scale quivers by their respective FA intensity (edited to not use FA)
        FA_slice = FA_all(:,:,current_slice);
        FA_slice(isnan(FA_slice)) = 0; % Replace NaNs with 0 for scaling
        FA_sampled = FA_slice(1:fraction:end, 1:fraction:end);

        scale_factor = (2*fraction/4); %scale the size of arrows (can adjust if wanted)
        
        % Extract the sampled orientation vectors (remember, x and y are swapped in the orientation vectors)
        Vx = V_orientation_all_interp(1:fraction:end, 1:fraction:end, current_slice, 2) * scale_factor;
        Vy = V_orientation_all_interp(1:fraction:end, 1:fraction:end, current_slice, 1) * scale_factor;
        
        h = quiver(xGrid, yGrid, Vx, Vy, 'off', 'r', 'LineWidth', 2, 'MaxHeadSize', 2);
        % Plot quivers in the opposite direction
        Vx_neg = -Vx;
        Vy_neg = -Vy;
        quiver(xGrid, yGrid, Vx_neg, Vy_neg, 'off', 'r', 'LineWidth', 2, 'MaxHeadSize', 2);
        
        hold off;
        title(['Slice ' num2str(current_slice)]);
    end

% Add a main title for the figure
sgtitle(['Orientation Vectors for Slices around Slice ' num2str(slice)]);
end
% figure
% imagesc(imageStack(:,:,slice));
% colormap(gray);
% hold on
% % Smooth the vector field using a Gaussian filter
% sigma = 0.75; % Standard deviation for Gaussian kernel
% Vx_smooth = imgaussfilt(Vx, sigma);
% Vy_smooth = imgaussfilt(Vy, sigma);

% % Update the quiver plot with the smoothed vectors
% h2 = quiver(xGrid, yGrid, Vx_smooth, Vy_smooth, 'off', 'r', 'LineWidth', 2, 'MaxHeadSize', 2);
% % plot quivers in the opposite direction
% Vx_smooth_neg = -Vx_smooth;
% Vy_smooth_neg = -Vy_smooth;
% quiver(xGrid, yGrid, Vx_smooth_neg, Vy_smooth_neg, 'off', 'r', 'LineWidth', 2, 'MaxHeadSize', 2);

% hold off



% end
        
