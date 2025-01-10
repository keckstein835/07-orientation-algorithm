%% Header --------------------------------------------------------------------------
% Image_grid_3D_FFT_KNE.m
% Author: Kevin Eckstein
% Date: 2024-02-09
%
% Description:
% This script reads a stack of DICOM files from a specified directory,
% performs a 3D Fast Fourier Transform (FFT) on the stack, and plots the
% resulting 3D power spectrum.
%
% Usage:
% Set the 'do_preset_filepath' variable to 1 to use the preset file path,
% or set it to 0 to manually select the DICOM files.
%
%--------------------------------------------------------------------------
% create message that code is running
disp('Image_grid_3D_FFT_KNE.m running...');
clear all
close all
%print the time
disp(datestr(now));

%% Todo list;
% When you find the point, you can refine accuracy by taking the centroid of a strel of nearby voxels (basically we're limited in resolution but neighboring voxels can help interpolate
% Scooping the XY plane might be a legitamate method to account for stinger problems (though I might have fixed this with the mask)
% Finally, relate to what NLI accepts for DTI vectors
% Binarize and try RANSAC
% A nearest-neighbor smoothing might be helpful down the line because I can see some of these jumping to 90 degree the wrong way



%% User inputs
    % solve_method = 1; % 1 = FFT direct, 2 = FFT with template fitting, 3 = RANSAC, 4 = Radon transform(?)

    do_preset_filepath = 1;
    display_plots = true;
    % Ask the user for the width of the cropped region in mm
    cropWidth_mm = 10; % mm; kernel width(?)
    % set a 3x1 vector for the crop location w.r.t. center
    % cropLocation_mm = [4, 0, 0]; % mm offset from center
    cropLocation_mm = [-2, -8, -4]; % mm offset from center
    voxel_size = 0.5; % mm
    color_axis_limit = [0 1e13];
    % filter_max_feature_size = 2.6; %mm; 2.6 is good
    % filter_min_feature_size = 2; %mm; 2.0 is good
    % filter_max_feature_size = 2.6; %mm; 2.6 is good for scaled
    % filter_min_feature_size = 1.9; %mm; 1.9 is good for scaled
    filter_max_feature_size = 2.5; %mm; for Xbox
    filter_min_feature_size = 1.9; %mm; for Xbox
    percentile_ProminentFreqs = 99; % percentile of the most prominent frequencies to use as candidates for orientation vector voting procedure
        % higher percentile will make code go faster but may miss the correct orientation vector if the FFT signal is weak
    
    
    roll_increment_deg = 2; % degrees; Adjust as needed; for rotating template during template fitting (lower number is more accurate and expensive)
    dot_radius_factor = 1; % Set to 1 for a reasonable dot radius for template fitting, but can reduce if needed for accuracy
    
    % percentile_ProminentPoints_all_RANSAC = 98; % percentile of most prominent points to use for RANSAC regression

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

    % Display mask file and path so it may be copy and pasted back into code
    disp(mask_file)  % display file
    disp(mask_path)  % display path

end

%% Import image
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.nii') || strcmpi(ext, '.gz')
    imageStack = niftiread(fullfile(path, file));
else
    imageStack = getimageStack(file, path);
end

if display_plots == true
    originalImage = imageStack;
end

%% Mask the image (if desired)
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
% if mask is different size than imageStack, return error
if size(mask) ~= size(imageStack)
    error('Mask and image stack are different sizes');
end 

% Apply the mask to the  DICOM stack
imageStack = imageStack .* cast(mask, 'like', imageStack);
% Replace masked regions with the average voxel intensity
averageIntensity = median(imageStack(mask == 1), 'all');
imageStack(mask == 0) = averageIntensity;


%% Crop image
% Convert the crop width from mm to voxels
cropWidth = cropWidth_mm / voxel_size;
% cropWidth = input('Enter the width of the cropped region (in mm): ');

% Display the selected crop width
disp(['Selected crop width: ' num2str(cropWidth_mm) ' mm']);
disp(['Selected crop width in voxels: ' num2str(cropWidth)]);


% Crop dicom stack to 3D cube of specified width, centered in the volume
% Calculate the starting indices for cropping, offset by cropLocation_mm
cropStartX = round((size(imageStack, 1) - cropWidth) / 2) + cropLocation_mm(1) / voxel_size;
cropStartY = round((size(imageStack, 2) - cropWidth) / 2) + cropLocation_mm(2) / voxel_size;
cropStartZ = round((size(imageStack, 3) - cropWidth) / 2) + cropLocation_mm(3) / voxel_size;

% Ensure the starting indices are within the valid range
cropStartX = max(1, min(cropStartX, size(imageStack, 1) - cropWidth + 1));
cropStartY = max(1, min(cropStartY, size(imageStack, 2) - cropWidth + 1));
cropStartZ = max(1, min(cropStartZ, size(imageStack, 3) - cropWidth + 1));

% Crop the DICOM stack
imageStack = imageStack(cropStartX:cropStartX+cropWidth-1, cropStartY:cropStartY+cropWidth-1, cropStartZ:cropStartZ+cropWidth-1);
% % Crop the 3D mask
% mask = mask(cropStartX:cropStartX+cropWidth-1, cropStartY:cropStartY+cropWidth-1, cropStartZ:cropStartZ+cropWidth-1);

% Display the size of the cropped DICOM stack
disp('The size of the cropped DICOM stack is:');
domainSize = size(imageStack, 1);
disp(size(imageStack));

midSlice = round(size(imageStack, 3) / 2);

% Create simple grids x, y, z the size of imageStack for plotting
[x, y, z] = ndgrid(1:size(imageStack, 1), 1:size(imageStack, 2), 1:size(imageStack, 3));


%% Crop cube to a sphere, if desired
% imageStack = cropCubeToSphere(imageStack); % 2024-10-24: this doesn't seem to help at all (I though other papers did this but maybe not)


%% Display the cropped image
displayMidplanes(imageStack, x*voxel_size, y*voxel_size, z*voxel_size);
display_multi_slice_around_midplanes(imageStack, [min(imageStack(:)) max(imageStack(:))], x*voxel_size, y*voxel_size, z*voxel_size);



%% Display full image (if desired)
if display_plots == true
    
    % Display the original image with the crop region outlined
    figure;
    midSlice_full = round(size(imageStack, 3) / 2)+cropStartZ;
    imagesc(squeeze(originalImage(:, :, midSlice_full)));
    colormap(gray);
    title('Original Image with Crop Region Outlined');
    hold on;
    % Calculate the position of the rectangle in the original image
    rectX = cropStartY ;
    rectY = cropStartX ;
    rectWidth = cropWidth-1 ;
    rectHeight = cropWidth-1 ;

    % Draw the rectangle on the original image
    rectangle('Position', [rectX, rectY, rectWidth, rectHeight], 'EdgeColor', 'r', 'LineWidth', 2);
    xlabel('Y');
    ylabel('X');
    hold off;
end







% % Apply a 3D Tukey window to the image stack
% alpha = 0.5; % Tukey window parameter (0 to 1)
% tukeyWindowX = tukeywin(size(imageStack, 1), alpha);
% tukeyWindowY = tukeywin(size(imageStack, 2), alpha);
% tukeyWindowZ = tukeywin(size(imageStack, 3), alpha);

% % Create a 3D Tukey window by taking the outer product of the 1D windows
% [tukeyWindowX, tukeyWindowY, tukeyWindowZ] = ndgrid(tukeyWindowX, tukeyWindowY, tukeyWindowZ);
% tukeyWindow3D = tukeyWindowX .* tukeyWindowY .* tukeyWindowZ;

% % Apply the 3D Tukey window to the image stack
% imageStack = double(imageStack) .* tukeyWindow3D;


%% 3D FFT and power spectrum computation

% Create a frequency grid
[nx, ny, nz] = size(imageStack);
fx = (-floor(nx/2):ceil(nx/2)-1) / nx;
fy = (-floor(ny/2):ceil(ny/2)-1) / ny;
fz = (-floor(nz/2):ceil(nz/2)-1) / nz;
[Fx, Fy, Fz] = ndgrid(fx, fy, fz);

% Perform a 3D FFT on the DICOM stack
fftResult = fftshift(fftn(imageStack));
% fftResult = (fftn(imageStack));

% Compute the power spectrum
powerSpectrum = abs(fftResult).^2;

% Display the 3D power spectrum
display3DPS(powerSpectrum);

% Display the cumulative magnitudes of the 3D volume projected onto the XY, XZ, and YZ planes
projectCumulativeMagnitudes(powerSpectrum, color_axis_limit, Fx, Fy, Fz)

%% Save the power spectrum as a 3D .tif stack, if desired
savePowerSpectrumAsTiff(powerSpectrum);


%% Frequency filtering: isolating ~2-2.5 mm wavelengths
% filter_max_feature_size = 2.6; %mm
% filter_min_feature_size = 2; %mm

freqMask = createFrequencyMask(powerSpectrum, filter_max_feature_size, filter_min_feature_size, voxel_size);

% Apply the mask to the power spectrum
filteredPowerSpectrum = powerSpectrum .* freqMask;

% display_multi_slice_around_midplanes(powerSpectrum, color_axis_limit, Fx,Fy,Fz); % display the power spectrum
display_multi_slice_around_midplanes(filteredPowerSpectrum, color_axis_limit, Fx,Fy,Fz); % display the filtered power spectrum
freqMask_double = double(freqMask);
% display_multi_slice_around_midplanes(freqMask_double, [0 1], Fx,Fy,Fz); % display the mask



%% Solver #1: Direct FFT: Find the most prominent frequencies and their orientation vectors from the filtered power spectrum
numProminentFreqs = 4;
[V, lambda, mostProminentIndex, mostProminentFreq]  =  findMostProminentFrequency(filteredPowerSpectrum, Fx, Fy, Fz, voxel_size, numProminentFreqs);
% display the returned values, descripttion
% disp('Orientation Unit Vector: ');
% disp(V);
disp('Frequency in mm: ');
disp(lambda);



%% Solver #2: Template fitting of candidate orientations

% 99th percentile filter to identify candidates
% filter the power spectrum to isolate the 99th percentile of most prominent frequencies

% percentile_ProminentFreqs = 99.9; % percentile of the most prominent frequencies to display

    % Calculate the threshold value for the 99th percentile
    thresholdValue = prctile(filteredPowerSpectrum(:), percentile_ProminentFreqs);

    % Create a mask for the 99th percentile
    mask_99th = filteredPowerSpectrum >= thresholdValue;

    % Apply the mask to the filtered power spectrum
    filteredPowerSpectrum_99th = filteredPowerSpectrum .* mask_99th;
    % display number of remaining prominent frequencies
    % disp(['Number of remaining prominent frequencies: ' num2str(sum(mask_99th(:)))]);
    % Find the indices of the candidate 99th percentile frequencies
    candidateIndices = find(mask_99th);

    % Convert the linear indices to subscripts
    [candidateX, candidateY, candidateZ] = ind2sub(size(filteredPowerSpectrum_99th), candidateIndices);

    % Compute the corresponding frequency vectors
    candidateFreqs = [Fx(candidateIndices), Fy(candidateIndices), Fz(candidateIndices)];

    % Normalize the frequency vectors to get orientation vectors
    candidateOrientationVectors = candidateFreqs ./ vecnorm(candidateFreqs, 2, 2);

    % Filter candidate orientation vectors to a half-space
    % Define the normal vector of the half-space (e.g., along the Z-axis)
    normalVector_halfspace = [1, 0, 0];

    % Calculate the dot product of each candidate orientation vector with the normal vector
    dotProducts = candidateOrientationVectors * normalVector_halfspace';

    % Keep only the orientation vectors that have a positive dot product (i.e., in the same half-space as the normal vector)
    candidateOrientationVectors = candidateOrientationVectors(dotProducts > 0, :);



    display_multi_slice_around_midplanes(filteredPowerSpectrum_99th, color_axis_limit, Fx,Fy,Fz); % display the filtered power spectrum


    dot_spacing = size(filteredPowerSpectrum,1)/4; % Adjust as needed
    % Define the size of the grid and spacing between dots
    gridSize = 5; % e.g. 5x5 grid
    % dot_spacing = 10; % Adjust as needed

    % Initialize arrays to store the coordinates
    x_coords = zeros(gridSize, gridSize);
    y_coords = zeros(gridSize, gridSize);

    % Loop through the grid and assign coordinates
    for i = 1:gridSize
        for j = 1:gridSize
            x_coords(i, j) = (i - 1) * dot_spacing;
            y_coords(i, j) = (j - 1) * dot_spacing;
        end
    end

    % Combine the coordinates into a single array
    coordinates = [x_coords(:), y_coords(:)];
    % disp('Coordinates of the 2D grid:');
    % disp(coordinates);

    %  rotate a dot grid and compare to the projected power spectrum at candidate orientation
    % Create array of grids with increasing rotation up to 90 degrees:
    % dot_radius = 0.72; % Adjust as needed (voxels)
    dot_radius = dot_radius_factor * 2 * domainSize/32;
    % Call the function to rotate the dot grid and return a set of grids (i.e. templates)
    rotated_dot_grids = rotateDotGrid(coordinates, roll_increment_deg, dot_radius, domainSize, dot_spacing);


    % % Plot the first 9 grids of rotated_dot_grids
    figure;
    tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = 1:9
        nexttile;
        imagesc(rotated_dot_grids(:, :, i*5-5+1));
        colormap(gray);
        colorbar;
        title(['Rotation: ' num2str((i*5-5) * roll_increment_deg) ' degrees']);
        xlabel('Y');
        ylabel('X');
    end

    high_pass_cutoff_mm = 3; %could be, for example, 4
    freqMask_highpass = createFrequencyMask(powerSpectrum, high_pass_cutoff_mm, 0, voxel_size);
    % apply mask to power spectrum
    powerSpectrum_highpass = powerSpectrum .* freqMask_highpass;

        % Calculate the threshold value for the 90th percentile
        thresholdValue_90th = prctile(powerSpectrum_highpass(:), 99);

        % Create a mask for the 90th percentile
        mask_90th = powerSpectrum_highpass >= thresholdValue_90th;

        % Apply the mask to the high-pass filtered power spectrum
        powerSpectrum_highpass_90th = powerSpectrum_highpass .* mask_90th;

        % Display the number of remaining prominent frequencies
        disp(['Number of remaining prominent frequencies after xxth percentile masking: ' num2str(sum(mask_90th(:)))]);

        % % Display the masked high-pass filtered power spectrum
        % display_multi_slice_around_midplanes(powerSpectrum_highpass_90th, color_axis_limit, Fx, Fy, Fz);

    powerSpectrum_templatefit_input = powerSpectrum_highpass_90th; % Here, you can chose whether to input a high-pass filtered power spectrum or the original power spectrum
    
    
    % Call the function to find the most intense orientation (template fitting)
    mostIntenseOrientation = findMostIntenseOrientation(powerSpectrum_templatefit_input, rotated_dot_grids, candidateOrientationVectors, Fx, Fy, Fz, roll_increment_deg);



        % Display the most intense orientation
        disp('Most Intense Orientation (degrees of roll):');
        disp(mostIntenseOrientation);

        % Calculate yaw and pitch of the most intense orientation vector
        yaw = atan2d(mostIntenseOrientation.frequency(2), mostIntenseOrientation.frequency(1));
        pitch = asind(mostIntenseOrientation.frequency(3));


%% Plot results so far

V2 = mostIntenseOrientation.frequency;
display3DPS(powerSpectrum_templatefit_input, V, V2); %Plot both direct FFT (V) and template-fitting (V2) orientation vectors on the power spectrum



%% Solver #3: RANSAC regression to find direction off of k-space pattern (if desired)
% % First, high-pass filter the power spectrum to isolate the desired wavelengths 

% freqMask_RANSAC = createFrequencyMask(powerSpectrum, filter_max_feature_size, 0, voxel_size);
% % apply mask to power spectrum
% powerSpectrum_RANSAC = powerSpectrum .* freqMask_RANSAC;

% % pull the strongest points, at the 99th percentile, from the power spectrum
% strongest_points = find_strongest_points(powerSpectrum_RANSAC, percentile_ProminentPoints_all_RANSAC);

% distance_threshold = 2; % voxels
% [line_dir, point_on_line] = MM_3d_grid_ransac(strongest_points,distance_threshold);


%% Plot arrow over T1
% Display the cropped image with the most intense orientation vector overlaid
% Display the cropped image with the most intense orientation vector overlaid
figure;
midSlice_cropped = round(size(imageStack, 3) / 2);
imagesc(squeeze(imageStack(:, :, midSlice_cropped)));
colormap(gray);
title('Cropped Image with Most Intense Orientation Vector Overlaid');
hold on;

% Calculate the position of the vector in the cropped image
centerX = size(imageStack, 2) / 2;
centerY = size(imageStack, 1) / 2;

% Plot the unit vector (from direct FFT) in blue
quiver(centerX, centerY, V(2) * 10, V(1) * 10, 'b', 'LineWidth', 3, 'MaxHeadSize', 2);
% Plot the most intense orientation vector in red
quiver(centerX, centerY, mostIntenseOrientation.frequency(2) * 10, mostIntenseOrientation.frequency(1) * 10, 'r', 'LineWidth', 2, 'MaxHeadSize', 2);

legend('Solver 1: Direct FFT', 'Solver 2: template fitting', 'Solver 3: RANSAC');
xlabel('Y');
ylabel('X');
hold off;

%% Inverse FFT to obtain the spatial image (if desired)
    % % Assuming 'filteredPowerSpectrum' is the frequency domain image
    % spatialImage = inverseFFT(fftResult);
    % % Display the midplane (XY) of the spatial image
    % displayMidplanes(spatialImage,x*voxel_size, y*voxel_size, z*voxel_size);





%% Archived code
% % Looking into alternate or complimentary methods: Radon transform:
%     % Perform the 3D Radon transform on the spatial image
%     % radon3D = radonTransform3D(powerSpectrum);
%     radon3D = radonTransform3D(powerSpectrum);

%     % Plot the results of the 3D Radon transform with respect to theta and phi
%     theta = 0:179;
%     phi = 0:179;

%     % Create a figure for the 3D Radon transform results
%     figure;

%     % Plot the Radon transform for a specific slice along the third dimension
%     sliceIndex = round(size(radon3D, 3) / 2);

%     % Plot the maximum intensity projection of the Radon transform
%     maxProjection = squeeze(max(radon3D, [], 3:4));
%     nexttile;
%     imagesc(maxProjection);
%     colormap(jet);
%     colorbar;
%     title('Maximum Intensity Projection of Radon Transform');
%     xlabel('Theta (degrees)');
%     ylabel('Phi (degrees)');


% % Example usage of fitOrthogonalDirections function

%     % Use the powerSpectrum as input data grid
%     dataGrid = powerSpectrum;

%     % Fit orthogonal directions to the power spectrum data grid
%     [principalDirections, eigenvalues] = fitOrthogonalDirections(dataGrid);

%     % Display the principal directions and corresponding eigenvalues
%     disp('Principal Directions:');
%     disp(principalDirections);
%     disp('Eigenvalues:');
%     disp(eigenvalues);


%% Footer
    % display that code is ended
    disp('Code finished running!');

    % Apply the formatting function to all figures with heatmaps
    figHandles = findall(0, 'Type', 'figure');
    for i = 1:length(figHandles)
        axHandles = findall(figHandles(i), 'Type', 'axes');
        for j = 1:length(axHandles)
            formatColorbarTicks(axHandles(j));
        end
    end

    % Note to the user that matlab flips its x and y axes
    disp('Note to user: Matlab runs first index down the left axis, then second left-to-right');
    disp('Note to user 2: 3D plots may look different (flipped x and y) but arrows are valid');

return
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

%% Functions

% % Callback function for the slider
% function updateSlice(~, ~)
%     sliceIdx = round(get(hSlider, 'Value'));
%     disp(['Updating slice: ' num2str(sliceIdx)]);
%     set(hImg, 'CData', imageStack(:, :, sliceIdx));
%     title(hAx, ['Slice ' num2str(sliceIdx)]);
% end
% Function to format colorbar tick labels in scientific notation
function formatColorbarTicks(ax)
    % Get the colorbar associated with the axes
    c = colorbar(ax);
    
    % Set the tick labels to scientific notation
    c.TickLabels = arrayfun(@(x) sprintf('%.1e', x), c.Ticks, 'UniformOutput', false);
end

% Function to perform inverse FFT on a frequency domain image
function spatialImage = inverseFFT(frequencyImage)
    % Perform the inverse FFT
    ifftResult = ifftn(ifftshift(frequencyImage));
    
    % Take the real part of the result (since the imaginary part should be negligible)
    spatialImage = real(ifftResult);
end

% Function to perform the Radon transform on a 3D image
function radon3D = radonTransform3D(image3D)
    % Get the size of the input 3D image
    [nx, ny, nz] = size(image3D);

    % Define the angles for the Radon transform (resolution of 2 degrees)
    theta = 0:1:179;
    phi = 0:1:179;

    % Initialize the Radon transform result
    radon3D = zeros(length(theta), length(phi), max([nx, ny, nz]), max([nx,ny,nz]));

    % Loop through each angle pair (theta, phi)
    for t = 1:length(theta)
        for p = 1:length(phi)
            % Rotate the 3D image around the z-axis by theta(t)
            rotatedImage = imrotate3(image3D, theta(t), [0, 0, 1], 'crop');

            % Rotate the 3D image around the y-axis by phi(p)
            rotatedImage = imrotate3(rotatedImage, phi(p), [0, 1, 0], 'crop');

            % Sum the rotated image along the x-axis to get the projection
            projection = squeeze(sum(rotatedImage, 1));
            size(rotatedImage);
            size(projection);
            
            % Store the result in the 3D Radon transform array
            radon3D(t, p, :,:) = projection;
        end
    end
end

% Function to fit orthogonal directions to data arranged in a grid
function [principalDirections, eigenvalues] = fitOrthogonalDirections(dataGrid)
    [nx, ny, nz] = size(dataGrid);
    [X, Y, Z] = ndgrid(1:nx, 1:ny, 1:nz);
    dataPoints = [X(:), Y(:), Z(:)];
    intensities = dataGrid(:);
    
    % Normalize the intensities to sum to 1
    intensities = intensities / sum(intensities);
    
    % Compute the weighted mean of the data points
    meanPoint = sum(dataPoints .* intensities, 1);
    
    % Center the data points by subtracting the mean
    centeredDataPoints = dataPoints - meanPoint;
    
    % Compute the weighted covariance matrix of the centered data points
    covarianceMatrix = (centeredDataPoints' * (centeredDataPoints .* intensities)) / sum(intensities);
    
    % Perform eigenvalue decomposition on the covariance matrix
    [eigenvectors, eigenvaluesMatrix] = eig(covarianceMatrix);
    
    % Extract the eigenvalues and sort them in descending order
    eigenvalues = diag(eigenvaluesMatrix);
    [eigenvalues, sortIdx] = sort(eigenvalues, 'descend');
    
    % Sort the eigenvectors according to the sorted eigenvalues
    principalDirections = eigenvectors(:, sortIdx);
    
    % Display the principal directions and corresponding eigenvalues
    fprintf('Principal Directions and Corresponding Eigenvalues:\n');
    for i = 1:length(eigenvalues)
        fprintf('Direction %d: (%.4f, %.4f, %.4f), Eigenvalue: %.4f\n', ...
            i, principalDirections(:, i), eigenvalues(i));
    end
end


function displayMidplanes(spatialImage, Fx, Fy, Fz)
    % Display the midplane (XY) of the spatial image
    midSliceX = round(size(spatialImage, 1) / 2);
    midSliceY = round(size(spatialImage, 2) / 2);
    midSliceZ = round(size(spatialImage, 3) / 2);
    % Define a consistent color axis limit for all plots
    consistentColorAxisLimit = [min(spatialImage(:)), max(spatialImage(:))];
    % consistentColorAxisLimit = [0 2.87e4]
    figure;
    tiledlayout(2, 2);

    % Display the midplane (XY)
    nexttile;
    imagesc(spatialImage(:, :, midSliceZ));
    colormap(gray);
    colorbar;

    clim(consistentColorAxisLimit);
    title('Midplane (XY)');
    xlabel('Y');
    ylabel('X');
    xticks(1:size(Fx, 1));
    yticks(1:size(Fy, 2));
    xticklabels(arrayfun(@(x) sprintf('%.2f', x), Fx(:, 1, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(y) sprintf('%.2f', y), Fy(1, :, 1), 'UniformOutput', false));

    % Display the midplane (XZ)
    nexttile;
    imagesc(squeeze(spatialImage(:, midSliceY, :)));
    colormap(gray);
    colorbar;
    clim(consistentColorAxisLimit);
    title('Midplane (XZ)');
    xlabel('Z');
    ylabel('X');
    xticks(1:size(Fx, 1));
    yticks(1:size(Fz, 3));
    xticklabels(arrayfun(@(x) sprintf('%.2f', x), Fx(:, 1, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(z) sprintf('%.2f', z), Fz(1, 1, :), 'UniformOutput', false));

    % Display the midplane (YZ)
    nexttile;
    imagesc(squeeze(spatialImage(midSliceX, :, :)));
    colormap(gray);
    colorbar;
    clim(consistentColorAxisLimit);
    title('Midplane (YZ)');
    xlabel('Z');
    ylabel('Y');
    xticks(1:size(Fy, 2));
    yticks(1:size(Fz, 3));
    xticklabels(arrayfun(@(y) sprintf('%.2f', y), Fy(1, :, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(z) sprintf('%.2f', z), Fz(1, 1, :), 'UniformOutput', false));
end

function display_multi_slice_around_midplanes(image3D, color_axis_limit,Fx,Fy,Fz)
    if isempty(color_axis_limit)
        color_axis_limit = [0 1e12];
    end
    % Display the midplane (XY) of the filtered power spectrum, and 5 planes above and below, all as tiles in the same figure
    midSlice = round(size(image3D, 3) / 2);
    figure;
    tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

    for i = -3:4
        nexttile;

        slice = image3D(:, :, midSlice + i);
        % % Set zero values to the middle of the color axis limit
        % middleValue = mean(color_axis_limit);
        % slice(slice == 0) = middleValue;
        slice(slice == 0) = NaN;
                % If using pcolor, to make a nice figure where NaN's are black: % % Add padding to slice
                % % paddedSlice = padarray(slice, [1 1], NaN, 'post');
                % % Use pcolor to display the slice and remove edge lines
                % pcolor(slice);
                % shading flat;
                imagesc(slice);
        if color_axis_limit(1) == 0 %if this is a power spectrum
                colormap(jet); %if this is a power spectrum
        else
            colormap(gray); %if this is an image (which always happen to get input with non-zero lower limit)
        end
        clim(color_axis_limit);
        set(gca,'Color','black')

        xticks(1:size(Fx, 1));
        yticks(1:size(Fy, 2));
        xticklabels(arrayfun(@(x) sprintf('%.2f', x), Fx(:, 1, 1), 'UniformOutput', false));
        yticklabels(arrayfun(@(y) sprintf('%.2f', y), Fy(1, :, 1), 'UniformOutput', false));
        % colormap(jet);
        if (i+1) == midSlice
            title(['Slice ' num2str(midSlice + i) ' (Midplane)']);
        else
            title(['Slice ' num2str(midSlice + i)]);
        end
        axis equal
    end
end

function imageStack = getimageStack(file,path)
% Check if the selected file is a TIFF stack
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.tif') || strcmpi(ext, '.tiff')
    % Read the TIFF stack
    tiffInfo = imfinfo(fullfile(path, file));
    numImages = numel(tiffInfo);
    imageStack = [];
    for k = 1:numImages
        imageStack = cat(3, imageStack, imread(fullfile(path, file), k));
    end
else
    % Get list of all DICOM files in the selected folder
    allDicomFiles = dir(fullfile(path, '*.dcm'));

    % Initialize an empty array for the DICOM stack
    imageStack = [];

    % Read and concatenate all DICOM files in the folder
    for i = 1:length(allDicomFiles)
        dicomFile = fullfile(path, allDicomFiles(i).name);
        dicomData = dicomread(dicomFile);
        imageStack = cat(3, imageStack, dicomData);
    end
end

% Display the size of the imported DICOM stack
disp('The size of the imported DICOM stack is:');
disp(size(imageStack));

% Display a message indicating successful import of DICOM stack
disp('3D image imported successfully!');
end


function display3DPS(powerSpectrum,V_1,V_2,V_3)
% Create a 3D plot of the power spectrum, with each entry scaled to the size of its magnitude
[x, y, z] = ndgrid(1:size(powerSpectrum, 1), 1:size(powerSpectrum, 2), 1:size(powerSpectrum, 3));

% Shift the axes to be zero-centered
x = x - ceil(size(powerSpectrum, 1) / 2);
y = y - ceil(size(powerSpectrum, 2) / 2);
z = z - ceil(size(powerSpectrum, 3) / 2);
% Filter the power spectrum to retain only the top 10% highest amplitudes
threshold = prctile(powerSpectrum(:), 90);
top90_PowerSpectrum = powerSpectrum;
top90_PowerSpectrum(top90_PowerSpectrum < threshold) = 0;
% Create a mask for non-zero values
nonZeroMask = top90_PowerSpectrum(:) > 0;

% Create a 3D scatter plot of the power spectrum with transparency for non-zero values
figure;
scatter3(x(nonZeroMask), y(nonZeroMask), z(nonZeroMask), 200, top90_PowerSpectrum(nonZeroMask), 'filled', 'MarkerFaceAlpha', 0.25);
colormap([linspace(1, 0, 256)', linspace(1, 0, 256)', linspace(1, 0, 256)']);

% Check if V_1 exists and plot it
if exist('V_1', 'var')
    hold on;
    V_1
        % Plot the most intense orientation vector
        quiver3(1, 1, 1, ...
            V_1(1) * 10, V_1(2) * 10, V_1(3) * 10, ...
            'LineWidth', 3, 'Color', 'b', 'MaxHeadSize', 2);
    hold off;
end

% Check if V_2 exists and plot it
if exist('V_2', 'var')
    hold on;
    V_2
    % Plot the second orientation vector
    quiver3(1, 1, 1, ...
        V_2(1) * 10, V_2(2) * 10, V_2(3) * 10, ...
        'LineWidth', 2.5, 'Color', 'r', 'MaxHeadSize', 2);
    hold off;
end

% Check if V_3 exists and plot it
if exist('V_3', 'var')
    hold on;
    V_3
    % Plot the third orientation vector
    quiver3(size(powerSpectrum, 1)/2+1, size(powerSpectrum, 2)/2+1, size(powerSpectrum, 3)/2+1, ...
        V_3(1) * 10, V_3(2) * 10, V_3(3) * 10, ...
        'LineWidth', 2, 'Color', 'g', 'MaxHeadSize', 2);
    hold off;
end


colorbar;
title('3D Power Spectrum with Transparency for Non-Zero Values');
xlabel('X');
ylabel('Y');
zlabel('Z');
caxis([0 5e11]*24/size(powerSpectrum,1));
end



% Function to save the power spectrum as a 3D .tif stack
function savePowerSpectrumAsTiff(powerSpectrum)
    % Create a subdirectory for saving the power spectrum
    outputDir = fullfile(pwd, 'PowerSpectrum');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Save the power spectrum as a 3D .tif stack
    outputFileName = fullfile(outputDir, 'PowerSpectrum.tif');
    tiffObj = Tiff(outputFileName, 'w');

    for k = 1:size(powerSpectrum, 3)
        % Convert the power spectrum slice to uint8
        slice = uint8(mat2gray(powerSpectrum(:, :, k)) * 255);

        % Set the tag structure for the TIFF file
        tagstruct.ImageLength = size(slice, 1);
        tagstruct.ImageWidth = size(slice, 2);
        tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        tagstruct.BitsPerSample = 8;
        tagstruct.SamplesPerPixel = 1;
        tagstruct.RowsPerStrip = 16;
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        tagstruct.Software = 'MATLAB';

        % Set the tags and write the slice
        tiffObj.setTag(tagstruct);
        tiffObj.write(slice);

        % Write a new directory if not the last slice
        if k < size(powerSpectrum, 3)
            tiffObj.writeDirectory();
        end
    end

    % Close the TIFF file
    tiffObj.close();
    % Display a message indicating successful saving of the power spectrum
    disp(['tiff saved as ' outputFileName]);

end

% Function to project the cumulative magnitudes of the 3D volume onto the YZ, XZ, and XY planes
function projectCumulativeMagnitudes(powerSpectrum, color_axis_limit,Fx,Fy,Fz)
    % Project the cumulative magnitudes of the 3D volume onto the YZ plane
    % Create a figure with subpanels for the projections
    % color_axis_limit = color_axis_limit*size(powerSpectrum,1)/3;
    figure;
    tiledlayout(2, 2);
    % Project the cumulative magnitudes of the 3D volume onto the XY plane
    xyProjection = squeeze(sum(powerSpectrum, 3));
    % Plot the XY projection
    nexttile;
    imagesc(xyProjection);
    xticks(1:size(Fx, 1));
    yticks(1:size(Fy, 2));
    xticklabels(arrayfun(@(x) sprintf('%.2f', x), Fx(:, 1, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(y) sprintf('%.2f', y), Fy(1, :, 1), 'UniformOutput', false));
    colormap(jet);
    colorbar;
    title('XY Projection of the Cumulative Magnitudes');
    xlabel('Y');
    ylabel('X');
    caxis(color_axis_limit);
 

    % Project the cumulative magnitudes of the 3D volume onto the XZ plane
    xzProjection = squeeze(sum(powerSpectrum, 2));
    % Plot the XZ projection
    nexttile;
    imagesc(xzProjection);
    xticks(1:size(Fx, 1));
    yticks(1:size(Fz, 3));
    xticklabels(arrayfun(@(x) sprintf('%.2f', x), Fx(:, 1, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(z) sprintf('%.2f', z), Fz(1, 1, :), 'UniformOutput', false));

    colormap(jet);
    colorbar;
    title('XZ Projection of the Cumulative Magnitudes');
    xlabel('Z');
    ylabel('X');
    caxis(color_axis_limit);

   % Project the cumulative magnitudes of the 3D volume onto the YZ plane
   yzProjection = squeeze(sum(powerSpectrum, 1));
   % Plot the YZ projection
   nexttile;
   imagesc(yzProjection);
    xticks(1:size(Fy, 2));
    yticks(1:size(Fz, 3));
    xticklabels(arrayfun(@(y) sprintf('%.2f', y), Fy(1, :, 1), 'UniformOutput', false));
    yticklabels(arrayfun(@(z) sprintf('%.2f', z), Fz(1, 1, :), 'UniformOutput', false));

   colormap(jet);
   colorbar;
   title('YZ Projection of the Cumulative Magnitudes');
   xlabel('Z');
   ylabel('Y');
   caxis(color_axis_limit);
end

function freqMask = createFrequencyMask(powerSpectrum, filter_max_feature_size, filter_min_feature_size, voxel_size)
        % Define the frequency range to isolate (in pixels)
        minFreq = 1/(filter_max_feature_size/voxel_size); % Corresponds to a period of (e.g. 8) pixels
        maxFreq = 1/(filter_min_feature_size/voxel_size); % Corresponds to a period of (e.g. 4) pixels

        % Create a frequency grid
        [nx, ny, nz] = size(powerSpectrum);
        fx = (-floor(nx/2):ceil(nx/2)-1) / nx;
        fy = (-floor(ny/2):ceil(ny/2)-1) / ny;
        fz = (-floor(nz/2):ceil(nz/2)-1) / nz;
        [Fx, Fy, Fz] = ndgrid(fx, fy, fz);

        % Compute the magnitude of the frequency vectors
        freqMagnitude = sqrt(Fx.^2 + Fy.^2 + Fz.^2);

        % Create a mask to isolate the desired frequency range
        freqMask = (freqMagnitude >= minFreq) & (freqMagnitude <= maxFreq);

        % % Visualize the freqMask
        % figure;
        % tiledlayout(2, 2);

        % % Midplane (XY)
        % nexttile;
        % imagesc(squeeze(freqMask(:, :, round(size(freqMask, 3) / 2 +1))));
        % colormap(gray);
        % colorbar;
        % title('Midplane (XY) of freqMask');
        % xlabel('Y');
        % ylabel('X');

        % % Midplane (XZ)
        % nexttile;
        % imagesc(squeeze(freqMask(:, round(size(freqMask, 2) / 2 +1), :)));
        % colormap(gray);
        % colorbar;
        % title('Midplane (XZ) of freqMask');
        % xlabel('Z');
        % ylabel('X');

        % % Midplane (YZ)
        % nexttile;
        % imagesc(squeeze(freqMask(round(size(freqMask, 1) / 2 +1), :, :)));
        % colormap(gray);
        % colorbar;
        % title('Midplane (YZ) of freqMask');
        % xlabel('Z');
        % ylabel('Y');
    end

% Function to find and display the most prominent frequencies and their orientation vectors
function [unitVector, frequency_mm, mostProminentIndex, mostProminentFreq] = findMostProminentFrequency(filteredPowerSpectrum, Fx, Fy, Fz, voxel_size, numProminentFreqs)
    % Find the most prominent frequencies in the filtered power spectrum
    % numProminentFreqs = 4; % Number of prominent frequencies to find

    
    % Flatten the filtered power spectrum and sort by magnitude
    [sortedValues, sortedIndices] = sort(filteredPowerSpectrum(:), 'descend');

    % Get the indices of the most prominent frequencies
    prominentIndices = sortedIndices(1:numProminentFreqs);

        % Compute the corresponding frequency vectors
    prominentFreqs = [Fx(prominentIndices), Fy(prominentIndices), Fz(prominentIndices)];


    % Filter candidate orientation vectors to a half-space
    % Define the normal vector of the half-space (e.g., along the Z-axis)
    normalVector_halfspace = [1, 0, 0];


    % Calculate the dot product of each candidate orientation vector with the normal vector
    dotProducts = prominentFreqs * normalVector_halfspace';

    % Keep only the orientation vectors that have a positive dot product (i.e., in the same half-space as the normal vector)
    prominentFreqs = prominentFreqs(dotProducts > 0, :);
    prominentIndices = prominentIndices(dotProducts > 0);
    sortedValues = sortedValues(dotProducts > 0);
    numProminentFreqs = (size(prominentFreqs,1));
     
    
    % Convert the linear indices to subscripts
    [prominentX, prominentY, prominentZ] = ind2sub(size(filteredPowerSpectrum), prominentIndices);




    % Print a table with the most prominent frequencies and their orientation vectors
    fprintf('Most Prominent Frequencies and Their Orientation Vectors (after applying hollow sphere & half-space filter):\n');
    fprintf('Index\tFrequency Magnitude\tOrientation Vector (Fx, Fy, Fz)\n');
    for i = 1:numProminentFreqs
        fprintf('%d\t%.2e\t\t\t(%.2f, %.2f, %.2f)\n', i, sortedValues(i), prominentFreqs(i, 1), prominentFreqs(i, 2), prominentFreqs(i, 3));
    end

    % Determine the orientation vector as a unit vector for the most prominent frequency
    mostProminentFreq = prominentFreqs(1, :);
    magnitude = norm(mostProminentFreq);
    unitVector = mostProminentFreq / magnitude;

    % Compute the frequency in millimeters
    domain_size = size(filteredPowerSpectrum, 1);
    frequency_mm = (1/magnitude) * voxel_size;

    % Compute the rotation angles (in degrees) of the unit vector
    % Use euler angles (ZYX) to represent the rotation
    % The first angle is the rotation about the Z-axis (yaw)
    % The second angle is the rotation about the Y-axis (pitch)
    % The third angle is the rotation about the X-axis (roll)
    yaw = atan2d(unitVector(2), unitVector(1));
    pitch = asind(unitVector(3));
    roll = 0; % Assume no roll for now

    % Display the orientation vector and corresponding rotation angles
    fprintf('Orientation Unit Vector: (%.2f, %.2f, %.2f)\n', unitVector(1), unitVector(2), unitVector(3));
    fprintf('Rotation Angles (Yaw, Pitch, Roll): (%.2f, %.2f, %.2f)\n', yaw, pitch, roll);

    % Define the index of the most prominent frequency
    mostProminentIndex = [prominentX(1), prominentY(1), prominentZ(1)];
    % Display the index of the most prominent frequency
    % disp('Index of the most prominent frequency:');
    % disp(mostProminentIndex);
end

% function [V_list, ProminentIndices, ProminentFreqs] = collect_prominent_Frequencies(filteredPowerSpectrum, Fx, Fy, Fz, voxel_size, percentile_ProminentFreqs)
%     % Find the most prominent frequencies in the filtered power spectrum
%     % numProminentFreqs = 4; % Number of prominent frequencies to find

%     % Flatten the filtered power spectrum and sort by magnitude
%     [sortedValues, sortedIndices] = sort(filteredPowerSpectrum(:), 'descend');

%     % Get the indices of the most prominent frequencies
%     prominentIndices = sortedIndices(1:numProminentFreqs);
%     % Get the threshold value for the specified percentile
%     thresholdValue = prctile(sortedValues, percentile_ProminentFreqs);

%     % Find the indices of the frequencies that are above the threshold
%     prominentIndicesWithinPercentile = sortedIndices(sortedValues >= thresholdValue);

%     % Convert the linear indices to subscripts
%     [prominentX_within, prominentY_within, prominentZ_within] = ind2sub(size(filteredPowerSpectrum), prominentIndicesWithinPercentile);

%     % Compute the corresponding frequency vectors
%     prominentFreqsWithinPercentile = [Fx(prominentIndicesWithinPercentile), Fy(prominentIndicesWithinPercentile), Fz(prominentIndicesWithinPercentile)];

%     % Store the results in the output variables
%     V_list = prominentFreqsWithinPercentile;
%     ProminentIndices = [prominentX_within, prominentY_within, prominentZ_within];
%     ProminentFreqs = sortedValues(sortedValues >= thresholdValue);

%     % Compute the corresponding frequency vectors
%     ProminentFreqs = [Fx(ProminentIndices), Fy(ProminentIndices), Fz(ProminentIndices)];

%     % Determine the orientation vector as a unit vector for the most prominent frequency
%     magnitude = norm(ProminentFreqs);

%     % display size of prominentFreqs and magnitude
%     disp('Size of prominentFreqs');
%     disp(size(ProminentFreqs));
%     disp('Magnitude of prominentFreqs');
%     disp(magnitude);

%     V_list = ProminentFreqs./magnitude;

%     % % Display the orientation vector and corresponding rotation angles
%     % fprintf('Orientation Unit Vector: (%.2f, %.2f, %.2f)\n', unitVector(1), unitVector(2), unitVector(3));
%     % fprintf('Rotation Angles (Yaw, Pitch, Roll): (%.2f, %.2f, %.2f)\n', yaw, pitch, roll);

% end


function imageStack = cropCubeToSphere(imageStack)
    % Create a spherical mask inscribed in the cubic volume
    [x, y, z] = ndgrid(1:size(imageStack, 1), 1:size(imageStack, 2), 1:size(imageStack, 3));
    centerX = size(imageStack, 1) / 2;
    centerY = size(imageStack, 2) / 2;
    centerZ = size(imageStack, 3) / 2;
    radius = min([centerX, centerY, centerZ]);

    % Create the spherical mask
    sphericalMask = sqrt((x - centerX).^2 + (y - centerY).^2 + (z - centerZ).^2) <= radius;

    % Apply the spherical mask to the image stack
    maskedImageStack = imageStack .* sphericalMask;

    % Calculate the average intensity within the mask
    averageIntensity = mean(maskedImageStack(sphericalMask));

    % Replace zero values with the average intensity within the mask
    imageStack(~sphericalMask) = averageIntensity;
end


function strongest_points = find_strongest_points(powerSpectrum, percentile)
    % Find the threshold value for the given percentile
    thresholdValue = prctile(powerSpectrum(:), percentile);

    % Create a mask for the strongest points
    strongest_points_mask = powerSpectrum >= thresholdValue;

    % Get the indices of the strongest points
    [x, y, z] = ind2sub(size(powerSpectrum), find(strongest_points_mask));

    % Combine the indices into a single array
    strongest_points = [x, y, z];
end


function [rotated_dot_grids] = rotateDotGrid(coordinates, roll_increment_deg, dot_radius, domainSize, dot_spacing)
    % Define the size of the grid
    

    %  % Initialize a 3D array to store the rotated dot grids
    %  rotated_dot_grids = zeros(size(coordinates, 1), size(coordinates, 2), ceil(90 / roll_increment_deg) + 1);
   

    % Initialize a 3D array to store the rotated dot grids
    numRotations = ceil(90 / roll_increment_deg) + 1;
    rotated_dot_grids = zeros(domainSize, domainSize, numRotations);

    % Loop through the rotations from 0 to 90 degrees in roll_increment_deg increments
    for roll_deg = 0:roll_increment_deg:90
        % Rotate the coordinates by the specified angle
        rotatedCoordinates = rotateCoordinates(coordinates, roll_deg);

        % Center the grid at the center of the image
        centerX = domainSize / 2;
        centerY = domainSize / 2;

        % Translate the rotated coordinates to the center of the grid
        rotatedCoordinates(:, 1) = rotatedCoordinates(:, 1)-mean(rotatedCoordinates(:, 1)) + centerX +1;
        rotatedCoordinates(:, 2) = rotatedCoordinates(:, 2)-mean(rotatedCoordinates(:, 2)) + centerY +1;

        % Create a grid with dots at the rotated coordinates
        dot_grid = zeros(domainSize, domainSize);
        for i = 1:size(rotatedCoordinates, 1)
            x = (rotatedCoordinates(i, 1)) ;
            y = (rotatedCoordinates(i, 2)) ;
            [X, Y] = ndgrid(1:domainSize, 1:domainSize);
            mask = sqrt((X - x).^2 + (Y - y).^2) <= dot_radius;
            dot_grid(mask) = 1;
        end

        % Store the rotated dot grid in the 3D array
        rotated_dot_grids(:, :, round(roll_deg / roll_increment_deg) + 1) = dot_grid;
    end
end


   


function mostIntenseOrientation = findMostIntenseOrientation(PowerSpectrum, rotated_dot_grids, V, Fx, Fy, Fz, roll_increment_deg, axis_for_projection)
    % Iterate through the most promising orientations, re-orient the computer's 'view' to each orientation, and fit template of dots (checking at 0-90 degree roll angles)

    % Initialize variables to store the maximum intensity and corresponding orientation
    maxIntensity = -Inf;
    mostIntenseOrientation = struct('frequency', [], 'orientation', [], 'intensity', []);
    do_all_figures = false; % Set to false to only display the most intense orientations (recommend setting to false normally)
    % Loop through each prominent frequency
    for i = 1:size(V, 1)
        % Get the orientation vector for the current frequency
        orientationVector = V(i, :);

        % Rotate the power spectrum according to the direction of the frequency
        rotatedPowerSpectrum = rotatePowerSpectrum(PowerSpectrum, orientationVector, Fx, Fy, Fz);

        % Project the resultant power spectrum in its new x-direction
        projectedPowerSpectrum = squeeze(sum(rotatedPowerSpectrum, 1));

        % Measure the intensity of points in the projection that align with each dot grid
        for j = 1:size(rotated_dot_grids, 3)
            dotGrid = rotated_dot_grids(:, :, j);
            intensity = sum(projectedPowerSpectrum(dotGrid == 1), 'all');

            % Update the maximum intensity and corresponding orientation if needed
            if intensity > maxIntensity
                maxIntensity = intensity;
                mostIntenseOrientation.frequency = orientationVector;
                mostIntenseOrientation.orientation = (j - 1) * roll_increment_deg;
                mostIntenseOrientation.intensity = intensity;
                            % Create a new figure for each max intensity found
                figure;
                tiledlayout(2, 2);

                % First panel: Projected power spectrum
                nexttile;
                imagesc(projectedPowerSpectrum);
                colormap(jet);
                colorbar;
                title(['Projection with Orientation Vector: (', num2str(orientationVector), ') and Rotation: ', num2str((j - 1) * roll_increment_deg), ' degrees']);
                xlabel('Y');
                ylabel('Z');

                % Second panel: Corresponding grid of dots
                nexttile;
                imagesc(dotGrid);
                colormap(gray);
                colorbar;
                title(['Dot Grid with Rotation: ', num2str((j - 1) * roll_increment_deg), ' degrees']);
                xlabel('Y');
                ylabel('Z');

                % Third panel: Combined image
                nexttile;
                combinedImage = projectedPowerSpectrum.*dotGrid;
                imagesc(combinedImage);
                colormap(gray);
                colorbar;
                title('Combined Image');
                xlabel('Y');
                ylabel('Z');

            else
                if (i == 8 && j < 10) && do_all_figures
                        figure;
                    tiledlayout(2, 2);

                    % First panel: Projected power spectrum
                    nexttile;
                    imagesc(projectedPowerSpectrum);
                    colormap(jet);
                    colorbar;
                    title(['Specified Projection with Orientation Vector: (', num2str(orientationVector), ') and Rotation: ', num2str((j - 1) * roll_increment_deg), ' degrees']);
                    xlabel('Y');
                    ylabel('Z');


                    % Second panel: Corresponding grid of dots
                    nexttile;
                    imagesc(dotGrid);
                    colormap(gray);
                    colorbar;
                    title(['Dot Grid with Rotation: ', num2str((j - 1) * roll_increment_deg), ' degrees']);
                    xlabel('Y');
                    ylabel('Z');

                    % Third panel: Combined image
                    nexttile;
                    combinedImage = projectedPowerSpectrum.*dotGrid;
                    imagesc(combinedImage);
                    colormap(gray);
                    colorbar;
                    title('Combined Image');
                    xlabel('Y');
                    ylabel('Z');
                end

            end

        end
    end

    % % Display the most intense orientation
    % disp('Most Intense Orientation:');
    % disp(mostIntenseOrientation);
end

function rotatedPowerSpectrum = rotatePowerSpectrum(powerSpectrum, orientationVector, Fx, Fy, Fz)
    % Compute the rotation matrix to align the orientation vector with the x-axis
    targetVector = [1, 0, 0];
    B_0 = [1, 0, 0; 0, 1, 0; 0, 0, 1];
    % define vector perpendicular to orientationVector, and parallel to the XY plane
    orientationVector_2 = cross(orientationVector, [0, 0, 1]);
    orientationVector_2 = orientationVector_2 / norm(orientationVector_2); % Normalize the vector
    orientationVector_3 = cross(orientationVector, orientationVector_2);
    B_rotated_orientation = [orientationVector; orientationVector_2; orientationVector_3];

    % B_rotated_orientation = [orientationVector; ]
    Q = B_0*B_rotated_orientation';
    rotationMatrix = Q';
    % rotationMatrix = vrrotvec2mat(vrrotvec(orientationVector, targetVector));

    % Create a grid of coordinates
    [nx, ny, nz] = size(powerSpectrum);

    % Determine the rotation angles about the z and y axes from the orientation vector

    

    % Determine the Euler ZYX rotation angles (assuming no roll) from the orientation vector
    yaw = atan2d(orientationVector(2), orientationVector(1)); % Rotation about the Z-axis
    pitch = asind(orientationVector(3)); % Rotation about the Y-axis
    roll = 0; % Assume no roll

    % phi = asind(orientationVector(3)); % Rotation about the y-axis
    % theta = atan2d(orientationVector(2), orientationVector(1)); % Rotation about the z-axis
    rotatedPowerSpectrum = imrotate3(powerSpectrum, yaw, [0, 0, -1], 'crop'); % Rotate about z-axis (needs to be [0 0 -1] for z because imrotate3 is a bit ass-backwards)
    rotatedPowerSpectrum = imrotate3(rotatedPowerSpectrum, pitch, [1, 0, 0], 'crop'); % Rotate about y-axis (needs to be [0 -1 0] for y because imrotate3 is a bit ass-backwards)

    % Rotate the power spectrum using imrotate3



    % % [X, Y, Z] = ndgrid(1:nx, 1:ny, 1:nz);

    % % Apply the rotation matrix to the coordinates
    % coords_rotated = [Fx(:), Fy(:), Fz(:)] * rotationMatrix';

    % % Reshape the rotated coordinates back to the original grid shape
    % Xq = reshape(coords_rotated(:, 1), [nx, ny, nz]);
    % Yq = reshape(coords_rotated(:, 2), [nx, ny, nz]);
    % Zq = reshape(coords_rotated(:, 3), [nx, ny, nz]);

    % disp('sizes of Xq, Yq, Zq');
    % disp(size(Xq));
    % disp(size(Yq));
    % disp(size(Zq));
    % disp('Sizes of powerSpectrum, Fx, Fy, Fz');
    % disp(size(powerSpectrum));
    % disp(size(Fx));
    % disp(size(Fy));
    % disp(size(Fz));

    % rotatedPowerSpectrum = interp3(Fx, Fy, Fz, powerSpectrum, Xq,Yq,Zq, 'linear', 0);
    % rotatedPowerSpectrum = reshape(rotatedPowerSpectrum, [nx, ny, nz]);
end

function rotatedCoordinates = rotateCoordinates(coordinates, deg_n)
    % Convert the angle from degrees to radians
    theta = deg2rad(deg_n);

    % Define the rotation matrix for rotation around the Z-axis
    rotationMatrix = [cos(theta), -sin(theta); sin(theta), cos(theta)];

    % Apply the rotation matrix to the coordinates
    rotatedCoordinates = (rotationMatrix * coordinates')';
end
