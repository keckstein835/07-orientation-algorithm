%% Header --------------------------------------------------------------------------
% Image_grid_3D_FFT_KNE.m
% Author: Kevin Eckstein
% Date: 2024-12-30
%
% Description:
% Function that accepts a 3D image, along with some information about that image, and returns an orientation vector; meant to work on MRI T1 images of 3D printed anisotropic rectangular lattice materials embedded in an MR-visible matrix, where the lattice isn't very clear but a direction may be recognized when squinting your eyes.
% Written to run from the main file "Find_3D_lattice_fiber_dir_KNE.m"; the main function iteratues through a full volume (while this function only looks at a sub-volume, or whatever you give it)
% This function performs a 3D Fast Fourier Transform (FFT) on the stack, and plots the
% resulting 3D power spectrum
% The 3D power spectrum forms a 3D grid of intensity spikes; this algorithm finds the orientation of that grid by projecting the at various angles and fitting a 2D grid of dots to the projection.

%
% Usage:
%
%    % Note to the user that matlab flips its x and y axes
% disp('Note to user: Matlab runs first index down the left axis, then second left-to-right');
% disp('Note to user 2: 3D plots may look different (flipped x and y) but arrows are valid');
%--------------------------------------------------------------------------

%% function definition -----------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% IMAGE_GRID_3D_FFT_KNE - Computes the orientation vector and fractional anisotropy (FA) from a sub-volume of a 3D image stack using FFT and template-fitting.
%
% Syntax:  [V_orientation, R_basis, FA] = Image_grid_3D_FFT_KNE(imageStack, mask, crop_center_Location_voxels_from_origin, cropWidth_voxels, voxel_size)
%
% Inputs:
%    imageStack - 3D matrix representing the domain (volume) to be analyzed.
%    periodicity_mm - Scalar specifying the periodicity of the structure in mm (e.g. 2mm unit cells).
%    voxel_size - Scalar specifying the size of each voxel
%   do_primary_plots - Boolean specifying whether to display primary plots (default = false)
%   do_secondary_plots - Boolean specifying whether to display secondary plots (default = false)
%
% Outputs:
%    V_orientation - Most dominant orientation vector within subvolume
%    R_basis - Rotation matrix for the orientation vector (including lateral symmetric directions)
%    FA - a "strength of signal" measure that doesn't really work. (unfortunately FA is rather meaningless here, it doesn't accurately represent strength of signal)
function [V_orientation, R_basis,FA] = Image_grid_3D_FFT_KNE(imageStack, voxel_size, periodicity_mm, do_primary_plots, do_secondary_plots);

addpath('functions'); % Add the functions folder to the path (not sure if this is needed)

% Check if voxel_size and periodicity_mm are non-zero integers
if ~isscalar(voxel_size) || voxel_size <= 0
    error('voxel_size must be a non-zero positive number.');
end

if ~isscalar(periodicity_mm) || periodicity_mm <= 0
    error('periodicity_mm must be positive number.');
end

% Set default values for do_primary_plots and do_secondary_plots if they are not passed into function
if ~exist('do_primary_plots', 'var')
    do_primary_plots = false;
end
if ~exist('do_secondary_plots', 'var')
    do_secondary_plots = false;
end

disp(['Image_grid_3D_FFT_KNE.m started: {time:} ' char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"))]);


%% User settings
    % solve_method = 1; % 1 = FFT direct, 2 = FFT with template fitting, 3 = RANSAC, 4 = Radon transform(?) 
    % FFT with template fitting; the other methods don't work as well -KNE 2024-12-30
    % 
    % Decide if you're outputting plots:
    
    do_print_results_to_command_window = false; % print results to command window

 
    % voxel_size = 0.5; % mm
    color_axis_limit = [0 1e13];

    % Define filter_max_feature_size and filter_min_feature_size; this is where a different sized unit-cell or periodic structure (i.e. differing periodicity) can be used
    % - I generalized this so that it could accept other unit cell sizes but so far it's only been tested on 2 mm vintile unit-cell structures -KNE 2024-12-30
   
    % filter_max_feature_size = 2.6; %mm; 2.6 is good
    % filter_min_feature_size = 2; %mm; 2.0 is good
    % filter_max_feature_size = 2.6; %mm; 2.6 is good for scaled
    % filter_min_feature_size = 1.9; %mm; 1.9 is good for scaled
    filter_max_feature_size = 2.5 * periodicity_mm/2; %mm; for Xbox. 
    filter_min_feature_size = 1.9 * periodicity_mm/2; %mm; for Xbox
    
    
    percentile_ProminentFreqs = 99; % percentile of the most prominent frequencies, within feature size filter, to use as candidates for orientation vector voting procedure. 94 seems pretty good as a number
        % higher percentile will make code go faster but may miss the correct orientation vector if the FFT signal is weak. 98 seems reliable so far
    threshold_percentile = 90; % percentile of k-space intensities to use in template fitting (not sure this is necessary, as the strongest freq's should dominate, but here it is anyway)
    % 90 slower but more accurate, 98 is faster
    roll_increment_deg = 2; % degrees; Adjust as needed; for rotating template during template fitting (lower number is more accurate and expensive)
    dot_radius_factor = 0.9; % [voxels] Set to 1 for a reasonable dot radius for template fitting, but can reduce if needed for accuracy. 0.9 seems good.
    
    high_pass_cutoff_mm = 3 * periodicity_mm/2; % mm; cutoff for high-pass filter to remove low-frequency noise (3 mm seems good)
    % percentile_ProminentPoints_all_RANSAC = 98; % percentile of most prominent points to use for RANSAC regression

    do_save_PS_tif = false; % save the power spectrum as a 3D .tif stack




% Display the size of the cropped DICOM stack
% disp('The size of the cropped DICOM stack is:');
domainSize = size(imageStack, 1);
% disp(size(imageStack));

midSlice = round(size(imageStack, 3) / 2);

% Create simple grids x, y, z the size of imageStack for plotting
[x, y, z] = ndgrid(1:size(imageStack, 1), 1:size(imageStack, 2), 1:size(imageStack, 3));


%% Crop cube to a sphere, if desired
% imageStack = cropCubeToSphere(imageStack); % 2024-10-24: this doesn't seem to help at all (I though other papers did this but maybe not)

%% Display the cropped image
if do_primary_plots
    displayMidplanes(imageStack, x*voxel_size, y*voxel_size, z*voxel_size);
    display_multi_slice_around_midplanes(imageStack, [min(imageStack(:)) max(imageStack(:))], x*voxel_size, y*voxel_size, z*voxel_size);
end


%% Apply a Tukey window to the image stack, if desired
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
if do_primary_plots
    display3DPS(powerSpectrum);
end

% Display the cumulative magnitudes of the 3D volume projected onto the XY, XZ, and YZ planes
if do_primary_plots
    projectCumulativeMagnitudes(powerSpectrum, color_axis_limit, Fx, Fy, Fz);
end

%% Save the power spectrum as a 3D .tif stack, if desired
if do_save_PS_tif
    savePowerSpectrumAsTiff(powerSpectrum);
end

%% Frequency filtering: isolating ~2-2.5 mm wavelengths
% filter_max_feature_size = 2.6; %mm
% filter_min_feature_size = 2; %mm

freqMask = createFrequencyMask(powerSpectrum, filter_max_feature_size, filter_min_feature_size, voxel_size);

% Apply the mask to the power spectrum
filteredPowerSpectrum = powerSpectrum .* freqMask;

% display_multi_slice_around_midplanes(powerSpectrum, color_axis_limit, Fx,Fy,Fz); % display the power spectrum
if do_primary_plots
    display_multi_slice_around_midplanes(filteredPowerSpectrum, color_axis_limit, Fx,Fy,Fz); % display the filtered power spectrum
end
freqMask_double = double(freqMask);
% display_multi_slice_around_midplanes(freqMask_double, [0 1], Fx,Fy,Fz); % display the mask

% Check if the filtered power spectrum is empty
if isnan(sum(filteredPowerSpectrum(:))) || sum(filteredPowerSpectrum(:)) == 0
    disp('Filtered power spectrum is empty.');
    V_orientation = [1 0 0]; % return a default orientation vector
    return
end

% sum(filteredPowerSpectrum(:))
% isnan(sum(filteredPowerSpectrum(:)))
% return

% if ( sum(imageStack(:)) == 0 || isnan(sum(imageStack(:))) )
%     disp('Masked image stack is empty.');
%     V_orientation = [1 0 0]; % return a default orientation vector
%     return
% end

%% Solver #1: Direct FFT: Find the most prominent frequencies and their orientation vectors from the filtered power spectrum
[V_1, lambda, mostProminentIndex, mostProminentFreq]  =  findMostProminentFrequency(filteredPowerSpectrum, Fx, Fy, Fz, voxel_size);
% display the returned values, descripttion
% disp('Orientation Unit Vector: ');
% disp(V);
if do_print_results_to_command_window
    disp('Frequency in mm: ');
    disp(lambda);
end


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


if do_secondary_plots
    display_multi_slice_around_midplanes(filteredPowerSpectrum_99th, color_axis_limit, Fx,Fy,Fz); % display the filtered power spectrum
end

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
if do_secondary_plots == true
    figure;
    tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    for i = 1:9
        nexttile;
        interval = 1;
        imagesc(rotated_dot_grids(:, :, i*interval-interval+1));
        colormap(gray);
        colorbar;
        title(['Rotation: ' num2str((i*interval-interval) * roll_increment_deg) ' degrees']);
        xlabel('Y');
        ylabel('X');
    end
end

% high_pass_cutoff_mm = 3; %could be, for example, 4
freqMask_highpass = createFrequencyMask(powerSpectrum, high_pass_cutoff_mm, 0, voxel_size);
% apply mask to power spectrum
powerSpectrum_highpass = powerSpectrum .* freqMask_highpass;
    
% Calculate the threshold value for the 90th percentile
thresholdValue_90th = prctile(powerSpectrum_highpass(:), threshold_percentile);

% Create a mask for the 90th percentile
mask_90th = powerSpectrum_highpass >= thresholdValue_90th;

% Apply the mask to the high-pass filtered power spectrum
powerSpectrum_highpass_90th = powerSpectrum_highpass .* mask_90th;

% Display the number of remaining prominent frequencies
% disp(['Number of remaining prominent frequencies after xxth percentile masking: ' num2str(sum(mask_90th(:)))]);

% % Display the masked high-pass filtered power spectrum
% display_multi_slice_around_midplanes(powerSpectrum_highpass_90th, color_axis_limit, Fx, Fy, Fz);

powerSpectrum_templatefit_input = powerSpectrum_highpass_90th; % Here, you can chose whether to input a high-pass filtered power spectrum or the original power spectrum


% Call the function to find the most intense orientation (template fitting)
[mostIntenseOrientation] = findMostIntenseOrientation(powerSpectrum_templatefit_input, rotated_dot_grids, candidateOrientationVectors, Fx, Fy, Fz, roll_increment_deg,do_secondary_plots);


    if do_print_results_to_command_window
        % Display the most intense orientation
        disp('Most Intense Orientation from template fitting: ');
        disp(mostIntenseOrientation);
    end
    % Calculate yaw and pitch of the most intense orientation vector
    yaw = atan2d(mostIntenseOrientation.frequency(2), mostIntenseOrientation.frequency(1));
    pitch = asind(mostIntenseOrientation.frequency(3));


%% Plot results so far

V_2 = mostIntenseOrientation.frequency;
roll_angle_V_2 = mostIntenseOrientation.orientation;

normalized_V2_power = mostIntenseOrientation.normalized_intensity;
if do_primary_plots
    display3DPS(powerSpectrum_templatefit_input, V_1, V_2); %Plot both direct FFT (V) and template-fitting (V2) orientation vectors on the power spectrum
end
% Define the basis vectors using yaw, pitch, and roll_angle_V_2

yaw_rad = deg2rad(yaw);
pitch_rad = -deg2rad(pitch); %Don't ask why this is negative, it just is (must be different conventions in matlab somehwere in here)
roll_rad = deg2rad(roll_angle_V_2);

% Rotation matrix for yaw (Z-axis rotation)
R_yaw = [cos(yaw_rad), -sin(yaw_rad), 0;
         sin(yaw_rad), cos(yaw_rad), 0;
         0, 0, 1];

% Rotation matrix for pitch (Y-axis rotation)
R_pitch = [cos(pitch_rad), 0, sin(pitch_rad);
           0, 1, 0;
           -sin(pitch_rad), 0, cos(pitch_rad)];

% Rotation matrix for roll (X-axis rotation)
R_roll = [1, 0, 0;
          0, cos(roll_rad), -sin(roll_rad);
          0, sin(roll_rad), cos(roll_rad)];

% Combined rotation matrix
R_basis = R_yaw * R_pitch * R_roll;


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
if do_primary_plots
    figure;
    midSlice_cropped = floor(size(imageStack, 3) / 2)+1;
    imagesc(squeeze(imageStack(:, :, midSlice_cropped)));
    colormap(gray);
    title('Cropped Image with Most Intense Orientation Vector Overlaid');
    hold on;

    % Calculate the position of the vector in the cropped image
    centerX = size(imageStack, 2) / 2;
    centerY = size(imageStack, 1) / 2;

    % Plot the unit vector (from direct FFT) in blue
    quiver(centerX, centerY, V_1(2) * 10, V_1(1) * 10, 'b', 'LineWidth', 3, 'MaxHeadSize', 2);
    % Plot the most intense orientation vector in red
    quiver(centerX, centerY, mostIntenseOrientation.frequency(2) * 10, mostIntenseOrientation.frequency(1) * 10, 'r', 'LineWidth', 2, 'MaxHeadSize', 2);

    legend('Solver 1: Direct FFT', 'Solver 2: template fitting');
    xlabel('Y');
    ylabel('X');
    hold off;
end


%% Inverse FFT to obtain the spatial image (if desired)
    % % Assuming 'filteredPowerSpectrum' is the frequency domain image
    % spatialImage = inverseFFT(fftResult);
    % % Display the midplane (XY) of the spatial image
    % displayMidplanes(spatialImage,x*voxel_size, y*voxel_size, z*voxel_size);




%% Conclusion: Define outputs
V_orientation = V_2; % orientation vector from FFT and template fitting
FA = normalized_V2_power; % fractional anisotropy




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
    % disp('Code finished running!');

    if do_primary_plots
        disp('V_1 is direct FFT solve, while V_2 is from template fitting; V_2 is what gets returned by function -KNE');
        % Apply the formatting function to all figures with heatmaps
        figHandles = findall(0, 'Type', 'figure');
        for i = 1:length(figHandles)
            axHandles = findall(figHandles(i), 'Type', 'axes');
            for j = 1:length(axHandles)
                formatColorbarTicks(axHandles(j));
            end
        end
    end

    % % Note to the user that matlab flips its x and y axes
    % disp('Note to user: Matlab runs first index down the left axis, then second left-to-right');
    % disp('Note to user 2: 3D plots may look different (flipped x and y) but arrows are valid');

end
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
        clim(color_axis_limit);
    

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
        clim(color_axis_limit);

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
    clim(color_axis_limit);
    
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
    end

% Function to find and display the most prominent frequencies and their orientation vectors
function [unitVector, frequency_mm, mostProminentIndex, mostProminentFreq] = findMostProminentFrequency(filteredPowerSpectrum, Fx, Fy, Fz, voxel_size)
    % Find the most prominent frequencies in the filtered power spectrum
    % numProminentFreqs = 4; % Number of prominent frequencies to find
    numProminentFreqs = 2;
    print_results = false; % Print the results to the command window
    % Flatten the filtered power spectrum and sort by magnitude
    [sortedValues, sortedIndices] = sort(filteredPowerSpectrum(:), 'descend');

    % Get the indices of the most prominent frequencies
    prominentIndices = sortedIndices(1:numProminentFreqs);
    
        % Compute the corresponding frequency vectors
    prominentFreqs = [Fx(prominentIndices), Fy(prominentIndices), Fz(prominentIndices)];
    prominentFreqs;

    % Filter candidate orientation vectors to a half-space
    % Define the normal vector of the half-space (e.g., along the Z-axis)
    normalVector_halfspace = [1, 0, 0];


    % Calculate the dot product of each candidate orientation vector with the normal vector
    dotProducts = prominentFreqs * normalVector_halfspace';
    if sum(dotProducts > 0) == 0
        normalVector_halfspace = [0, 1, 0];
        dotProducts = prominentFreqs * normalVector_halfspace'; %just an exception for when prominent frequencies lie at edge of Y half-space
    end
    if sum(dotProducts > 0) == 0
        normalVector_halfspace = [0, 0, 1];
        dotProducts = prominentFreqs * normalVector_halfspace'; %just an exception for when prominent frequencies lie at edge of X and Y half-space
    end

    % Keep only the orientation vectors that have a positive dot product (i.e., in the same half-space as the normal vector)
    prominentFreqs = prominentFreqs(dotProducts > 0, :);
    prominentIndices = prominentIndices(dotProducts > 0);
    sortedValues = sortedValues(dotProducts > 0);
    numProminentFreqs = (size(prominentFreqs,1));
     
    
    % Convert the linear indices to subscripts
    [prominentX, prominentY, prominentZ] = ind2sub(size(filteredPowerSpectrum), prominentIndices);




    % % Print a table with the most prominent frequencies and their orientation vectors
    % if print_results 
    %     fprintf('Most Prominent Frequencies and Their Orientation Vectors (after applying hollow sphere & half-space filter):\n');
    %     fprintf('Index\tFrequency Magnitude\tOrientation Vector (Fx, Fy, Fz)\n');
    %     for i = 1:numProminentFreqs
    %         fprintf('%d\t%.2e\t\t\t(%.2f, %.2f, %.2f)\n', i, sortedValues(i), prominentFreqs(i, 1), prominentFreqs(i, 2), prominentFreqs(i, 3));
    %     end
    % end
    

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

    % % Display the orientation vector and corresponding rotation angles
    % fprintf('Orientation Unit Vector: (%.2f, %.2f, %.2f)\n', unitVector(1), unitVector(2), unitVector(3));
    % fprintf('Rotation Angles (Yaw, Pitch, Roll): (%.2f, %.2f, %.2f)\n', yaw, pitch, roll);

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

function mostIntenseOrientation = findMostIntenseOrientation(PowerSpectrum, rotated_dot_grids, V, Fx, Fy, Fz, roll_increment_deg, do_plot)
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
        rotatedPowerSpectrum = rotatePowerSpectrum(PowerSpectrum, orientationVector, Fx, Fy, Fz, do_plot);

        % Project the resultant power spectrum in its new x-direction
        projectedPowerSpectrum = squeeze(sum(rotatedPowerSpectrum, 1));

        % Measure the intensity of points in the projection that align with each dot grid
        for j = 1:size(rotated_dot_grids, 3)
            dotGrid = rotated_dot_grids(:, :, j);
            intensity = sum(projectedPowerSpectrum(dotGrid == 1), 'all');
            % Measure the total intensity sum
            totalIntensitySum = sum(projectedPowerSpectrum(:));


        % Update the maximum intensity and corresponding orientation if needed
        
            if intensity > maxIntensity
                maxIntensity = intensity;
                mostIntenseOrientation.frequency = orientationVector;
                mostIntenseOrientation.orientation = (j - 1) * roll_increment_deg;
                mostIntenseOrientation.intensity = intensity;
                mostIntenseOrientation.normalized_intensity = intensity/totalIntensitySum;
                            % Create a new figure for each max intensity found
                if do_plot
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
                end

            else
                if (i == 8 && j < 10) && do_all_figures % a little switch for debugging here, don't mind this if-statement
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
    



function rotatedPowerSpectrum = rotatePowerSpectrum(powerSpectrum, orientationVector, Fx, Fy, Fz, do_plot)
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
