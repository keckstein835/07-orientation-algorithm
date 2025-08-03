% load DTI.mat and DTIregs.mat files; 
% DTI.mat has the automatically fitted orientations, while DTIregs.mat will be treated as ground-truth.
% close all

disp('Please select the DTI.mat file:');
[dtiFile, dtiPath] = uigetfile('*.mat', 'Select DTI.mat');
if isequal(dtiFile,0)
    error('No DTI.mat file selected.');
end
load(fullfile(dtiPath, dtiFile));


% Extract vectors from DTIregs.mat (assuming variable name is 'DTI' and is Nx3)
vectors = V1; %4D (width x length x height x 3) vector field

% Plot quiver plots of vectors (projected onto 2D slices)
figure;
sliceIdx = round(size(vectors,3)/2); % Middle slice in z-direction
quiverScale = 1; % Adjust as needed

% Extract the slice
vx = squeeze(vectors(:,:,sliceIdx,1));
vy = squeeze(vectors(:,:,sliceIdx,2));

% Create a grid for quiver
[xGrid, yGrid] = meshgrid(1:size(vx,2), 1:size(vx,1));
q1 = quiver(xGrid,yGrid,vy,vx,'off'); %quiver needs to swap indices (bnecause the y-axis is inverted in images))
hold on;
q1b = quiver(xGrid,yGrid,-vy,-vx,'off');
axis equal;
% xlabel('X');
% ylabel('Y');
title(['V1 as seen on ITK-snap and vis5d profile (slice ', num2str(sliceIdx), ')']);
set(gca,'Ydir','reverse')
hold off;



% disp('Please select the DTIregs.mat file (ground-truth orientations):');
% [dtiRegsFile, dtiRegsPath] = uigetfile('*.mat', 'Select DTIregs.mat');
% if isequal(dtiRegsFile,0)
%     error('No DTIregs.mat file selected.');
% end
% load(fullfile(dtiRegsPath, dtiRegsFile));

disp('Please select the SPRregs.mat file (for region definitions):');
[sprRegsFile, sprRegsPath] = uigetfile('*.mat', 'Select SPRregs.mat');
if isequal(sprRegsFile,0)
    error('No SPRregs.mat file selected.');
end
load(fullfile(sprRegsPath, sprRegsFile));

figure('Name', 'SPR Regions');
imagesc(regs(:,:,16));           % Display a slice of the regions
colormap(gray);                  % Set colormap to gray
colorbar;                        % Optional: show colorbar


figure('Name','DTI Vectors on SPR Regions');
imagesc(regs(:,:,16));           % Display a slice of the regions
colormap(gray);                  % Set colormap to gray
colorbar;                        % Optional: show colorbar


% Overlay quiver plot on SPR regions image
hold on;
q2 = quiver(xGrid, yGrid, vy, vx, 'off'); % Overlay arrows
q2.Color = 'r'; % Set arrow color to red for visibility
set(gca,'Ydir','reverse'); % Keep y-axis direction consistent
hold off;

%determine number of unique entries in regs:
nregs = numel(unique(regs)); % Assuming 'regs' is a vector with region labels
disp(['Number of unique regions: ', num2str(nregs)]);


% Create a tiled layout for all regions
figure('Name', 'DTI Regions and Distributions');
tiledlayout(nregs, 1, 'TileSpacing', 'compact');

for i = 1:nregs
    % isolate a preferred region for analysis:
    ROIreg = i-1; % Define your region of interest
    regionMask = regs == ROIreg; % Assuming 'regs' is a mask where region i is of interest

    % Reshape vectors and regionMask to 2D for masking
    sz = size(vectors);
    numVoxels = prod(sz(1:3));
    vectors_1D = reshape(vectors, numVoxels, 3);
    regionMask_1D = reshape(regionMask, numVoxels, 1);

    vectors_roi = vectors_1D(regionMask_1D, :); % Select only vectors within the region of interest

    % Convert vectors to azimuth and elevation (in degrees)
    [azimuth, elevation, ~] = cart2sph(vectors_roi(:,1), vectors_roi(:,2), vectors_roi(:,3));
    azimuth = rad2deg(azimuth);
    elevation = rad2deg(elevation);

    % Restrict azimuth to range [-90, 90]
    azimuth = mod(azimuth + 90, 180) - 90;

    % Restrict elevation to range [-90, 90]
    elevation = max(min(elevation, 90), -90);

    azBinSize = 5;
    elBinSize = 5;

    % Define bin edges
    azEdges = -90:azBinSize:90;
    elEdges = -90:elBinSize:90;

    % 2D histogram
    [N, azEdgesOut, elEdgesOut] = histcounts2(azimuth, elevation, azEdges, elEdges);

    % Prepare grid for plotting
    [AzGrid, ElGrid] = meshgrid(azEdges(1:end-1) + diff(azEdges)/2, elEdges(1:end-1) + diff(elEdges)/2);

    %normalize N
    N = N / sum(N(:)); % Normalize occurrences to sum to 1
    N = N'; % Transpose to match grid orientation
    % Plot in tiled layout
    nexttile;
    mesh(AzGrid, ElGrid, N);
    xlabel('Azimuth (deg)');
    ylabel('Elevation (deg)');
    zlabel('Normalized quantity');
    title(['Region ', num2str(ROIreg)]);
    xlim([-90 90]);
    ylim([-90 90]);
    colorbar;
    shading interp;
end

% End of the script
disp('~~~~~ fin ~~~~~');