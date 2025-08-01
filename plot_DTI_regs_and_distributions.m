% load DTI.mat and DTIregs.mat files; 
% DTI.mat has the automatically fitted orientations, while DTIregs.mat will be treated as ground-truth.

% disp('Please select the DTI.mat file (automatically fitted orientations):');
% [dtiFile, dtiPath] = uigetfile('*.mat', 'Select DTI.mat');
% if isequal(dtiFile,0)
%     error('No DTI.mat file selected.');
% end
% load(fullfile(dtiPath, dtiFile));

% disp('Please select the DTIregs.mat file (ground-truth orientations):');
% [dtiRegsFile, dtiRegsPath] = uigetfile('*.mat', 'Select DTIregs.mat');
% if isequal(dtiRegsFile,0)
%     error('No DTIregs.mat file selected.');
% end
% load(fullfile(dtiRegsPath, dtiRegsFile));

% disp('Please select the SPRregs.mat file (for region definitions):');
% [sprRegsFile, sprRegsPath] = uigetfile('*.mat', 'Select SPRregs.mat');
% if isequal(sprRegsFile,0)
%     error('No SPRregs.mat file selected.');
% end
% load(fullfile(sprRegsPath, sprRegsFile));

% Extract vectors from DTIregs.mat (assuming variable name is 'DTI' and is Nx3)
vectors = V1; %4D (width x length x height x 3) vector field

% isolate a preferred region for analysis:
ROIreg = 3; % Define your region of interest
regionMask = regs == ROIreg; % Assuming 'regs' is a mask where region 1 is of interest

% Reshape vectors and regionMask to 2D for masking
sz = size(vectors);
numVoxels = prod(sz(1:3));
vectors_1D = reshape(vectors, numVoxels, 3);
regionMask_1D = reshape(regionMask, numVoxels, 1);

vectors = vectors_1D(regionMask_1D, :); % Select only vectors within the region of interest

% Select only vectors within the region of interest

% Convert vectors to azimuth and elevation (in degrees)
[azimuth, elevation, ~] = cart2sph(vectors(:,1), vectors(:,2), vectors(:,3));
azimuth = rad2deg(azimuth);
elevation = rad2deg(elevation);

% Define bin edges
azEdges = linspace(-180, 180, 37); % 10-degree bins
elEdges = linspace(-90, 90, 19);   % 10-degree bins

% 2D histogram
[N, azCenters, elCenters] = histcounts2(azimuth, elevation, azEdges, elEdges);

% Prepare grid for plotting
[AzGrid, ElGrid] = meshgrid(azCenters(1:end-1) + diff(azCenters)/2, elCenters(1:end-1) + diff(elCenters)/2);

% Plot 3D surface
figure;
surf(AzGrid, ElGrid, N');
xlabel('Azimuth (deg)');
ylabel('Elevation (deg)');
zlabel('Occurrences');
title('3D Histogram of Vector Orientations');
colorbar;
shading interp;