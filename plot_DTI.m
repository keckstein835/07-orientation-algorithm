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
title(['V1 as oriented like T1 viewed on ITK-snap and vis5d profile (slice ', num2str(sliceIdx), ')']);
set(gca,'Ydir','reverse')
hold off;


% End of the script
disp('~~~~~ fin ~~~~~');