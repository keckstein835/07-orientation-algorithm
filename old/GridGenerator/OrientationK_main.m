clear all 
close all
% Define the grid parameters
A_length_pixels = 2; %in PIXELS
pixel_size = 2; % mm

cell_x_Wavelength = 2;  % Spacing in X-direction
cell_y_Wavelength = 1;   % Spacing in Y-direction
cell_z_Wavelength = 1;   % Spacing in Z-direction
grid_x_spacing = cell_x_Wavelength / (A_length_pixels*pixel_size);
grid_y_spacing = cell_y_Wavelength / A_length_pixels*pixel_size;
grid_z_spacing = cell_z_Wavelength / A_length_pixels*pixel_size;
% grid_x_spacing = cell_x_Wavelength / (A_length_pixels*pixel_size);
% grid_y_spacing = cell_y_Wavelength / A_length_pixels;
% grid_z_spacing = cell_z_Wavelength / A_length_pixels;

grid_n_pointsXYZ = [9 5 5]; 
xLength = grid_n_pointsXYZ(1)* grid_x_spacing-1;  % Length of domain in X-direction
yLength = grid_n_pointsXYZ(2)* grid_y_spacing-1;  % Length of domain in Y-direction
zLength = grid_n_pointsXYZ(3)* grid_z_spacing-1;  % Length of domain in Z-direction





% Create an instance of Grid3D
grid = Grid3D(xLength, yLength, zLength, grid_x_spacing, grid_y_spacing, grid_z_spacing);

% Display original coordinates (including the origin)
grid.displayCoordinates();

% Define a 3x3 rotation matrix (rotation about Z-axis by 45 degrees)
theta = pi/4;  % 45 degrees
rotationMatrix = [
    cos(theta), -sin(theta), 0;
    sin(theta), cos(theta),  0;
    0,          0,           1];

% Apply the transformation to the grid
grid = grid.applyTransformation(rotationMatrix);

% Display transformed coordinates (including the origin)
grid.displayCoordinates();

% Plot the grid without specifying a plot range
% grid.plotGrid();

% Plot the grid with specified axis limits

plotRange = [-A_length_pixels, A_length_pixels; -A_length_pixels , A_length_pixels ; -A_length_pixels , A_length_pixels ] /2 ;  % Range for X, Y, Z axes
grid.plotGrid(plotRange);

title('units in mm')

% display a code finished message... 
disp('Code finished running!')
