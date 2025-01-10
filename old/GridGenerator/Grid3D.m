classdef Grid3D
    properties
        xLength      % Length of the domain along X-axis
        yLength      % Length of the domain along Y-axis
        zLength      % Length of the domain along Z-axis
        xSpacing     % Grid spacing for X-axis
        ySpacing     % Grid spacing for Y-axis
        zSpacing     % Grid spacing for Z-axis
        coordinates  % Matrix to hold 3D grid coordinates
    end
    
    methods
        % Constructor to initialize the grid with domain and spacing
        function obj = Grid3D(xLength, yLength, zLength, xSpacing, ySpacing, zSpacing)
            % Initialize class properties
            obj.xLength = xLength;
            obj.yLength = yLength;
            obj.zLength = zLength;
            obj.xSpacing = xSpacing;
            obj.ySpacing = ySpacing;
            obj.zSpacing = zSpacing;
            
            % Generate the grid
            obj = obj.generateGrid();
        end
        
        % Function to generate the 3D grid points and include the origin
        function obj = generateGrid(obj)
            % Calculate the range for each axis based on domain length and grid spacing
            xRange = -obj.xLength/2:obj.xSpacing:obj.xLength/2;
            yRange = -obj.yLength/2:obj.ySpacing:obj.yLength/2;
            zRange = -obj.zLength/2:obj.zSpacing:obj.zLength/2;
            
            % Create the 3D grid
            [X, Y, Z] = meshgrid(xRange, yRange, zRange);
            
            % Reshape into vectors and store in coordinates property
            X = X(:);
            Y = Y(:);
            Z = Z(:);
            obj.coordinates = [X, Y, Z];
            
            % Add the origin point [0, 0, 0] to the grid
            obj.coordinates = [obj.coordinates; 0, 0, 0];
        end
        
        % Function to apply a 3D transformation matrix (3x3) to the grid points
        function obj = applyTransformation(obj, transformationMatrix)
            % Ensure the transformation matrix is 3x3
            if size(transformationMatrix, 1) ~= 3 || size(transformationMatrix, 2) ~= 3
                error('Transformation matrix must be 3x3.');
            end
            
            % Apply the 3x3 transformation matrix to the coordinates
            obj.coordinates = (transformationMatrix * obj.coordinates')';
        end
        
        % Function to display the coordinates
        function displayCoordinates(obj)
            disp('3D Grid Coordinates (including origin):');
            disp(obj.coordinates);
        end
        
        % Function to plot the 3D points with an optional input for plot range
        function plotGrid(obj, plotRange)
            % Extract X, Y, Z coordinates
            X = obj.coordinates(:, 1);
            Y = obj.coordinates(:, 2);
            Z = obj.coordinates(:, 3);
            
            % Create the 3D scatter plot
            figure;
            scatter3(X, Y, Z, 'filled');
            xlabel('X');
            ylabel('Y');
            zlabel('Z');
            title('3D Grid Points');
            grid on;
            axis equal;
            
            % If plotRange is specified, set the axes limits
            if nargin == 2
                xlim(plotRange(1, :));
                ylim(plotRange(2, :));
                zlim(plotRange(3, :));
            end
        end
    end
end
