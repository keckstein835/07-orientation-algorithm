function [line_dir, point_on_line] = MM_3d_grid_ransac(points,distance_threshold)
    % Check if points are provided, otherwise generate a 3D grid of points with some noise
    if nargin < 1 || isempty(points)
        % Create a grid of points in 3D space
        grid_range = linspace(-10, 10, 10);  % Grid from -10 to 10 with 10 points per axis
        [grid_x, grid_y, grid_z] = meshgrid(grid_range, grid_range, grid_range);
        points = [grid_x(:), grid_y(:), grid_z(:)];  % Create a 3D grid of points
        
        % Add uniform noise to the grid points
        noise_scale = 0.1;
        points = points + rand(size(points)) * 2 * noise_scale - noise_scale;
    end
    
    [line_dir, point_on_line, ~] = fit_line_ransac(points,distance_threshold);
    % Display the line direction as unit vector
    disp('Line direction:');
    disp(line_dir);


    % Plot the points and the fitted line
    plot_line(points, line_dir, point_on_line)
end



function [line_dir, point_on_line, inliers] = fit_line_ransac(points,distance_threshold)
    % Fits a line to a set of 3D points using RANSAC.
    % Parameters for RANSAC
    max_iterations = 1000;
    % distance_threshold = 0.1;
    num_points = size(points, 1);
    best_inliers = [];
    best_line_dir = [];
    best_point_on_line = [];

    for i = 1:max_iterations
        % Randomly select two points
        sample_indices = randperm(num_points, 2);
        sample_points = points(sample_indices, :);

        % Compute the line direction
        line_dir = sample_points(2, :) - sample_points(1, :);
        line_dir = line_dir / norm(line_dir);  % Normalize the direction

        % Compute the point on the line (mean of the sample points)
        point_on_line = mean(sample_points, 1);

        % Compute distances of all points to the line
        distances = vecnorm(cross(points - point_on_line, repmat(line_dir, num_points, 1)), 2, 2);

        % Determine inliers
        inliers = distances < distance_threshold;

        % Update the best model if the current one has more inliers
        if sum(inliers) > sum(best_inliers)
            best_inliers = inliers;
            best_line_dir = line_dir;
            best_point_on_line = point_on_line;
        end
    end

    % Output the best model
    line_dir = best_line_dir;
    point_on_line = best_point_on_line;
    % point_on_line = [0 0 0];
    inliers = best_inliers;
    % line_dir = V(:, 1);  % Direction of the line (first right singular vector)
    % point_on_line = mean(points, 1);  % Mean of the points as a point on the line
    
    % Find inliers: calculate residuals and apply a threshold
    residuals = vecnorm(points - point_on_line, 2, 2);
    threshold = 0.1;
    inliers = residuals < threshold;
end

function plot_line(points, line_dir,point_on_line)
    % Plot the 3D points
    figure;
    scatter3(points(:, 1), points(:, 2), points(:, 3), 'filled');
    hold on;
    
    % Plot the fitted line
    size(lines);
   
    % Create points along the line for plotting
    t = linspace(-10, 10, 100);
    
    % disp('Size of line_dir')
    % size(line_dir)
    % disp('Size of t')
    % size(t)
    % point_on_line = [0 0 0];
    line_points = point_on_line + t' * line_dir;
    % size(line_points)
   
    plot3(line_points(:, 1), line_points(:, 2), line_points(:, 3), 'r', 'LineWidth', 2);
    
    % Set plot labels and title
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('3D Points and Fitted Line');
    grid on;
    axis equal;
    hold off;
end