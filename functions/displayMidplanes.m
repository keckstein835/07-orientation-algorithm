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