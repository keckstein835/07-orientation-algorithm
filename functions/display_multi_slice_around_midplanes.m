function display_multi_slice_around_midplanes(image3D, color_axis_limit,Fx,Fy,Fz )
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