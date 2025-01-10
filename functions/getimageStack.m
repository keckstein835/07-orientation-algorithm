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
        % Display V_1 and label it
        % disp('Solver 1: Direct FFT (V_1)');
        % V_1
        % Plot the most intense orientation vector
            quiver3(1, 1, 1, ...
                V_1(1) * 10, V_1(2) * 10, V_1(3) * 10, ...
                'LineWidth', 3, 'Color', 'b', 'MaxHeadSize', 2);
        hold off;
    end
    
    % Check if V_2 exists and plot it
    if exist('V_2', 'var')
        hold on;
        % disp('Solver 2: Template Fitting (V_2)');
        % V_2
        % Plot the second orientation vector
        quiver3(1, 1, 1, ...
            V_2(1) * 10, V_2(2) * 10, V_2(3) * 10, ...
            'LineWidth', 2.5, 'Color', 'r', 'MaxHeadSize', 2);
        hold off;
    end
    
    % Check if V_3 exists and plot it
    if exist('V_3', 'var')
        hold on;
        disp(['V_3: ' num2str(V_3)])
        
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
    clim([0 5e11]*24/size(powerSpectrum,1));
    end