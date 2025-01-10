Kevin Eckstein
2024-12-30: reworking code to be more user friendly; stay tuned for more details on readme, otherwise refer to below instructions...



----------------------------------------------------------------

Kevin Eckstein
2024-11-8

main code: Find_3D_lattice_fiber_dir_KNE.m
(this code iterates through volume to fit a field of orientations; at each point, it calls Image_grid_3D_FFT_KNE.m)
--> outputs "FFTO3D_... .m" file

Image_grid_3D_FFT_KNE.m can also be called as a standalone function, where you can isolate specific spots to see how the fit works

post_process_visualization loads an output file ("FFTO3D_20240207scaled.mat") and visualizes results, saves "DTI.mat" for input to NLI