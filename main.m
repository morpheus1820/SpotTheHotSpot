clear all;
close all;
clc;

%% Settings

% Settings Flags
PUBLISH_REPORT = 0;
PLOT = 1;

% Input file name
inputFile = 'trimmed30sec.avi';

% FrameDrop settings
frameDrop = 5; %default 5

%%% Trim Box settings %%%

% No-Watermark Settings:
trimBox.left = 16;
trimBox.top = 30;
trimBox.right = 640;
trimBox.bottom = 465;

% Full-frame Settings
%
% trimBoxLeft = 1;
% trimBoxTop = 1;
% trimBoxRight = video.width;

%%%%%%%%%%%%%%%%%%%%%%%%%%

% Geometry of modules array
modulesPerRow = 3;

% Gaussian filter properties
gaussian.size = [20, 20]; % default [20 20]
gaussian.standardDeviation = 0.5; % default 0.5

% Logical conversion threshold
bwThreshold = 0.650;

% Diamond morphological filter properties
diamond.size = 1;

%% Video DownSampling and Trimming
[vidFrames, vidWidth, vidHeight] = videoPreprocess('trimmed30sec.avi', trimBox, frameDrop);

%% Frames Segmentation
vidSeg = segmentModules( vidFrames, gaussian, bwThreshold, diamond );

%% Compute Centroids Coordinates
for i=1:size((vidSeg),3)

    % Centroid computation
    centroid = regionprops(vidSeg(:,:,i),'centroid');
    for k = 1:size(centroid,1)
         centroids.X(i,k) = centroid(k).Centroid(1); % centroid X coordinate
         centroids.Y(i,k) = centroid(k).Centroid(2); % centroid Y coordinate
    end

end

%% Process centroids distances
Ydistance = abs(centroids.Y(:,:) - ceil(size(vidSeg,1)/2));
[sortedYdistance, modulesIndexes] = sort(Ydistance,2);
threeClosestPoints = sortedYdistance(:,1:modulesPerRow);
avDistance = mean(threeClosestPoints,2);

[PKS,LOCS] = findpeaks(-avDistance, 'MINPEAKDISTANCE', ceil(10/frameDrop));

rowFrames = vidFrames(:,:,[LOCS]);
rowSegFrames = vidSeg(:,:,[LOCS]);

selectedCentroids.X = centroids.X([LOCS],:)
selectedCentroids.Y = centroids.Y([LOCS],:)
selectedCentroidsIndexes = modulesIndexes([LOCS],1:3);

%% Modules Extraction 
% Modules are segmented from the central row, from left to right

shuffledSelectedX = zeros(length(LOCS), modulesPerRow);
sortedSelectedX = zeros(length(LOCS), modulesPerRow);
Xindexes = zeros(length(LOCS), modulesPerRow);
sortedIndexes = zeros(length(LOCS), modulesPerRow);
row(length(LOCS),modulesPerRow)=struct('BoundingBox',[],'FilledImage',[]);

for rowIdx=1:length(LOCS)

   shuffledSelectedX(rowIdx,:) = selectedCentroids.X(rowIdx,[selectedCentroidsIndexes(rowIdx,:)]);

   [sortedSelectedX(rowIdx,:), Xindexes(rowIdx,:)] = sort(shuffledSelectedX(rowIdx,:));
   sortedIndexes(rowIdx,:) = selectedCentroidsIndexes(rowIdx,[Xindexes(rowIdx,:)]);
   
   segModules = regionprops(rowSegFrames(:,:,rowIdx),'FilledImage','BoundingBox'); 
   row(rowIdx,:) = segModules(sortedIndexes(rowIdx,:));
   
   for k=1:modulesPerRow
   Modules{rowIdx,k} = imcrop(rowFrames(:,:,rowIdx), row(rowIdx,k).BoundingBox);
   end

end

%% Check on the n-th row


%% Plot
if PLOT
   run('./report.m')
end

%% Generate Report

if PUBLISH_REPORT 
   publish('report.m','html'); 
end