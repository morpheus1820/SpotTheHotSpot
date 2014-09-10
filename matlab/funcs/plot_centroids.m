function [ ] = plot_centroids( centroidsYcoordinates, locations, videoHeight )
%PLOT_CENTROIDS Summary of this function goes here
%   Detailed explanation goes here

figure();
hold on;

Lidx = 1;
LOCS_ = [locations;0]; %prevent LOCS overflow during the plot
m='r*';

hy = graph2d.constantline(ceil(videoHeight/2), 'Color',[.9 .9 .9], 'LineWidth',70); % x-axis
hy2 = graph2d.constantline(ceil(videoHeight/2), 'Color',[0 0 0], 'LineWidth',1); % x-axis

% Centroids Clean Plot (not showing 0 points, highlighting selected frames)
for r = 1:1:size(centroidsYcoordinates,1)
    
    if (r == LOCS_(Lidx))
        m='ro';
        Lidx = Lidx + 1;
    else
        m = '*';
    end
    
    for c=1:1:size(centroidsYcoordinates,2)
        if (centroidsYcoordinates(r,c) ~= 0)
            plot(r,centroidsYcoordinates(r,c),m); %TODO make it independent of vidSeg
        end
    end
    
end

xlabel('frame number')
ylabel('centroid Y coordinate [pixel]')

end

