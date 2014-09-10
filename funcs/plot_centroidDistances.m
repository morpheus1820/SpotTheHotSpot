function [] = plot_centroidDistances( averageDistance, locations, peaks )
%PLOT_CENTROIDDISTANCES Summary of this function goes here
%   Detailed explanation goes here
figure();
hold on;

plot(averageDistance);
plot(locations,-peaks,'ro');

xlabel('frame number')
ylabel('distance [pixel]')

end

