function [ vid_post ] = segmentModules( vid, gaussianProperties, bwThreshold, diamondProperties )
%SEGMODULES Summary of this function goes here
%   Detailed explanation goes here

vid_post = false(size(vid,1),size(vid,2),size(vid,3));

for i=1:size((vid),3)

    tmpImage = vid(:,:,i);
    
    % Apply Guassian filter
    h = fspecial('gaussian', gaussianProperties.size, gaussianProperties.standardDeviation);
    Ifilt = imfilter(tmpImage,h);
    
    % Black and White conversion
    Ifilt = im2bw(Ifilt, bwThreshold);
    
    % Fill holes in bw image
    Ifilt = imfill(Ifilt, 'holes');
    
    % Diamond erosion for borders smoothing
    seD = strel('diamond',diamondProperties.size);
    Ifilt = imerode(Ifilt,seD);
    
    % Remove objects with less then 500 pixels inside
    vid_post(:,:,i) = bwareaopen(Ifilt, 500);

end

