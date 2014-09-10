function [ ] = plot_nthRow( frames, segFrames, rowStruct, modulesCell, rowNumber )
%PLOT_NTHROW Summary of this function goes here
%   Detailed explanation goes here

figure()

subplot(3,2,1)
imshow(segFrames(:,:,rowNumber))

subplot(3,2,2)
imshow(frames(:,:,rowNumber))

subplot(3,3,4)
imshow(rowStruct(rowNumber,1).FilledImage)

subplot(3,3,5)
imshow(rowStruct(rowNumber,2).FilledImage)

subplot(3,3,6)
imshow(rowStruct(rowNumber,3).FilledImage)

subplot(3,3,7)
imshow(modulesCell{rowNumber,1})

subplot(3,3,8)
imshow(modulesCell{rowNumber,2})

subplot(3,3,9)
imshow(modulesCell{rowNumber,3})

end

