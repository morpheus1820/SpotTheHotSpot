function [ vid_post, trimWidth, trimHeight ] = videoPreprocess( vidFile, trimCorners, framedropFactor )
%VIDEOPREPROCESS Summary of this function goes here
% Trim rules:
%   
%       1─────────────────>video.width
%      1 &&&&&&&&&&&&&&&&&&
%      | &&&&&&&&&&&&&&&&&&
%      | &&&&&&&&&&&&&&&&&&
%      | &&┌──────────�&&&&
%      | &&│          │&&&&
%      | &&│   trim   │&&&&
%      | &&│    Box   │&&&&
%      | &&└──────────┘&&&&
%      └ &&&&&&&&&&&&&&&&&&
%       video. height
%

k = 1;

vid = VideoReader(vidFile);

trimHeight = trimCorners.bottom - trimCorners.top + 1;
trimWidth = trimCorners.right - trimCorners.left + 1;

vid_post = uint8(zeros(trimHeight,...
                        trimWidth,...
                        ceil(vid.NumberOfFrames/framedropFactor)));

for i=1:framedropFactor:vid.NumberOfFrames

    frame = rgb2gray(read(vid,i));
    vid_post(:,:,k) = frame(...
                             trimCorners.top:trimCorners.bottom,...
                             trimCorners.left:trimCorners.right);
    k = k+1;
    
end

end

