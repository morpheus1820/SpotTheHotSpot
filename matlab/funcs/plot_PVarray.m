function [] = plot_PVarray( Modules )
%PLOT_PVARRAY Summary of this function goes here
%   Detailed explanation goes here

for r=1:size(Modules,1)
    
    figure();
        
        subplot(1,size(Modules,2),1);
        imshow(Modules{r,1});
       
        subplot(1,size(Modules,2),2);
        imshow(Modules{r,2});
        
        subplot(1,size(Modules,2),3);
        imshow(Modules{r,3});
        
 
end

end

