function [] = plot_PVarray( Modules, SegmentedModules )
%PLOT_PVARRAY Summary of this function goes here
%   Detailed explanation goes here

for r=1:size(Modules,1)
    
    figure();
        
        subplot(1,size(Modules,2),1);
        imshow(Modules{r,1});
        hold on;
        
         % modifiche ste 
         st = regionprops(SegmentedModules{r,1}, 'convexhull','boundingbox')
         bb=st.BoundingBox
         s=size(bb) 
         ratio=bb(3)/bb(4);
         ch =st.ConvexHull
         
         for n=1:s(1)       
             hold on;
     
             if(ratio(n)>0.1) 
                plot(ch(n,1), ch(n,2),'r','LineWidth',4);
            else    
                plot(ch(n,1), ch(n,2),'g','LineWidth',2);
         end
         
        hold off;
        % fine modifiche
        
        subplot(1,size(Modules,2),2);
        imshow(Modules{r,2});
  % modifiche ste 
         st = regionprops(SegmentedModules{r,2}, 'convexhull','boundingbox')
         bb=st.BoundingBox
         s=size(bb) 
         ratio=bb(3)/bb(4);
         ch =st.ConvexHull
         
         for n=1:s(1)       
             hold on;
     
             if(ratio(n)>0.1) 
                plot(ch(n,1), ch(n,2),'r','LineWidth',4);
            else    
                plot(ch(n,1), ch(n,2),'g','LineWidth',2);
         end
         
        hold off;
        % fine modifiche
        
        subplot(1,size(Modules,2),3);
        imshow(Modules{r,3});
         % modifiche ste 
         st = regionprops(SegmentedModules{r,3}, 'convexhull','boundingbox')
         bb=st.BoundingBox
         s=size(bb) 
         ratio=bb(3)/bb(4);
         ch =st.ConvexHull
         
         for n=1:s(1)       
             hold on;
     
             if(ratio(n)>0.1) 
                plot(ch(n,1), ch(n,2),'r','LineWidth',4);
            else    
                plot(ch(n,1), ch(n,2),'g','LineWidth',2);
         end
         
        hold off;
        % fine modifiche   
 
end


% for r=1:size(SegmentedModules,1)
%     
%     figure();
%     
%         subplot(1,size(SegmentedModules,2),1);
%         imshow(SegmentedModules{r,1});
%         hold on;
%         [B,L,N] = bwboundaries(SegmentedModules{r,1});
%         for k=1:length(B)
%             boundary = B{k};
%             st = regionprops(boundary, 'eccentricity' );
%             if([st.Eccentricity]>0.5)
%                 plot(boundary(:,2), boundary(:,1),'r','LineWidth',4);
%             else
%                 plot(boundary(:,2), boundary(:,1),'w','LineWidth',2);
%   
%             end
%                
%             
%         end
%         hold off;
%         
%         subplot(1,size(SegmentedModules,2),2);
%         imshow(SegmentedModules{r,2});
%         
%         subplot(1,size(SegmentedModules,2),3);
%         imshow(SegmentedModules{r,3});
%  
% end

end

