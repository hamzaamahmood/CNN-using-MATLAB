function [mask]=create_mask_from_window(x)
%     """
%     Creates a mask from an input matrix x, to identify the max entry of x.
% 
%     Arguments:
%     x -- Array of shape (f, f)
% 
%     Returns:
%     mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
%     """

%     ### START CODE HERE ### (≈1 line)
    [i,j]=size(x);
    for s=1:1:i
        for k=1:1:j
         if x(s,k)== max(x(:))
             mask(s,k) = 1;
         else
             mask(s,k) = 0;
         end
        end
    end
%     ### END CODE HERE ###
end