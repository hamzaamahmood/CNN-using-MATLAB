function [X_PAD] = zero_pad(X, pad)
 
%     Argument:
%     X -- Matlab array of shape (m, n_H, n_W, n_C) representing a batch of m images
%     pad -- integer, amount of padding around each image on vertical and horizontal dimensions
% 
%     Returns:
%     X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    

%     START CODE HERE ### (≈ 1 line)
    [n_H, n_W, n_C, m] = size(X);
    n_H_new =  n_H + 2 * pad;
    n_W_new =  n_W + 2 * pad;

    X_PAD = zeros(n_H_new, n_W_new, n_C, m);

    for h = (1+pad):1:n_H+pad
        for w = (1+pad):1:n_W+pad
            for c = 1:1:n_C
                X_PAD(h, w, c, :) = X(h-pad, w-pad, c, :);
            end
        end
    end
end
% END CODE HERE