function [Z,cache] = conv_forward(A_prev, W, b, hparameters)
%     Implements the forward propagation for a convolution function
% 
%     Arguments:
%     A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
%     W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
%     b -- Biases, numpy array of shape (1, 1, 1, n_C)
%     hparameters -- python dictionary containing "stride" and "pad"
% 
%     Returns:
%     Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
%     cache -- cache of values needed for the conv_backward() function
%     

%     ### START CODE HERE ###
%     # Retrieve dimensions from A_prev's shape (≈1 line)
        [n_H_prev, n_W_prev, n_C_prev, m] = size(A_prev);
%     # Retrieve dimensions from W's shape (≈1 line)
        [f, f, n_C_prev, n_C] = size(W);

%     # Retrieve information from "hparameters" (≈2lines) stride = hparameters.stride;
        pad = hparameters.pad;
        stride = hparameters.stride;
%     # Compute the dimensions of the CONV output volume using the formula given above.
%     # Hint: use int() to floor. (≈2 lines)
        n_H = floor(n_H_prev + 2*pad - f)/stride + 1;
        n_W = floor(n_W_prev + 2*pad - f)/stride + 1;
%     # Initialize the output volume Z with zeros. (≈1 line)
        Z = zeros(n_H, n_W, n_C, m);
        A_prev_pad = zero_pad(A_prev, pad);

        for i = 1:1:m
            a_prev_pad = A_prev_pad(:, :, :, i);
            for h = 0:1:n_H-1
                vert_start = stride * h + 1;
                vert_end = vert_start + f - 1;
                for w = 0:1:n_W-1
                    horiz_start = stride * w + 1;
                    horiz_end = horiz_start + f - 1;
                    for c = 1:1:n_C
                        a_slice_prev = A_prev_pad(vert_start:vert_end,horiz_start:horiz_end, :, i);
                        weights = W(:, :, :, c);
                        biases = b(:, :, :, c);
                        s = a_slice_prev .* weights;
                        Z_sum = sum(sum(sum(s)));%sum(sum(s(:)))
                        Z(h+1, w+1, c, i) = Z_sum + biases;

                    end
                end
            end
        end
% information used in the backword Propagation
cache.A_prev = A_prev;
cache.W = W;
cache.b = b;
cache.hparameters = hparameters;

end

