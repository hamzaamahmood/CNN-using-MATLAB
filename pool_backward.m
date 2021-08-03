function [dA_prev] = pool_backward(dA, cache, mode)

%{
Implements the backward pass of the pooling layer
    
Arguments:
dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

Returns:
dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    
%}

A_prev = cache.A_prev;
hparameters = cache.hparameters;

f = hparameters.f;
stride = hparameters.stride;

[n_H_prev, n_W_prev, n_C_prev, m] = size(A_prev);
[n_H, n_W, n_C, m] = size(dA);

dA_prev = zeros(n_H_prev, n_W_prev, n_C_prev, m);

for sample = 1:m
    
    a_prev = A_prev(:, :, :, sample);
    
    for i = 0:n_H-1
        
        vert_start = stride * i + 1;
        vert_end = vert_start + f - 1;
       
        for j = 0:n_W-1
            
            horiz_start = stride * j + 1;
            horiz_end = horiz_start + f - 1;
            
            for c = 1:n_C
                
                a_prev_slice = A_prev(:, :, :, sample);
                
                if mode == 'max'
                    
                    a_prev_slice = a_prev(vert_start:vert_end, horiz_start:horiz_end, c);
                    mask = create_mask_from_window(a_prev_slice);
                    correct_entry = dA(i+1,j+1,c, sample);
                    dA_prev(vert_start: vert_end, horiz_start: horiz_end, c, sample) = ...
                        dA_prev(vert_start: vert_end, horiz_start: horiz_end, c, sample) + ...
                        mask .* correct_entry;
                        
                
                elseif mode == 'average'
                    
                    da = dA(i+1, j+1, c, sample);
                    shape = [f, f];
                    dA_prev(vert_start: vert_end, horiz_start: horiz_end, c, sample) = ...
                        dA_prev(vert_start: vert_end, horiz_start: horiz_end, c, sample) + ...
                        distributeValue(da,shape(1),shape(2));
                        
                end
                
                    
            end
                        
        end
    end
    
end

end

