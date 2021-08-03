function [A, cache] = pool_forward(A_prev,hparameters,mode)
        %{
        Implements the forward pass of the pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max"=1 or "average"=2)

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 

        %}

        [n_H_prev, n_W_prev, n_C_prev, m] = size(A_prev);

        f = hparameters.f;
        stride = hparameters.stride;

        n_H = floor(1 + (n_H_prev - f) / stride);
        n_W = floor(1 + (n_W_prev - f) / stride);
        n_C = n_C_prev;

        A = zeros(n_H, n_W, n_C, m);

        for i = 1:1:m

            for h = 0:1:n_H-1

                vert_start = stride * h + 1;
                vert_end = vert_start + f - 1;

                for w = 0:1:n_W-1

                    horiz_start = stride * w + 1;
                    horiz_end = horiz_start + f - 1;

                    for c = 1:1:n_C

                        a_prev_slice = A_prev(:, :, :, i);

                        if mode == 'max'
                            A(h+1, w+1, c, i) = max(max(a_prev_slice(vert_start:vert_end, horiz_start:horiz_end, c)));
                        elseif mode == 'average'
                            A(h+1, w+1, c, i) = mean(mean(a_prev_slice(vert_start:vert_end, horiz_start:horiz_end, c)));
                        end


                    end

                end
            end

        end

cache.A_prev = A_prev;
cache.hparameters = hparameters;
end

