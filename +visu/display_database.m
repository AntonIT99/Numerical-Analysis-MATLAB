function [hh, h, display_array] = displayDatabase(X, example_width, example_height, titleToDisplay)

    %-- DISPLAYDATA Display 2D data in a nice grid
    %--   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
    %--   stored in X in a nice grid. It returns the figure handle h and the 
    %--   displayed array if requested.

    %-- Gray Image
    hh = figure;
    colormap(gray);

    %-- Compute rows, cols
    [m,n] = size(X);
    if (m>(example_width*example_height*3))
        m = example_width*example_width*3;
        X = X(:,1:m);
    end

    %-- Compute number of items to display
    display_rows = floor(sqrt(n));
    display_cols = ceil(n / display_rows);

    %-- Between images padding
    pad = 1;

    %-- Setup blank display
    display_array = - ones(pad + display_rows * (example_height + pad), ...
                           pad + display_cols * (example_width + pad), 3);

    %-- Copy each example into a patch on the display array
    curr_ex = 1;
    for j = 1:display_rows
        for i = 1:display_cols
            if (curr_ex > n)
                break; 
            end
            %-- Copy the patch

            %-- Get the max value of the patch
            display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                          pad + (i - 1) * (example_width + pad) + (1:example_width),:) = ...
                            reshape(X(:,curr_ex), example_height, example_width, 3) ;
            curr_ex = curr_ex + 1;
        end
        if (curr_ex > n)
            break; 
        end
    end

    %-- Display Image
    h = imagesc(display_array);    

    %-- Do not show axis
    axis image off;
    title(titleToDisplay)
    drawnow;

end
