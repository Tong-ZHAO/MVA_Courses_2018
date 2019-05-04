function [ v ] = gradn_bilateral( u )
%This function calculates the bilateral gradient of an image
    [m, n] = size(u);
    v = sqrt((u(3:m, 2:n-1) - u(1:m-2, 2:n-1)).^2 + (u(2:m-1, 3:n) - u(2:m-1, 1:n-2)).^2);

end

