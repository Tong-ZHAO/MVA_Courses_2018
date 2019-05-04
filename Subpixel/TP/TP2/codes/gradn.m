function [ v ] = gradn( u )
% This function calculates the gradient of an image
    [m, n] = size(u);
    v = sqrt((u(2:m, 1:n-1) - u(1:m-1, 1:n-1)).^2 + (u(1:m-1, 2:n) - u(1:m-1, 1:n-1)).^2);
end

