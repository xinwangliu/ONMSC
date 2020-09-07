function cF = mycombFun(Y,gamma)

m = size(Y,3);
cF = zeros(size(Y,1),size(Y,2));
for p =1:m
    cF = cF + Y(:,:,p)*gamma(p);
end