UBitName = 'smalipat';
personNumber = '50169097';

load 'proj2.mat';

M1 = 10;
lambda1 = 2;

[w1, mu1, trainPer1, validPer1] = trainBatch(M1, lambda1);

mu1 = mu1';
Sigma1 = zeros(46,46,M1);
for i=1:M1
    Sigma1(:,:,i) = var_matrix;
end

save 'proj2.mat';