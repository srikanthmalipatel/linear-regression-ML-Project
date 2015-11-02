UBitName = 'smalipat';
personNumber = '50169097';

load 'proj2.mat';

%M1 = 10;
%lambda1 = 2;

%M2 = 4;
%lambda2 = 1;
max_m = 10;

clearvars w1 mu1 Sigma1;
clearvars w2 mu2 Sigma2;
clearvars w w01 dw1 eta1;

validPer1 =10000;

for i=2:max_m
    [lambda, Sigma, w, mu, trainPer, validPer] = trainBatch(i);
    fprintf('M=%d\t%f\t\t%f\n', i, trainPer, validPer);
    if validPer1 > validPer
       % fprintf('updating mu1 Sigma1');
       M1 = i;
       lambda1 = lambda;
       Sigma1 = Sigma;
       w1 = w;
       mu1 = mu;
       trainPer1 = trainPer;
       validPer1 = validPer;
    end
end

validPer2 = 10000;



for i=2:max_m
    [lambda, w, mu, Sigma, trainPer, validPer, trainInd, validInd] = trainSynBatch(i);
    fprintf('M=%d\t%f\t\t%f\n', i, trainPer, validPer);
    if validPer2 > validPer
       % fprintf('updating mu1 Sigma1');
       M2 = i;
       lambda2 = lambda;
       Sigma2 = Sigma;
       w2 = w;
       mu2 = mu;
       trainPer2 = trainPer;
       validPer2 = validPer;
    end
end

w01 = w1*10;
[dw1, w_real, eta1] = trainGd(M1,lambda1, w01, mu1);
w02 = w2*10;
[dw2, w_syn, eta2] = trainSynGd(M2,lambda2, w02, mu2);

mu1 = mu1';
mu2 = mu2';

Sigma1 = zeros(46,46,M1);
for i=1:M1
    Sigma1(:,:,i) = var_matrix;
end

save 'proj2.mat';