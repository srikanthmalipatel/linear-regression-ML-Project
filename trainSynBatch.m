function[w, mu, Sigma2, train_erms, valid_erms, trainInd2, validInd2] = trainSynBatch(M, lambda)
    load 'synthetic.mat'
    % divide synthetic data
    syn_data = transpose(x);
    training_data = syn_data(1:1600,:);
    validation_data = syn_data(1600+1:2000,:);
    
    target_train_data = t(1:1600,1);
    target_validate_data = t(1600+1:2000,1);
    
    % compute variance
    var_data = var(training_data);
    var_matrix = diag(var_data)*0.115+eye(10);
    
    % compute desing matrix for traning set
    designMat = ones(length(training_data),1);
    mu = ones(M, 10);
    for i=2:M
        random_rows = randperm(length(training_data),20);
        mu(i,:) = mean(training_data(random_rows,:));
        phi = zeros(length(training_data),1);
        for j=1:length(training_data)
           X = training_data(j,1:10) - mu(i,:);
           c = -1/2*(X/var_matrix*transpose(X));
           phi(j) = exp(c);
        end
        designMat(:,i) = phi;
    end

    % compute weight vectors
    lMatrix = lambda*eye(M);
    w = inv(lMatrix + transpose(designMat)*designMat)*transpose(designMat)*target_train_data;
    trainErr = target_train_data - designMat*w;
    trainE =((trainErr'*trainErr)/2);
    train_erms=sqrt((2*trainE)/length(training_data));
    
    % compute design matrix for validation set
    designValidMat = ones(length(validation_data),1);
    for i=2:M
        phi = zeros(length(validation_data),1);
        for j=1:length(validation_data)
           X = validation_data(j,1:10) - mu(i,:);
           c = -1/2*(X/var_matrix*transpose(X));
           phi(j) = exp(c);
        end
        designValidMat(:,i) = phi;
    end

    % compute weight vectors
    validErr = target_validate_data - designValidMat*w;
    validE =((validErr'*validErr)/2);
    valid_erms=sqrt((2*validE)/length(validation_data));
    
    % construc sigma matrix
    Sigma2 = zeros(10,10,M);
    for i=1:M
        Sigma2(:,:,i) = var_matrix;
    end

    trainInd2 = 1:1600;
    trainInd2 = transpose(trainInd2);
    validInd2 = 1601:2000;
    validInd2 = transpose(validInd2);
end