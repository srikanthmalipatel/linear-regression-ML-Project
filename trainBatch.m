function[w, mu, train_erms, valid_erms] = trainBatch(M, lambda)
    load 'proj2.mat'
    % compute desing matrix for traning set
    designMat = ones(length(training_data),1);
    mu = ones(M, 46);
    for i=2:M
        random_rows = randperm(length(training_data),100);
        mu(i,:) = mean(training_data(random_rows,:));
        phi = zeros(length(training_data),1);
        for j=1:length(training_data)
           X = training_data(j,1:46) - mu(i,:);
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
           X = validation_data(j,1:46) - mu(i,:);
           c = -1/2*(X/var_matrix*transpose(X));
           phi(j) = exp(c);
        end
        designValidMat(:,i) = phi;
    end

    % compute weight vectors
    lMatrix = lambda*eye(M);
    validErr = target_validate_data - designValidMat*w;
    validE =((validErr'*validErr)/2);
    valid_erms=sqrt((2*validE)/length(validation_data));
end