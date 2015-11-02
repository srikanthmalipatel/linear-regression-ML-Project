function [dw2, w_new, eta2]= trainSynGd(M,lambda, w_new, mu_old) 
    load 'synthetic.mat'
    % divide synthetic data
    syn_data = transpose(x);
    training_data = syn_data(1:1600,:);
    target_train_data = t(1:1600,1);

    % compute variance
    var_data = var(training_data);
    var_matrix = diag(var_data)*0.115+eye(10);
    
    % compute desing matrix for traning set
    designMat = ones(length(training_data),1);
    for i=2:M
        phi = zeros(length(training_data),1);
        for j=1:length(training_data)
           X = training_data(j,1:10) - mu_old(i,:);
           c = -1/2*(X/var_matrix*transpose(X));
           phi(j) = exp(c);
        end
        designMat(:,i) = phi;
    end
    
    eta2 = zeros(1);
    breakloop = zeros(M,1);
    for i=1:M
       breakloop(i) = 0.001; 
    end
    
    n=0.1;
    dw2 = zeros(M, 1);
    b = false;
    first = true;
    while 1
        for j=1:length(training_data)
            %fprintf('iteration: %d', j);
            deltaED=-1*(target_train_data(j)-(designMat(j,:)*w_new))*designMat(j,:)';
            deltaW = -1*n*(deltaED + lambda*w_new);
            if first == false
                if deltaW <= breakloop
                    %fprintf('w Converged \n');
                    b = true;
                    break;
                end
            end
            dw2 = [dw2 deltaW];
            w_new = w_new + deltaW;
            eta2 = [eta2 n];
        end
        if first == true
            first = false;
        end
        if b == true
           break; 
        end
    end
    eta2(:,1) = [];
    dw2(:,1) = [];
end
