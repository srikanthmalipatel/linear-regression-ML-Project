function [dw1, w_new, eta1]= trainGd(M,lambda, w_new, mu_old) 
    load 'proj2.mat'
    % compute desing matrix for traning set
    designMat = ones(length(training_data),1);
    for i=2:M
        phi = zeros(length(training_data),1);
        for j=1:length(training_data)
           X = training_data(j,1:46) - mu_old(i,:);
           c = -1/2*(X/var_matrix*transpose(X));
           phi(j) = exp(c);
        end
        designMat(:,i) = phi;
    end
    
    eta1 = zeros(1);
    breakloop = zeros(M,1);
    for i=1:M
       breakloop(i) = 0.001; 
    end
    
    n=0.25;
    dw1 = zeros(M, 1);
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
            dw1 = [dw1 deltaW];
            w_new = w_new + deltaW;
            eta1 = [eta1 n];
        end
        if first == true
            first = false;
        end
        if b == true
           break; 
        end
    end
    eta1(:,1) = [];
    dw1(:,1) = [];
end
