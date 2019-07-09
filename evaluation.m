function [ acc, sens, spec ] = evaluation( ytest, ypred )
% Classification Evaluation function 
%   Given two arrays, that one contains the actual labels of the test set
%   and the other contains a prediction, this function returns the
%   Accuracy, the Sensitivity and the Specificity of the prediction.

% The class labels are: '2' for benign and '4' for malignant.

TP = 0;
FP = 0;
TN = 0;
FN = 0;
% For each label
for i = 1:length(ytest)
    % If we predicted yes(that it is malignant),
    if(ypred(i)==4)
        % and it really is.
        if(ytest(i)==4)
            TP = TP + 1;
        % If its not malignant
        elseif(ytest(i)==2)
            FP = FP + 1;
        end
    % If we predicted no,
    elseif(ypred(i)==2)
        % but it really is malignant.
        if(ytest(i)==4)
            FN = FN + 1;
        % and it really is not malignant.
        elseif(ytest(i)==2)
            TN = TN + 1;
        end
    end
end

% Compute the rates
acc = (TP + TN)/(TP + TN + FP + FN);
sens = TP/(TP + FN);
spec = TN/(TN + FP);

end

