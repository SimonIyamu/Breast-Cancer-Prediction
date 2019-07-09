clear; clc; close all;

% ------------- Preprocessing -------------

filename = 'breast-cancer-wisconsin.data';
fp = fopen(filename,'r');

M = [];
% For every line in the file
line = fgetl(fp);
while ischar(line)
    % Convert line string to 11x1 array
    values = str2double(split(line, ','));
    
    % Remove the id feature
    values(1)=[];
    
    % Append it to the matrix
    M = [M ; values'];

    % Read the next line
    line = fgetl(fp);
end
fclose(fp);

% In the following operations we will ignore the 10th column, as it
% represents the class of each instance.

% Replace NaNs with the mean 
for col = 1 : 9
    m = nanmean(M(:,col));
    M(isnan(M(:,col)),col) = m;
end

% Normalization
% Since all features are in the domain [1,10] we can divide by 10.
M(:,1:9) = M(:,1:9)/10;


% ------------- K-Fold Cross Validation -------------

% k-fold indices
indices = crossvalind('Kfold',M(:,10),10);

% The following 3x10 arrays will store the accuracy, sensitivity and
% specificity for each iteration of the k-fold cross validation
knnScores1 = [];
knnScores2 = [];
nbScores1 = [];
nbScores2 = [];
svmScores1 = [];
svmScores2 = [];
dtScores1 = [];
dtScores2 = [];

for i = 1:10
    % k-fold masks
    testMask = (indices == i);
    trainMask = ~testMask;
    
    xtrain = M(trainMask,1:9);
    ytrain = M(trainMask,10);
    
    xtest = M(testMask,1:9);
    ytest = M(testMask,10);
    
    % ----- Classification -----
    
    % Multilayer Feedforward Perceptron
    %net = feedforwardnet(10);
    %net = train(net,xtrain,ytrain);
    %ypred = predict(net, xtest);
    
    % KNN
    %   With NumNeighbors = 1
    knnModel = fitcknn(xtrain,ytrain);
    ypred = predict(knnModel,xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    % Append the evaluation rates to the scores matrix
    knnScores1 = [knnScores1 ; acc sens spec];
    
    %   With NumNeighbors = 5
    knnModel = fitcknn(xtrain,ytrain,'NumNeighbors',5);
    ypred = predict(knnModel,xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    % Append the evaluation rates to the scores matrix
    knnScores2 = [knnScores2 ; acc sens spec];
    
    % Bayes Classifier
    %   With gaussian kernel
    nbModel = fitcnb(xtrain,ytrain,'DistributionNames','kernel','Kernel','normal');
    ypred = predict(nbModel,xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    nbScores1 = [nbScores1 ; acc sens spec];
    
    %   With box kernel(uniform).
    nbModel = fitcnb(xtrain,ytrain,'DistributionNames','kernel','Kernel','box');
    ypred = predict(nbModel,xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    nbScores2 = [nbScores2 ; acc sens spec];
    
    % SVM
    %   With linear kernel function
    svmModel = fitcsvm(xtrain,ytrain);
    ypred = predict(svmModel, xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    svmScores1 = [svmScores1 ; acc sens spec];
    
    %   With Radial Basis Function kernel
    svmModel = fitcsvm(xtrain,ytrain,'KernelFunction','rbf');
    ypred = predict(svmModel, xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    svmScores2 = [svmScores2 ; acc sens spec];
    
    % Decision Tree
    %   With gdi split criterion(Gini's diversity index)
    dtModel = fitctree(xtrain,ytrain);
    ypred = predict(dtModel, xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    dtScores1 = [dtScores1 ; acc sens spec];
    
    %   With cross entropy as split criterion for maximum deviance
    %   reduction
    dtModel = fitctree(xtrain,ytrain,'SplitCriterion','deviance');
    ypred = predict(dtModel, xtest);
    [acc, sens, spec] = evaluation(ytest, ypred);
    dtScores2 = [dtScores2 ; acc sens spec];
end

% Compute the mean scores of the 10 iterations and display them
disp("KNN:")
disp("    With NumNeighbors = 1:")
disp("        Accuracy: " + mean(knnScores1(:,1)) + "   Sensitivity: " + mean(knnScores1(:,2)) + "   Specificity: " + mean(knnScores1(:,3)))
disp("    With NumNeighbors = 5:")
disp("        Accuracy: " + mean(knnScores2(:,1)) + "   Sensitivity: " + mean(knnScores2(:,2)) + "   Specificity: " + mean(knnScores2(:,3)))
disp(" ")

disp("Bayes Classifier:")
disp("    With gaussian kernel:")
disp("        Accuracy: " + mean(nbScores1(:,1)) + "   Sensitivity: " + mean(nbScores1(:,2)) + "   Specificity: " + mean(nbScores1(:,3)))
disp("    With box kernel(uniform):")
disp("        Accuracy: " + mean(nbScores2(:,1)) + "   Sensitivity: " + mean(nbScores2(:,2)) + "   Specificity: " + mean(nbScores2(:,3)))
disp(" ")

disp("SVM:")
disp("    With linear kernel function:")
disp("        Accuracy: " + mean(svmScores1(:,1)) + "   Sensitivity: " + mean(svmScores1(:,2)) + "   Specificity: " + mean(svmScores1(:,3)))
disp("    With Radial Basis Function kernel:")
disp("        Accuracy: " + mean(svmScores2(:,1)) + "   Sensitivity: " + mean(svmScores2(:,2)) + "   Specificity: " + mean(svmScores2(:,3)))
disp(" ")

disp("Decision Tree:")
disp("    With gdi split criterion(Gini's diversity index):")
disp("        Accuracy: " + mean(dtScores1(:,1)) + "   Sensitivity: " + mean(dtScores1(:,2)) + "   Specificity: " + mean(dtScores1(:,3)))
disp("     With cross entropy as split criterion:")
disp("        Accuracy: " + mean(dtScores2(:,1)) + "   Sensitivity: " + mean(dtScores2(:,2)) + "   Specificity: " + mean(dtScores2(:,3)))
