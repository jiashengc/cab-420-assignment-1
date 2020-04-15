% %% %% %% %% %% %%
%   CAB420 Machine Learning
%   Assignment 1
%   Date:  15 April 2018
% 
%   This file was written by Jia Sheng Chong(n9901990) 
%   and Ian Daniel(n5372828).
% %% %% %% %% %% %%

%%% Question 1
clear;
delete(findall(0,'Type','figure'));

%% (1a) Plot the training data in a scatter plot
% Load the data
mTrain = load('data/mcycleTrain.txt');

linearFig = figure(1);
ytr = mTrain(:,1); xtr = mTrain(:,2);
plot(xtr,ytr,'bo');
hold on;

%% (1b) Create a linear predictor
Xtr = polyx(xtr, 1);
linearLearner1 = linearReg(Xtr, ytr);
yhat    = predict(linearLearner1, Xtr);

xline = [0:.01:2]'; % transpose : make a column vector , like training x
yline = predict (linearLearner1 , polyx ( xline ,1) ); % assuming quadratic features

plot(xline, yline, 'r');
hold on;

%% (1c) Create another plot with data and a 5th polynomial
newXtr = polyx(xtr, 5);
linearLearner5 = linearReg(newXtr, ytr);
newYline = predict(linearLearner5, polyx(xline, 5));
plot(xline, newYline, 'g.');

%% (1d) Calculate the mean squared error
err1 = mse(linearLearner1, Xtr, ytr);
err2 = mse(linearLearner5, newXtr, ytr);

%% (1e) Calculate the MSE for each model on the test data
mTest = load('data/mcycleTest.txt');

ytest = mTest(:,1); xtest = mTest(:,2);
Xtest1 = polyx(xtest, 1);
Xtest5 = polyx(xtest, 5);
err3 = mse(linearLearner1, Xtest1, ytest);
err4 = mse(linearLearner5, Xtest5, ytest);

%% (1f) Label your plots
title('Linear Regression showing trained polynomials and training data');
legend('Training Data points', '1st order polynomial', '5th order polynomial');
xlabel('X'); ylabel('Y');


%%% Question 2
%% (2b) Plot the predicted fuction for several values

% Create the kNN learners
knnRegLearner1  = knnRegress(1,  xtr, ytr);
knnRegLearner2  = knnRegress(2,  xtr, ytr);
knnRegLearner3  = knnRegress(3,  xtr, ytr);
knnRegLearner5  = knnRegress(5,  xtr, ytr);
knnRegLearner10 = knnRegress(10, xtr, ytr);
knnRegLearner50 = knnRegress(50, xtr, ytr);

xline = [0:.01:2]';
% Run the predictions
yKnnRegLine1  = predict (knnRegLearner1  , xline);
yKnnRegLine2  = predict (knnRegLearner2  , xline);
yKnnRegLine3  = predict (knnRegLearner3  , xline);
yKnnRegLine5  = predict (knnRegLearner5  , xline);
yKnnRegLine10 = predict (knnRegLearner10 , xline);
yKnnRegLine50 = predict (knnRegLearner50 , xline);

% Display the results on one figure using subplots
knnFig = figure(2);
suptitle('kNN with various values of K')
set(knnFig, 'Position', [150, 0, 1500, 1000]);

subplotKnn1 = subplot(3, 2, 1);
plot(subplotKnn1, xtr,ytr,'bo');
hold on;
plot(subplotKnn1, xline, yKnnRegLine1,  'r');
title(subplotKnn1, 'K=1');

subplotKnn2 = subplot(3, 2, 2);
plot(subplotKnn2, xtr,ytr,'bo');
hold on;
plot(subplotKnn2, xline, yKnnRegLine2,  'r');
title(subplotKnn2, 'K=2)');

subplotKnn3 = subplot(3, 2, 3);
plot(subplotKnn3, xtr,ytr,'bo');
hold on;
plot(subplotKnn3, xline, yKnnRegLine3,  'r');
title(subplotKnn3, 'K=3)');

subplotKnn5 = subplot(3, 2, 4);
plot(subplotKnn5, xtr,ytr,'bo');
hold on;
plot(subplotKnn5, xline, yKnnRegLine5,  'r');
title(subplotKnn5, 'K=5');

subplotKnn10 = subplot(3, 2, 5);
plot(subplotKnn10, xtr,ytr,'bo');
hold on;
plot(subplotKnn10, xline, yKnnRegLine10,  'r');
title(subplotKnn10, 'K=10');

subplotKnn50 = subplot(3, 2, 6);
plot(subplotKnn50, xtr,ytr,'bo');
hold on;
plot(subplotKnn50, xline, yKnnRegLine50,  'r');
title(subplotKnn50, 'K=50');


%%% Question 3: Hold-out and Cross-validation
%% (3a) Compute the MSE of the test data
ytest20 = mTest(1:20,1); xtest20 = mTest(1:20,2);

kErrors20 = [];
for k = 1:100,
    knnRegLearner = knnRegress(k, xtr, ytr);
    kErrors20 = [kErrors20, mse(knnRegLearner, xtest20, ytest20)];
end

mseFig1 = figure(3);
loglog(1:100, kErrors20, '-s');
hold on
grid on;

%% (3b) Repeat but using all the training data
ytest = mTest(:,1); xtest = mTest(:,2);

kErrors = [];
for k = 1:100,
    knnRegLearner = knnRegress(k, xtr, ytr);
    kErrors = [kErrors, mse(knnRegLearner, xtest, ytest)];
end

loglog(1:100, kErrors, '-s');

%% (3c) Estimate the curve using 4-fold cross-validation
% Runing test using test data points 1-20, 21-40, 41-60 and 61-80
% Using training data points for all remaining vaules on each of the 
% four groups

fourFoldErrors = zeros(4, 100);
for k = 1:100,
    r = 1;
    for xval = 1:4,
        yTest = mTrain(r:r+19, 1); xTest = mTrain(r:r+19, 2);
        yTrain = setdiff(mTrain(:, 1), yTest); xTrain = setdiff(mTrain(:, 2), xTest); 
        
        learner = knnRegress(k, xTrain, yTrain);
        fourFoldErrors(xval, k) = mse(learner, xTest, yTest);
		
        r = r + 20;
    end
end

mseAvg  = mean(fourFoldErrors);
loglog(1:100, mseAvg, '-s');

grid on;
legend('20 Test Data points', 'All Data points', '4 Fold Hold Out');
title('Graphed MSE with K=1 to k=100');
xlabel('K'); ylabel('MSE');


%%% Question 4: Nearest Neigbour Classifer
%% (4a) Display the 3 classes of data
iris = load('data/iris.txt');
pi = randperm(size(iris,1)); % randomize the order of data
yIris = iris(pi, 5); xIris = iris(pi, 1:2); %split data between training and test

irisClasses = unique(yIris);

% plot to figure
irisFig = figure(5);
hold on;
for i = 1:length(yIris)
    switch(yIris(i))
        case 0
            plot(xIris(i,1), xIris(i,2), 'r*');
        case 1
            plot(xIris(i,1), xIris(i,2), 'b*');
        case 2
            plot(xIris(i,1), xIris(i,2), 'g*');
    end
end

legend('Setosa', 'Versicolour ', 'Virginica');
title('Plot of sepal length & width for 3 Iris flowers');
xlabel('Length(cm)'); ylabel('width(cm)');

%% (4b) Plot just k=1 kNN classifier
knnClasLearner1 = knnClassify(1, xIris, yIris);
class2DPlot(knnClasLearner1, xIris, yIris);

title('1NN with decision regions and training data');
legend('Setosa', 'Versicolour ', 'Virginica');
xlabel('Length(cm)'); ylabel('width(cm)');

%% (4c) Plot k=3, 10 and 30 kNN classifier
knnClasLearner3  = knnClassify(3,  xIris, yIris);
knnClasLearner10 = knnClassify(10, xIris, yIris);
knnClasLearner30 = knnClassify(30, xIris, yIris);

class2DPlot(knnClasLearner3,  xIris, yIris);
title('3NN with decision regions and training data');
legend('Setosa', 'Versicolour ', 'Virginica');
xlabel('Length(cm)'); ylabel('width(cm)');

class2DPlot(knnClasLearner10, xIris, yIris);
title('10NN with decision regions and training data');
legend('Setosa', 'Versicolour ', 'Virginica');
xlabel('Length(cm)'); ylabel('width(cm)');

class2DPlot(knnClasLearner30, xIris, yIris);
title('30NN with decision regions and training data');
legend('Setosa', 'Versicolour ', 'Virginica');
xlabel('Length(cm)'); ylabel('width(cm)');

%% (4d) Use 80/20 split and check the error of the data


%Build the train and test arrays 
aIris = iris(:, 5) == 0;
bIris = iris(:, 5) == 1; 
cIris = iris(:, 5) == 2; 

aIris = iris(aIris, :);
bIris = iris(bIris, :);
cIris = iris(cIris, :);

aTotalCount = size(aIris, 1);
bTotalCount = size(bIris, 1);
cTotalCount = size(cIris, 1);

aPi = randperm(aTotalCount);
bPi = randperm(bTotalCount);
cPi = randperm(cTotalCount);

aIris = aIris(aPi, 1:2);
bIris = bIris(bPi, 1:2);
cIris = cIris(cPi, 1:2);

aTrainCount = floor(aTotalCount*0.8);
bTrainCount = floor(bTotalCount*0.8);
cTrainCount = floor(cTotalCount*0.8);

aIrisTrain = aIris(1:aTrainCount,:);
bIrisTrain = bIris(1:bTrainCount,:);
cIrisTrain = cIris(1:cTrainCount,:);

aIrisTest = aIris(aTrainCount+1:end,:);
bIrisTest = bIris(bTrainCount+1:end,:);
cIrisTest = cIris(cTrainCount+1:end,:);

aYTrain(1:size(aIrisTrain,1)) = 0; aYTest(1:size(aIrisTest,1)) = 0;
bYTrain(1:size(bIrisTrain,1)) = 1; bYTest(1:size(bIrisTest,1)) = 1;
cYTrain(1:size(cIrisTrain,1)) = 2; cYTest(1:size(cIrisTest,1)) = 2;

%put it all together

xIrisTrain = [aIrisTrain; bIrisTrain; cIrisTrain];
yIrisTrain = [aYTrain'; bYTrain'; cYTrain'];

xIrisValid = [aIrisTest; bIrisTest; cIrisTest];
yIrisValid = [aYTest'; bYTest'; cYTest'];

kValues = [1 2 5 10 50 100 200];
kClasErr = [];

for i = 1:length(kValues)
    learner = knnClassify(kValues(i), xIrisTrain, yIrisTrain);
    yhat = predict(learner, xIrisValid);
    predicted = find(~(yhat - yIrisValid));
    kClasErr = [kClasErr, (1 - (length(predicted)/length(yhat)))*100];
end

figure(6);
plot(kValues, kClasErr, '-o');
title('Percent error for k=[1 2 5 10 50 100 200]');
ylabel('% Wrong');
xlabel('K');


%%% Question 5: Perceptrons and Logistic Regression
%% (5a) Plot classes 0 and 1 then classes 1 and 2
iris = load('data/iris.txt');
xIrisData = iris(:,1:2); yIrisData = iris(:,end);
[xIrisData, yIrisData] = shuffleData(xIrisData, yIrisData);
xIrisData = rescale(xIrisData);
XA = xIrisData(yIrisData<2,:); YA = yIrisData(yIrisData<2);
XB = xIrisData(yIrisData>0,:); YB = yIrisData(yIrisData>0);

figure(7); hold on;
for i = 1:length(YA)
    switch(YA(i))
        case 0
            plot(XA(i,1), XA(i,2), 'r*');
        case 1
            plot(XA(i,1), XA(i,2), 'b*');
    end
end

title('Iris data set - only showing Classes 0 and 1 (rescaled)');
legend('Class 0', 'Class 1');

figure(8); hold on;
for i = 1:length(YB)
    switch(YB(i))
        case 2
            plot(XB(i,1), XB(i,2), 'g*');
        case 1
            plot(XB(i,1), XB(i,2), 'b*');
    end
end

title('Iris data set - only showing Classes 1 and 2 (rescaled)');
legend('Class 1', 'Class 2');
%% (5b) Display two classes (0 and 1) with a decision boundary
logLearner = logisticClassify2();
logLearner = setClasses(logLearner, unique(YA));
wts = [.5, 1, -.25];
logLearner = setWeights(logLearner, wts);

plot2DLinear(logLearner, XA, YA);
title('Classes 0 and 1 with a line decision boundary (rescaled)');

%% (5c) Complete the predict function and calculate the error
f = @(x1, x2) wts(1) + wts(2)*x1 + wts(3)*x2;
figure;
plotClassify2D(logLearner, XA, YA);

yte = predict(logLearner, XA);
logLearnerError = err(logLearner, XA, YA);

title('Classes 0 and 1 with a coloured decision boundary (rescaled)');
legend('Postive Class','Negative Class');

%% (5f)
logLearnerA = logisticClassify2(); logLearnerB = logisticClassify2();
logLearnerA = setClasses(logLearnerA, unique(YA));
logLearnerB = setClasses(logLearnerB, unique(YB));

wts = [.5, 1, -.25];
logLearnerA = setWeights(logLearnerA, wts);
logLearnerB = setWeights(logLearnerB, wts);

train(logLearnerA, XA, YA);
train(logLearnerB, XB, YB);

logLearnerAError = err(logLearnerA, XA, YA);
logLearnerBError = err(logLearnerB, XB, YB);

figure;
plotClassify2D(logLearnerA, XA, YA);
title('Classes 0 and 1 with a coloured decision boundary (rescaled)');
legend('Postive Class','Negative Class');

figure;
plotClassify2D(logLearnerB, XB, YB);
title('Classes 1 and 2 with a coloured decision boundary (rescaled)');
legend('Postive Class','Negative Class');
