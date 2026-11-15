
%% VEHICULAR OFFLOADING - DATA PREPROCESSING
clear;
clc;

%% 1. LOAD DATASET
data = readtable('vehicular_offloading_dataset_v2.csv');

X = data(:,1:7);
Y = categorical(data.Offloading_Decision);

fprintf('\nDATASET LOADED: %d samples\n',height(data));

%% 2. STRATIFIED TRAINING AND TEMPORARY SPLIT

rng(42,'twister');

cv1 = cvpartition(Y,'HoldOut',0.30);

trainIdx = training(cv1);
tempIdx = test(cv1);

%% 3. SPLIT TEMPORARY DATA INTO VALIDATION AND TEST

tempY = Y(tempIdx);

cv2 = cvpartition(tempY,'HoldOut',0.50);

tempPositions = find(tempIdx);

valIdx = false(height(data),1);
testIdx = false(height(data),1);

valIdx(tempPositions(training(cv2))) = true;
testIdx(tempPositions(test(cv2))) = true;

%% 4. CREATE DATA SUBSETS

TrainData = data(trainIdx,:);
ValidationData = data(valIdx,:);
TestData = data(testIdx,:);

fprintf('\nPARTITION SIZES\n');

fprintf('Training: %d\n',height(TrainData));
fprintf('Validation: %d\n',height(ValidationData));
fprintf('Testing: %d\n',height(TestData));

%% 5. VERIFY CLASS DISTRIBUTION

%% 5. VERIFY CLASS DISTRIBUTION

classNames = {'Local','V2C','V2I','V2V'};

% Convert target columns to strings for safe comparison
trainLabels = string(TrainData.Offloading_Decision);
valLabels = string(ValidationData.Offloading_Decision);
testLabels = string(TestData.Offloading_Decision);

fprintf('\nCLASS DISTRIBUTION IN EACH SUBSET\n');

for c = 1:numel(classNames)

    fprintf('\nClass: %s\n',classNames{c});

    trainCount = sum(trainLabels == string(classNames{c}));
    valCount = sum(valLabels == string(classNames{c}));
    testCount = sum(testLabels == string(classNames{c}));

    fprintf('Training: %d\n',trainCount);
    fprintf('Validation: %d\n',valCount);
    fprintf('Testing: %d\n',testCount);

end


%% 6. SAVE PARTITIONS

writetable(TrainData,'Training_Dataset.csv');
writetable(ValidationData,'Validation_Dataset.csv');
writetable(TestData,'Testing_Dataset.csv');

save('Vehicular_Preprocessed_Data.mat', ...
    'TrainData','ValidationData','TestData', ...
    'trainIdx','valIdx','testIdx');

fprintf('\n====================================\n');
fprintf(' PREPROCESSING COMPLETED SUCCESSFULLY\n');
fprintf('====================================\n');
