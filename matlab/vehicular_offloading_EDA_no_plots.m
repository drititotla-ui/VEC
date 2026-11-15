
%% VEHICULAR OFFLOADING EDA - NO PLOTS VERSION
clear;
clc;

%% 1. LOAD DATASET
data = readtable('vehicular_offloading_dataset_v2.csv');

features = {'Task_Size','Deadline','Data_Size', ...
    'Latency','Bandwidth','Processing_Power', ...
    'Energy_Consumption'};

X = data{:,features};
Y = categorical(data.Offloading_Decision);

fprintf('\n========== DATASET INFORMATION ==========\n');
fprintf('Samples: %d\n',height(data));
fprintf('Features: %d\n',length(features));
fprintf('Classes: %d\n',numel(categories(Y)));

%% 2. MISSING VALUE ANALYSIS
fprintf('\n========== MISSING VALUE ANALYSIS ==========\n');

Missing = sum(ismissing(data));

disp(array2table(Missing, ...
    'VariableNames',data.Properties.VariableNames));

%% 3. DESCRIPTIVE STATISTICS
fprintf('\n========== DESCRIPTIVE STATISTICS ==========\n');

Statistics = table( ...
    features', ...
    min(X)', ...
    max(X)', ...
    mean(X)', ...
    median(X)', ...
    std(X)', ...
    'VariableNames', ...
    {'Feature','Minimum','Maximum','Mean','Median','StdDev'});

disp(Statistics);

writetable(Statistics,'EDA_Statistics.csv');

%% 4. CLASS DISTRIBUTION
fprintf('\n========== CLASS DISTRIBUTION ==========\n');

classNames = categories(Y);
classCounts = countcats(Y);

ClassDistribution = table( ...
    classNames, ...
    classCounts, ...
    100*classCounts/height(data), ...
    'VariableNames', ...
    {'Class','Count','Percentage'});

disp(ClassDistribution);

writetable(ClassDistribution,'EDA_Class_Distribution.csv');

%% 5. CORRELATION MATRIX
fprintf('\n========== CORRELATION MATRIX ==========\n');

CorrelationMatrix = corr(X,'Rows','complete');

CorrelationTable = array2table( ...
    CorrelationMatrix, ...
    'VariableNames',features, ...
    'RowNames',features);

disp(CorrelationTable);

writetable(CorrelationTable, ...
    'EDA_Correlation_Matrix.csv', ...
    'WriteRowNames',true);

%% 6. CLASS-WISE FEATURE MEANS
fprintf('\n========== CLASS-WISE FEATURE MEANS ==========\n');

numClasses = numel(classNames);

ClassMeans = zeros(numClasses,length(features));

for i = 1:numClasses

    idx = (Y == classNames{i});

    ClassMeans(i,:) = mean(X(idx,:),1);

end

ClassMeansTable = array2table( ...
    ClassMeans, ...
    'VariableNames',features, ...
    'RowNames',classNames);

disp(ClassMeansTable);

writetable(ClassMeansTable, ...
    'EDA_Class_Wise_Means.csv', ...
    'WriteRowNames',true);

%% 7. OUTLIER ANALYSIS
fprintf('\n========== OUTLIER ANALYSIS ==========\n');

OutlierCounts = zeros(length(features),1);

for i = 1:length(features)

    OutlierCounts(i) = sum(isoutlier(X(:,i)));

end

OutlierSummary = table( ...
    features', ...
    OutlierCounts, ...
    100*OutlierCounts/height(data), ...
    'VariableNames', ...
    {'Feature','OutlierCount','OutlierPercentage'});

disp(OutlierSummary);

writetable(OutlierSummary, ...
    'EDA_Outlier_Summary.csv');

%% 8. SAVE ALL RESULTS
save('Vehicular_EDA_Results.mat', ...
    'Statistics', ...
    'ClassDistribution', ...
    'CorrelationMatrix', ...
    'ClassMeansTable', ...
    'OutlierSummary');

fprintf('\n====================================\n');
fprintf(' EDA COMPLETED SUCCESSFULLY!\n');
fprintf('====================================\n');
