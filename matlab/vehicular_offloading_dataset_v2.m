
%% ================================================================
%  VERSION 2: SYNTHETIC VEHICULAR TASK OFFLOADING DATASET
%  QoS-Aware Multi-Criteria Decision Making
%
%  Classes:
%  0 = Local
%  1 = V2V
%  2 = V2I
%  3 = V2C
%
%  Seven input features + one target column.
%
%  All destination decisions are calculated from the seven
%  observable features using deterministic resource models.
%
%  Units:
%  Task_Size          : Giga CPU cycles
%  Deadline           : seconds
%  Data_Size          : MB
%  Latency            : milliseconds
%  Bandwidth          : Mbps
%  Processing_Power   : GHz
%  Energy_Consumption : Joules
%% ================================================================

clear;
clc;
close all;

%% 1. CONFIGURATION

rng(42, 'twister');

numSamples = 12000;

% Multi-criteria objective weights
wLatency = 0.50;
wEnergy = 0.35;
wReliability = 0.15;

% Deadline violation penalty
deadlinePenalty = 10;

% Verify weights
if abs(wLatency + wEnergy + wReliability - 1) > 1e-10
    error('Objective weights must sum to 1.');
end

%% 2. GENERATE INPUT FEATURES

% Computational workload: 0.2 to 20 Giga cycles
Task_Size = exp(log(0.2) + rand(numSamples,1) .* ...
    (log(20) - log(0.2)));

% Deadline correlated with workload, plus task-specific variation.
% This produces a range of relatively relaxed and stringent tasks.
Deadline = 0.08 + 0.10 .* sqrt(Task_Size) + ...
    1.20 .* rand(numSamples,1);

% Input data size: 0.1 to 20 MB
Data_Size = exp(log(0.1) + rand(numSamples,1) .* ...
    (log(20) - log(0.1)));

% Reference communication latency: 2 to 80 ms
Latency = 2 + 78 .* rand(numSamples,1);

% Available bandwidth: 5 to 200 Mbps
Bandwidth = 5 + 195 .* rand(numSamples,1);

% Local vehicle processing power: 1 to 5 GHz
Processing_Power = 1 + 4 .* rand(numSamples,1);

% Local energy consumption.
% Energy-per-cycle coefficient introduces vehicle-level variation.
energyCoefficient = 0.045 + 0.055 .* rand(numSamples,1);

Energy_Consumption = Task_Size .* energyCoefficient .* ...
    (Processing_Power ./ 2.5).^2;

Energy_Consumption = max(Energy_Consumption, 0.001);

%% 3. INITIALIZE OUTPUTS AND DIAGNOSTICS

Offloading_Decision = strings(numSamples,1);
Offloading_Class = zeros(numSamples,1);

Selected_Time = zeros(numSamples,1);
Selected_Energy = zeros(numSamples,1);
Selected_Cost = zeros(numSamples,1);
Deadline_Satisfied = false(numSamples,1);

% Store predicted time and energy for every destination.
All_Times = zeros(numSamples,4);
All_Energies = zeros(numSamples,4);

%% 4. DYNAMIC OFFLOADING DECISION MODEL

for i = 1:numSamples

    % Read task features
    workload = Task_Size(i);
    deadline = Deadline(i);
    dataMB = Data_Size(i);
    latencySec = Latency(i) / 1000;
    bandwidth = Bandwidth(i);
    localCPU = Processing_Power(i);
    localEnergy = Energy_Consumption(i);

    %% DESTINATION 1: LOCAL EXECUTION

    localTime = workload / localCPU;

    localTotalEnergy = localEnergy;

    %% DESTINATION 2: VEHICLE-TO-VEHICLE (V2V)

    % Peer processing capacity is modeled as a function of
    % originating vehicle processing capacity.
    v2vCPU = min(8, 1.7 * localCPU + 1);

    % Effective bandwidth available for V2V transmission
    v2vBandwidth = max(2, 0.72 * bandwidth);

    % V2V communication delay
    v2vDelay = 0.55 * latencySec + 0.005;

    % Transmission time, including 5% return-result overhead
    v2vTransmission = (dataMB * 8 / v2vBandwidth) * 1.05;

    % Computation time
    v2vExecution = workload / v2vCPU;

    v2vTime = v2vDelay + v2vTransmission + v2vExecution;

    % Total modeled energy: radio + peer computation
    v2vTotalEnergy = 0.7 * v2vTransmission + ...
        0.05 * workload;

    %% DESTINATION 3: VEHICLE-TO-INFRASTRUCTURE (V2I)

    % Roadside edge-server processing capacity
    edgeCPU = 10;

    edgeBandwidth = max(3, 0.90 * bandwidth);

    edgeDelay = 0.80 * latencySec + 0.010;

    edgeTransmission = (dataMB * 8 / edgeBandwidth) * 1.05;

    edgeExecution = workload / edgeCPU;

    edgeTime = edgeDelay + edgeTransmission + edgeExecution;

    % Modeled communication + edge-computation energy
    edgeTotalEnergy = 1.0 * edgeTransmission + ...
        0.08 * workload;

    %% DESTINATION 4: VEHICLE-TO-CLOUD (V2C)

    % Cloud computing capacity
    cloudCPU = 25;

    cloudBandwidth = max(3, 0.70 * bandwidth);

    % Higher backhaul and network delay
    cloudDelay = 1.80 * latencySec + 0.025;

    cloudTransmission = (dataMB * 8 / cloudBandwidth) * 1.05;

    cloudExecution = workload / cloudCPU;

    cloudTime = cloudDelay + cloudTransmission + cloudExecution;

    % Modeled communication + cloud-computation energy
    cloudTotalEnergy = 1.4 * cloudTransmission + ...
        0.15 * workload;

    %% 5. COLLECT CANDIDATE PERFORMANCE

    % Order: Local, V2V, V2I, V2C

    executionTimes = [localTime, v2vTime, edgeTime, cloudTime];

    energyValues = [localTotalEnergy, v2vTotalEnergy, ...
        edgeTotalEnergy, cloudTotalEnergy];

    All_Times(i,:) = executionTimes;
    All_Energies(i,:) = energyValues;

    %% 6. RELIABILITY MODEL

    % Reliability risk is modeled using observable network conditions.
    % Lower risk means a more reliable execution option.

    localRisk = 0.01;

    v2vRisk = 0.08 + ...
        0.06 * (Latency(i)/80) + ...
        0.04 * (1 - min(Bandwidth(i)/200,1));

    edgeRisk = 0.045 + 0.04 * (Latency(i)/80);

    cloudRisk = 0.04 + 0.04 * (Latency(i)/80);

    reliabilityRisk = [localRisk, v2vRisk, edgeRisk, cloudRisk];

    %% 7. NORMALIZE PERFORMANCE METRICS

    normalizedTime = executionTimes ./ max(deadline, 0.001);

    normalizedEnergy = energyValues ./ max(localEnergy, 0.001);

    %% 8. WEIGHTED QoS COST FUNCTION

    % Lower cost represents a better offloading decision.

    objectiveCost = ...
        wLatency .* normalizedTime + ...
        wEnergy .* normalizedEnergy + ...
        wReliability .* reliabilityRisk;

    %% 9. DEADLINE-CONSTRAINED OPTIMIZATION

    % Determine destinations that can meet the deadline.

    feasible = executionTimes <= deadline;

    if any(feasible)

        % If at least one destination meets the deadline,
        % reject destinations that violate it.

        objectiveCost(~feasible) = Inf;

    else

        % If no destination can meet the deadline,
        % apply an explicit deadline-violation penalty.

        violation = max(0, normalizedTime - 1);

        objectiveCost = objectiveCost + ...
            deadlinePenalty .* violation;

    end

    % All four destinations are considered available in this
    % baseline scenario. Reliability is accounted for in the cost.

    % Select the destination with minimum objective cost.

    [bestCost, bestIndex] = min(objectiveCost);

    Selected_Cost(i) = bestCost;
    Selected_Time(i) = executionTimes(bestIndex);
    Selected_Energy(i) = energyValues(bestIndex);

    Deadline_Satisfied(i) = ...
        executionTimes(bestIndex) <= deadline;

    %% 10. ASSIGN TARGET CLASS

    switch bestIndex

        case 1
            Offloading_Class(i) = 0;
            Offloading_Decision(i) = "Local";

        case 2
            Offloading_Class(i) = 1;
            Offloading_Decision(i) = "V2V";

        case 3
            Offloading_Class(i) = 2;
            Offloading_Decision(i) = "V2I";

        case 4
            Offloading_Class(i) = 3;
            Offloading_Decision(i) = "V2C";

    end

end

%% 11. CREATE FINAL DATASET

Dataset = table( ...
    Task_Size, ...
    Deadline, ...
    Data_Size, ...
    Latency, ...
    Bandwidth, ...
    Processing_Power, ...
    Energy_Consumption, ...
    Offloading_Decision);

%% 12. SAVE DATASET

writetable(Dataset, ...
    'vehicular_offloading_dataset_v2.csv');

save('vehicular_offloading_dataset_v2.mat', ...
    'Dataset', ...
    'Offloading_Class', ...
    'Selected_Time', ...
    'Selected_Energy', ...
    'Selected_Cost', ...
    'Deadline_Satisfied', ...
    'All_Times', ...
    'All_Energies');

%% 13. CLASS DISTRIBUTION

classNames = ["Local"; "V2V"; "V2I"; "V2C"];

classCounts = zeros(4,1);

for k = 0:3
    classCounts(k+1) = sum(Offloading_Class == k);
end

classPercentages = 100 .* classCounts ./ numSamples;

Class_Summary = table( ...
    classNames, ...
    classCounts, ...
    classPercentages, ...
    'VariableNames', ...
    {'Class','Count','Percentage'});

%% 14. DISPLAY RESULTS

fprintf('\n============================================\n');
fprintf(' VEHICULAR OFFLOADING DATASET - VERSION 2\n');
fprintf('============================================\n');

fprintf('Number of samples: %d\n', height(Dataset));
fprintf('Number of features: %d\n', width(Dataset)-1);
fprintf('Number of target columns: 1\n');

fprintf('\nCLASS DISTRIBUTION:\n');
disp(Class_Summary);

fprintf('Deadline satisfaction: %.2f%%\n', ...
    100 * mean(Deadline_Satisfied));

fprintf('Average selected execution time: %.4f seconds\n', ...
    mean(Selected_Time));

fprintf('Average selected energy: %.4f Joules\n', ...
    mean(Selected_Energy));

fprintf('\nFIRST 10 DATASET ROWS:\n');
disp(Dataset(1:10,:));

%% 15. VISUALIZE CLASS DISTRIBUTION

figure;

bar(classCounts);

set(gca, ...
    'XTick', 1:4, ...
    'XTickLabel', {'Local','V2V','V2I','V2C'});

xlabel('Offloading Class');
ylabel('Number of Samples');
title('Vehicular Task Offloading Class Distribution');

grid on;

%% 16. VISUALIZE DEADLINE SATISFACTION

figure;

satisfiedCount = sum(Deadline_Satisfied);
violatedCount = numSamples - satisfiedCount;

pie([satisfiedCount, violatedCount]);

legend({'Deadline Satisfied','Deadline Violated'}, ...
    'Location','best');

title('Deadline Satisfaction Distribution');

%% 17. FINAL MESSAGE

fprintf('\n============================================\n');
fprintf(' DATASET GENERATION COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

fprintf('CSV: vehicular_offloading_dataset_v2.csv\n');
fprintf('MAT: vehicular_offloading_dataset_v2.mat\n');
