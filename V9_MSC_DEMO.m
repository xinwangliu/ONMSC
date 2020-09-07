clear
clc
warning off;

addpath(genpath('./ClusteringEvaluation'));
path_data = './';

DataName = cell(1,1);
DataName{1} = 'proteinFold';



for data_num = 1
    dataName = DataName{data_num};
    %% psortPos, psortNeg; plant
    load([path_data,dataName,'_Kmatrix'],'KH','Y');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cla_num = length(unique(Y));
    ker_num = size(KH,3);
    smp_num = size(KH,1);
    neb_num = round(0.2 * smp_num);
    qnorm1 = 1;
    ker_level = 3;
    res = zeros(2,3);
    lambda_range = 2.^(-15:3:15);
    for NNRate = 1
        %% local kernel construction and Laplacian Matrix construction
        [KHN] = V9_LocalKernelCalculation(KH , NNRate, cla_num);
        % Laplacian matrix generation
        LBas = V9_laplacian_generation(KHN);
        for lambda = 1 : length(lambda_range)
            lambda_value = lambda_range(lambda);
            [H_normalized6, H_normalized1,gamma6,obj6,KH6] = V9_MSC(LBas, cla_num, qnorm1, lambda_value);
            res(lambda,:) = myNMIACC_V6(H_normalized6,Y,cla_num)
        end
    end
    max(res)
end