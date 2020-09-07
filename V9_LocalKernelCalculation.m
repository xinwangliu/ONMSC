function [KHL] = V9_LocalKernelCalculation(KH , NNRate, cla_num)

ker_num = size(KH, 3);
smp_num = size(KH, 1);
gamma0 = ones(ker_num,1)/ker_num;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate average kernel
KH_temp = KH;
% remove too large components
KH_temp = remove_large(KH_temp);
% base kernel normalization
KH_temp = knorm(KH_temp);
% base kernel centerlization
KH_temp = kcenter(KH_temp);
% divide base kernels with std
KH_temp = divide_std(KH_temp);
% Generate average kernel
AVGKer = mycombFun(KH_temp,gamma0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Generate affinity matrix
Org_AVGKer = AVGKer;
NeiNum = round( NNRate * smp_num / cla_num );
[aff_matrix, indx] = genarateNeighborhood(AVGKer , NeiNum);
aff_matrix = (aff_matrix + aff_matrix');
% Cut index that are not good
% aff_matrix = cut_link(aff_matrix, indx, NeiNum, cla_num);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% filter the extreme values in each single kernel
KHL = zeros(size(KH));
for i_ker = 1 : size(KH, 3)
    Current_kernel = KH(:,:,i_ker) .* aff_matrix;
    Current_kernel(Current_kernel < 0) = 0;
    mean_value = mean(Current_kernel(:));
    std_value = std(Current_kernel(:));
    up_threshold = mean_value + std_value*3;
    down_threshold = max(0, mean_value - std_value*3);
    Current_kernel(Current_kernel > up_threshold) = up_threshold;
    Current_kernel(Current_kernel < down_threshold) = down_threshold;
    Current_kernel = kernel_completion(Current_kernel, Org_AVGKer);
    KHL(:,:,i_ker) = Current_kernel;
end

end

%% Generate neighbor index
function [aff_matrix, indx] = genarateNeighborhood(KC,tau)
aff_matrix = zeros(size(KC, 1));
smp_num = size(KC, 1);
KC0 = KC - 10^18*eye(smp_num);
% KC0 = KC;
[~,indx] = sort(KC0,'descend');
indx_0 = indx(1:tau,:);

smp_index = 1 : smp_num;
smp_index = repmat(smp_index, tau, 1);

indx_0 = indx_0(:);
smp_index = smp_index(:);
real_index = (smp_index-1) * smp_num + indx_0;
aff_matrix(real_index) = 1;
end

function aff_matrix = cut_link(aff_matrix, indx, NeiNum, cla_num)
similarity_sum = sum(aff_matrix, 2);
smp_num = size(aff_matrix, 1);
smp_threshold = round(smp_num/cla_num*1.2);
smp_threshold = min(smp_threshold, 4*NeiNum);
index = find(similarity_sum > smp_threshold);

index2 = indx(:, index);
index2 = index2(1:round(NeiNum/2), :);
aff_matrix(index, :) = 0;
aff_matrix(:, index) = 0;

index = repmat(index', round(NeiNum/2), 1);
index = index(:);
index2 = index2(:);

final_index = (index-1)*smp_num + index2;
temp_zero = zeros(smp_num);
temp_zero(final_index) = 1;
temp_zero = (temp_zero + temp_zero')>0;
aff_matrix = double(aff_matrix);
aff_matrix = aff_matrix + temp_zero;
end


function kernel = kernel_completion(kernel, org_kernel)

Avg_index = kernel ~= 0;
Ker_sum = sum(Avg_index, 2);
index = find(Ker_sum == 0);
threshold = mean(kernel(:));
if ~isempty(index)
    org_kernel = org_kernel - diag(diag(org_kernel));
    Small_samples = org_kernel(index, :);
    [~, smp_indexes] = sort(Small_samples,2,'descend');
    smp_indexes = smp_indexes(:,1:2);
    
    for ii = 1 : size(Small_samples, 1)
        kernel(index(ii), smp_indexes(ii,:)) = threshold;
    end
end
end

function K = remove_large(K)
if size(K,3)>1
    for i=1:size(K,3)
        current_kernel = K(:,:,i);
        mean_value = mean(current_kernel(:));
        std_value = std(current_kernel(:));
        threshold_up = mean_value + std_value*4;
        threshold_dn = mean_value - std_value*4;
        current_kernel(current_kernel > threshold_up) = threshold_up;
        current_kernel(current_kernel < threshold_dn) = threshold_dn;
        K(:,:,i) = current_kernel;
    end
else
    mean_value = mean(K(:));
    std_value = std(K(:));
    threshold_up = mean_value + std_value*4;
    threshold_dn = mean_value - std_value*4;
    K(K > threshold_up) = threshold_up;
    K(K < threshold_dn) = threshold_dn;
end
end

function K = divide_std(K)
if size(K,3)>1
    for i=1:size(K,3)
        current_kernel = K(:,:,i);
        std_value = std(current_kernel(:));
        current_kernel = current_kernel / std_value;
        K(:,:,i) = current_kernel;
    end
else
    std_value = std(K(:));
    K = K / std_value;
end
end
