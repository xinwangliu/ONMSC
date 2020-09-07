%% Generate Laplican matrix for each view
%% Input
% BseKer   -- base kernel
% S        -- absent index
% neighbor -- number of neighbors
%% Output
% LM - laplican matrixs
% ALM - average laplican matrix

function highorder_L = V9_laplacian_generation(BseKer)

ker_num = size(BseKer, 3);
smp_num = size(BseKer, 1);
L_record = zeros(smp_num, smp_num, ker_num);
for i = 1 : ker_num
    cur_ker = BseKer(:,:,i);
    laplacian = LaplicanGeneration(cur_ker);
    L_record(:,:,i) = laplacian;
end

highorder_L = cell(1, 2);
highorder_L{1} = L_record;

for i = 1 : ker_num
    L_record(:,:,i) = HO_laplacian(L_record(:,:,i));
end
highorder_L{2} = L_record;
end


%% Generate Laplican matrix of K according to index matrix A
function [laplacian] = LaplicanGeneration(K)
eye_matrix = 1 - eye(size(K));
K = K .* eye_matrix;
c_diag = sum(K, 2);
c_diag(c_diag == 0) = 1;
c_diag(c_diag < 10^(-10)) = 10^(-10);
c_diag = diag(sqrt(c_diag.^(-1)));
laplacian = eye(size(K)) - c_diag * K * c_diag;
end

%% Generate high-order laplacian
function L_record = HO_laplacian(L_record)
L_record = diag(diag(L_record)) - L_record;

% imagesc(L_record)
L_record = L_record * L_record;
L_record_P = L_record(L_record > 0);
L_record_org = L_record;
mean_value = mean(L_record_P);
std_value = std(L_record_P);
L_record(L_record < mean_value - std_value/2) = 0;
L_record = LaplicanGeneration(L_record);

L_record = kernel_completion(L_record, L_record_org);
% figure()
% imagesc(L_record)
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
    smp_indexes = smp_indexes(:,1);
    
    for ii = 1 : size(Small_samples, 1)
        kernel(index(ii), smp_indexes(ii)) = threshold;
    end
end
end