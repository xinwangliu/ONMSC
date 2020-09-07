%% S: num0*m, each column indicates the indices of absent samples
function [H_normalized1, H_normalized2, gamma, obj, LC] = V9_MSC(L, cluster_count, qnorm1, lambda_value)

ker_num = size(L{1},3);

%% Correlation matrix M
M_matrix_N1 = Cor_Calculation(L{1});
M_matrix_N2 = Cor_Calculation(L{2});

M_matrix = L_Similarity(L{1}) + L_Similarity(L{2});

matrix_N = (M_matrix_N1 + M_matrix_N2) * lambda_value;

%% initialization
% Initialize kernel weights
gamma = ones(ker_num,1) / ker_num;
% Combining each Laplacian matrix with average weight
LC1 = mycombFun(L{1}, gamma.^qnorm1);
LC2 = mycombFun(L{2}, gamma.^qnorm1);
LC = LC1+LC2;
% Initialize H
H = update_H(LC,  cluster_count);
% Initialize W
W = update_W(LC, H, cluster_count);

% alpha = obj2 / obj1;
%% optimization
obj = zeros(1, 30);
iter = 0;
flag = 1;
while flag
    
    iter = iter + 1;
    fprintf(1, 'running iteration of the proposed algorithm %d...\n', iter);
    % update lambda
    BB = 2 * W'* (LC1 + LC2 - 0.5*(H*H')) * W - 4*eye(size(W, 2));
    LAM = update_LAM(diag(BB));
    %     cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma)
    % update gamma
    gamma = update_G(L, W, LAM, M_matrix, matrix_N);
    LC1 = mycombFun(L{1}, gamma.^qnorm1);
    LC2 = mycombFun(L{2}, gamma.^qnorm1);
    %     cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma)
    % update W
    W = update_W(LC1+LC2, H, cluster_count);
    %     cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma)
    % update H
    L_E = eye(size(LC)) - W*LAM*W';
    H = update_H(L_E, cluster_count);
    %     cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma)
    obj(iter) = cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma);
    
    if iter>3 && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 ||iter>30)
        flag =0;
    end
    
end

obj = obj(1:iter);
% [H_normalized,~] = myspectral_clustering(LC, cluster_count);
% [H_normalized]= mykernelkmeans_V6(L_C);
H_normalized1 = H;
[H_normalized2,~] = myspectral_clustering(LC, cluster_count);
end


function H = update_H(A, cluster_count)
h_matrix = A;
[H,~] = myspectral_clustering(h_matrix, cluster_count);
end

function W = update_W(B, H, cluster_count)
% w_matrix = 2*B - 2*eye(size(B)) - H*H';
w_matrix = 2*B - H*H';
[W, ~] = myspectral_clustering(w_matrix, cluster_count);
end

function gamma = update_G(L, W, LAM, M_matrix, C_matrix)
ker_num = size(L{1}, 3);
smp_num = size(L{1}, 1);

L_E = eye(smp_num) - W*LAM*W';

% Calculate vector p
vector_p = zeros(ker_num, 1);
for i_level = 1 : 2
    L_current = L{i_level};
    for i = 1 : ker_num
        vector_p(i) = vector_p(i) + trace(L_current(:,:,i)*L_E');
    end
end

M = 2*M_matrix + 2* C_matrix;
f = -2*vector_p';
A = [];
b = [];
Aeq = ones(1, ker_num);
beq = 1;
opts = optimoptions('quadprog' , 'Algorithm' , 'interior-point-convex' , 'display' , 'off');
gamma = quadprog(M,f,A,b,Aeq,beq, [], [], [], opts);
% LB = [];
% UB = [];
% X0 = [];
% gamma = quadprog(M,f,A,b,Aeq,beq,LB,UB,X0);
end

function [obj, obj1, obj2]= cal_spec_obj_C(H, W, LAM, LC1, LC2, matrix_N, gamma)
matrix1 = eye(size(LC1)) - W*LAM*W';
H2 = H*H';
obj1 = trace(matrix1 * H2);
matrix2 = matrix1 - LC1;
matrix3 = matrix1 - LC2;
obj2 = trace(matrix2 * matrix2') + trace(matrix3 * matrix3');
obj3 = gamma'*matrix_N*gamma;
obj = obj1+obj2+obj3;
end


function [H_normalized,obj]= myspectral_clustering(K, cluster_count)

K = (K+K')/2;
% [H,value] = eigs(K, cluster_count+1, 'LM', opt);
[H,~] = myeig_V5(K, cluster_count);
% H = H(:,1:cluster_count);
% [H,value] = eigs(K, cluster_count, 'SM', opt);
obj = trace(H' * K * H);
% H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);
H_normalized = H;
end

function LAM = update_LAM(bb)
cla_num = size(bb, 1);
M = 4 * eye(cla_num);
ap = bb;
A = [];
b = [];
Aeq = [];
beq = [];
LB = zeros(size(bb));
UP = ones(size(bb));
opts = optimoptions('quadprog' , 'Algorithm' , 'interior-point-convex' , 'display' , 'off');
LAM = quadprog(M,ap,A,b,Aeq,beq, LB, UP, [], opts);
LAM = diag(LAM);
end


function L_Correlation = Cor_Calculation(Laplacian)
ker_num = size(Laplacian, 3);
ker_norm = zeros(ker_num, 1);
M_matrix = zeros(ker_num);
for i = 1 : ker_num
    ker_norm(i) = norm(Laplacian(:,:,i), 'fro');
end

% tic
% for ii = 1 : ker_num
%     for jj = ii : ker_num
%         %% using GPU
%         a1 = Laplacian(:,:,ii);
%         a2 = Laplacian(:,:,jj);
%         a_result = trace(a1 * a2);
%         M_matrix(ii, jj) = a_result / (ker_norm(ii) * ker_norm(jj));
%         M_matrix(jj, ii) = M_matrix(ii, jj);
%     end
% end
% toc

% tic
smp_num = size(Laplacian, 1);
ker_num = size(Laplacian, 3);
mm1 = zeros(ker_num, ker_num);
for ii = 1:ker_num
    for jj = ii : ker_num
        mm1(ii, jj) = ker_norm(ii) * ker_norm(jj);
        mm1(jj, ii) = mm1(ii, jj);
    end
end
b = reshape(Laplacian, smp_num*smp_num,ker_num);
Laplacian2 = permute(Laplacian, [2,1,3]);
b2 = reshape(Laplacian2, smp_num*smp_num, ker_num);
mm = b2' * b;
mm = mm./mm1;
% toc

L_Correlation = mm;
end


function L_Correlation = L_Similarity(Laplacian)
smp_num = size(Laplacian, 1);
ker_num = size(Laplacian, 3);
b = reshape(Laplacian, smp_num*smp_num,ker_num);
Laplacian2 = permute(Laplacian, [2,1,3]);
b2 = reshape(Laplacian2, smp_num*smp_num, ker_num);
mm = b2' * b;
L_Correlation = mm;
end