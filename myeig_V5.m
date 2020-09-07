function [eig_vector,eig_value] = myeig_V5(M, cluster_num)

% diff  = eps;
% [eig_vector, eig_value] = eigs(M, cluster_num, 'sm', diff);


[eig_vector,eig_value] = eigs(M,cluster_num,'sm');
% if issymmetric(M)
%eig_vector = vec;
%eig_value = val;
% else
%     temp = diag(val);
%     [B,I] = sort(temp,'ascend');
%     eig_vector = vec(:,I);
%     eig_value = diag(B);
% end

%eig_vector = eig_vector(:,1:cluster_num);