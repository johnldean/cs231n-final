% CS 231n project
clear all; close all;

n = 50;
N = 100;
data = [-ones(n,N/2) ones(n,N/2)] + rand(n,N);
labels = [ones(1,N/2) 2*ones(1,N/2)];


% Init
l1_size = 10;
A1 = randn(n, l1_size);
b1 = randn(l1_size,1);
A2 = randn(l1_size, 2);
b2 = randn(2,1);

%% Forward
z1 = data' * A1 + b1';
v1 = pos(z1);
z2 = v1 * A2 + b2';
v2 = pos(z2);

%% Backwards
dout = ones(size(v2));
dout(v2 <= 0) = 0;
db2     = sum(dout, 1)';
dA2     = (dout' * v1)';
dout    = (dout * A2');
db1     = sum(dout, 1)';
dA1     = (dout' * data')';

%%
dg = [reshape(dA1, [n*l1_size,1]); db1; reshape(dA2, [l1_size*2,1]); db2];
wk = [reshape(A1, [n*l1_size,1]); b1; reshape(A2, [l1_size*2,1]); b2];


%% cvx
ind = zeros(N,2);
for i=1:N
    ind(i,labels(i)) = 1;
end
ind = (ind==1);

%%
cvx_begin
% cvx_solver sedumi
    variable w(length(dg))
    variable yhat(size(v2))
    minimize( sum(-yhat(ind) + log_sum_exp(yhat')') + sum(w.*w) )
    
    subject to
        yhat == v2 + dg'*(w - wk)    
cvx_end
