% 加载数据
load('X.mat');

k = 2;
sigma = 0.03:0.02:2
%[0.03,0.05,0.08,0.1,0.2,0.5,0.8,1,2];
acc = zeros(length(sigma),1);
sample = 1;
for si = sigma
    
    labels = spectral_clustering(X, k,si);
    class1 = (labels == 1);
    class2 = (labels == 2);
    if sum(class1(1:100))>sum(class2(1:100))
        tmp = sum(class1(1:100))+sum(class2(101:200));
    else
        tmp = sum(class2(1:100))+sum(class1(101:200));
    end
    acc(sample) = tmp/200;
    sample = sample + 1;
end

plot(sigma,acc*100);
xlabel("\sigma")
ylabel("准确率/%")
% % 调用谱聚类函数
% labels = spectral_clustering(X, k,0.03);
% 
% % 显示聚类结果
% scatter(X(:,1), X(:,2), 20, labels, 'filled');
% title('Spectral Clustering Results');




function labels = spectral_clustering(X, k,sigma)
    % X: 数据点矩阵
    % k: 聚类的数量

    % 计算相似度矩阵 W
    %sigma = 0.5; % 高斯核的参数，可以根据需要调整
    n = size(X, 1);
    W = zeros(n, n);
    for i = 1:n
        for j = 1:n
            W(i, j) = exp(-norm(X(i, :) - X(j, :))^2 / (2 * sigma^2));
        end
    end
    W = (W + transpose(W))/2;

    % 计算度矩阵 D
    D = diag(sum(W, 2));

    % 计算拉普拉斯矩阵 L
    L = D - W;

    % 计算归一化拉普拉斯矩阵 L_norm
    D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
    L_norm = D_inv_sqrt * L * D_inv_sqrt;%+randn(n,n)*1e-15;

    %计算 L_norm 的前 k 个最小特征值对应的特征向量
    [V, ~] = eigs(L_norm, k, 'smallestabs');

    % 将特征向量标准化
    U = V ./ sqrt(sum(V.^2, 2));

    % 对标准化后的特征向量进行 k-means 聚类
    labels = kmeans(U, k);
end