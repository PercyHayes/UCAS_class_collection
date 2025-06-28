% 加载数据
load('data_x.mat');



% 设置聚类个数
k = 5;

% 运行K-means聚类算法
[idx, C] = kmeans_clustering(X, k);

% 计算聚类精度和误差
mu = [1, -1; 5.5, -4.5; 1, 4; 6, 4.5; 9, 0.0];
correspondence = zeros(k, 1);
for i = 1:k
    distances = sqrt(sum((C - mu(i, :)).^2, 2));
    [~, min_idx] = min(distances);
    correspondence(i) = min_idx;
end

% 计算误差
error = sqrt(sum((C(correspondence, :) - mu).^2, 2));
% 计算聚类精度
correct = 0;
for i = 1:k
    correct = correct + sum(idx == correspondence(i));
end
accuracy = correct / size(X, 1);


% 显示结果
disp('聚类中心:');
disp(C);
disp('误差:');
disp(error);
disp('聚类精度:');
disp(accuracy);

% 绘制聚类结果
figure;
colors = ['c', 'b', 'k', 'g', 'm'];
hold on;
for i = 1:k
    scatter(X(idx == i, 1), X(idx == i, 2), 7, colors(i), 'filled');
end
scatter(C(:, 1), C(:, 2), 100, 'r', 'filled', 'd');
hold off;

% K-means 聚类算法
function [idx, C] = kmeans_clustering(X, k)
    % 随机选择初始聚类中心
    C = X(randperm(size(X, 1), k), :);
    prev_C = C;
    max_iters = 100;
    for iter = 1:max_iters
        % 计算每个点到聚类中心的距离
        D = pdist2(X, C);
        % 分配每个点到最近的聚类中心
        [~, idx] = min(D, [], 2);
        % 更新聚类中心
        for i = 1:k
            C(i, :) = mean(X(idx == i, :), 1);
        end
        % 检查聚类中心是否收敛
        if isequal(C, prev_C)
            break;
        end
        prev_C = C;
    end
end