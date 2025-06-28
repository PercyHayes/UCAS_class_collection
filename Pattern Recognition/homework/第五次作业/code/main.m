%%%%%%%% ORL数据集%%%%% 
clear all;
load ORLData_25;
      X = ORLData';
X = double(X);
[n, dim] = size(X);
  
labels = X(:, dim);          %获取各样本的类别标签
labels = floor(double(labels));
c = max(labels);              % c = 40 
    
X(:, dim) = [];              % 获取样本数据
clear ORLData;

%%%%%%%%%%%%%%%Vehicle数据集%%%%%%%%%%
% clear all;
% load vehicle;
% out = UCI_entropy_data.train_data;
% 
% X = out'; 
% X = double(X);
% [n, dim] = size(X);  
% labels = X(:, dim);   
% labels = floor(double(labels));  % 获取各样本的类别标签
% c = max(labels);             % c = 4
% X(:, dim) = [];             % 获取样本数据
% clear UCI_entropy_data;
% clear out;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 程序%%%%%%%%%%%%%%
max_k = 25;%最大降维设置
step = 2;
method = 'pca_knn';%'lda_knn'; %'pca_knn'

%数据集切分
rng(1); %random seed
[train_ids,test_ids] = crossvalind('HoldOut',labels,0.2);
%训练
train_data = X(train_ids,:);
train_label = labels(train_ids);
%测试
test_data = X(test_ids,:);
test_label = labels(test_ids);


%降维设置
dimension = 1:step:max_k;%[1,2,3,5,10,15,20,25];%[1,2,4,8,10,12,16,32];

acc = zeros(length(dimension),1);
i=1;
%%% basline%%%
mdl = fitcknn(train_data,train_label,'NumNeighbors', 1);
pred = predict(mdl,test_data);
basline = sum(pred == test_label)/numel(test_label);

for d = dimension
    
    if strcmp(method,'pca_knn')%method == 'pca_knn'
        %PCA方法
        pca_acc = pca_knn(train_data,test_data,train_label,test_label,d);
        acc(i) = pca_acc;
        fprintf("PCA dim=%d, Accuracy = %.4f\n",d,pca_acc);
    else
        lda_acc = lda_knn(train_data,test_data,train_label,test_label,d);
        acc(i) = lda_acc;
        fprintf("LDA dim=%d, Accuracy = %.4f\n",d,lda_acc);
    end
    i = i+1;
end

plot(dimension,acc,'LineWidth',1);
xlabel("降维后维度");
ylabel("预测准确率/%");






