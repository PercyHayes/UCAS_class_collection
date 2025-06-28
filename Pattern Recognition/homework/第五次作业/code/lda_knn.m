function acc = lda_knn(train_data,test_data,train_label,test_label,n_components)
    %降维
    %lda = fitcdiscr(train_data,train_label,"DiscrimType","pseudolinear","ScoreTransform","logit");
    %train_data_lda = predict(lda, train_data);
    %test_data_lda = predict(lda, test_data);
    % 计算类内散布矩阵和类间散布矩阵
    classes = unique(train_label);
    numClasses = length(classes);
    [~, m] = size(train_data);
    meanTotal = mean(train_data);
    
    Sw = zeros(m, m);
    Sb = zeros(m, m);

    for i = 1:numClasses
        Xi = train_data(train_label == classes(i), :);     
        meanClass = mean(Xi); %各类别均值向量
        Sw = Sw + (Xi - meanClass)' * (Xi - meanClass);
        Sb = Sb + size(Xi, 1) * (meanClass - meanTotal)' * (meanClass - meanTotal);
    end
    % 计算特征值和特征向量
    Sw_inv = pinv(Sw);
    %[V, D] = eig(Sw_inv * Sb);
    [V, D] = eig(Sb, Sw);
    % 按特征值降序排序特征向量
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    W = V(:, 1:n_components);

    train_data_lda = train_data * W;
    test_data_lda = test_data * W;

    %分类
    mdl = fitcknn(train_data_lda,train_label,'NumNeighbors', 1);
    %预测
    pred = predict(mdl,test_data_lda);
    acc = sum(pred == test_label)/numel(test_label);

end