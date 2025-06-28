function acc = pca_knn(train_data,test_data,train_label,test_label,n_components)
    %降维
    coeff = pca(train_data,'NumComponents', n_components);
    train_data_pca = train_data * coeff;
    test_data_pca = test_data * coeff;

    %分类
    mdl = fitcknn(train_data_pca,train_label,'NumNeighbors', 1);

    %预测性能
    pred = predict(mdl,test_data_pca);
    acc = sum(pred == test_label)/numel(test_label);
end
