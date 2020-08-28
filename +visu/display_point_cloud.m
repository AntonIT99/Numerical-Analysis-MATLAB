function display_point_cloud(database,text)

    if ~exist('text','var')
        text = '';
    end

    %-- Extract points that belong to each class
    X_train_c0 = database.X_train(:,database.Y_train==0);
    X_train_c1 = database.X_train(:,database.Y_train==1);
    
    %-- Display them
    figure; plot(X_train_c0(1,:),X_train_c0(2,:),'ob','linewidth',2);
    hold on; plot(X_train_c1(1,:),X_train_c1(2,:),'or','linewidth',2);
    title(text);

end
